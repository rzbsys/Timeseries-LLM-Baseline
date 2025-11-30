from tqdm import tqdm
from pathlib import Path
import math

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import kornia

from src.model import BTSModel
from src.dataset import init_dataset, collate_fn
from src.utils import to_device, set_seed

import sys

import warnings

# TODO: Warning을 해결해보아요.
warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message="X does not have valid feature name")

dataset = sys.argv[1]

DATASET = dataset  # "manufacturing" or "biosignal"

WANDB_MODE = "online"
TEST_NAME = f"{DATASET}_ensemble_experiment_trm"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-small"
DEVICE = "cuda"
BATCH_SIZE = 32
DATALOADER_SHUFFLE = True
LEARNING_RATE = 5e-5
EPOCHS = 20
MAX_GRAD_NORM = 5.0
OUTPUT_DIR = f"./outputs/{TEST_NAME}"
CONTEXT_LENGTH = 12
# STRIDE = 12 if DATASET == "biosignal" else 1
STRIDE = 12
ACCUMULATE_STEPS = 1
NUM_CHANNELS = 19 if DATASET == "biosignal" else 25

task_name = "classification" if DATASET == "biosignal" else "forecasting"


def train_model(
    model: BTSModel,
    dataloader: DataLoader,
    optimizer,
    loss_fn,
    task_name: str,
    scaler,
):
    model.train()
    accumulate_losses = 0.0
    accumulate_steps = 0

    for i, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            model_outputs = model(
                llama_inputs=batch["patch_report_tokenized"],
                timeseries=batch["x_scaled"],
            )
            targets = batch["y"].long() if task_name == "classification" else batch["y_scaled"].float()
            loss = loss_fn(model_outputs, targets)

            accumulate_losses += loss
            accumulate_steps += 1

        if i % ACCUMULATE_STEPS == 0 or i == len(dataloader) - 1:
            accumulate_losses = accumulate_losses / accumulate_steps
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            wandb.log(
                {
                    "train/total_loss": accumulate_losses.item(),
                },
            )
            accumulate_losses = 0.0
            accumulate_steps = 0


@torch.no_grad()
def eval_model(model: BTSModel, dataloader, loss_fn, task_name):
    model.eval()
    total_losses = []
    accuracy_list = []
    for batch in tqdm(dataloader):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            model_outputs = model(
                llama_inputs=batch["patch_report_tokenized"],
                timeseries=batch["x_scaled"],
            )
            targets = batch["y"].long() if task_name == "classification" else batch["y_scaled"].float()

            loss = loss_fn(model_outputs, targets)
            if task_name == "classification":
                accuracy = (model_outputs.argmax(dim=-1) == targets).float().mean()
                accuracy_list.append(accuracy.item())
        total_losses.append(loss.item())
    wandb.log(
        {
            "eval/total_loss": sum(total_losses) / len(total_losses),
            "eval/accuracy": sum(accuracy_list) / len(accuracy_list) if task_name == "classification" else 0.0,
        }
    )


def save(epoch, model: BTSModel, optimizer):
    output_dir = Path(OUTPUT_DIR) / f"epoch_{epoch+1}"
    model.save(output_dir)
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")


def main():
    if Path(OUTPUT_DIR).exists():
        raise ValueError(f"Output directory {OUTPUT_DIR} already exists.")

    wandb.init(
        mode=WANDB_MODE,
        project="bts_model_training",
        name=f"BTSModel_{DATASET}",
        config={
            "dataset": DATASET,
            "llama_model": LLAMA_MODEL_NAME,
            "moment_model": MOMENT_MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "task_name": task_name,
        },
    )

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenizer_collate_fn = lambda batch: collate_fn(tokenizer, batch)
    train_dataset = init_dataset(
        dataset_name=DATASET,
        split="train",
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )

    test_dataset = init_dataset(
        dataset_name=DATASET,
        split="test",
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )
    scaler = torch.amp.GradScaler(device=DEVICE)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=DATALOADER_SHUFFLE,
        pin_memory=True,
        collate_fn=tokenizer_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=DATALOADER_SHUFFLE,
        pin_memory=True,
        collate_fn=tokenizer_collate_fn,
    )
    model = BTSModel(
        llama_model_name=LLAMA_MODEL_NAME,
        moment_model_name=MOMENT_MODEL_NAME,
        n_classes=2 if task_name == "classification" else 1,
        n_channels=NUM_CHANNELS,
        head_type="mlp",
    )
    model.to(DEVICE)
    # model.print_num_trainable_parameters()
    if task_name == "classification":
        loss_fn = kornia.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        wandb.log({"epoch": epoch + 1})
        print(f"Epoch {epoch + 1}/{EPOCHS} - Training Llama Model")

        train_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            task_name=task_name,
            scaler=scaler,
        )

        save(
            epoch,
            model,
            optimizer,
        )

        eval_model(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            task_name=task_name,
        )


if __name__ == "__main__":
    main()
