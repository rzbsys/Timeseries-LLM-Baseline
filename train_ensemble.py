from tqdm import tqdm
from pathlib import Path
import math

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_scheduler, AutoTokenizer


from src.model import BTSModel
from src.dataset import init_dataset, collate_fn
from src.utils import to_device, set_seed

import sys

import warnings

warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")

dataset = sys.argv[1]

DATASET = dataset  # "manufacturing" or "biosignal"

WANDB_MODE = "online"
TEST_NAME = f"{DATASET}_ensemble_experiment_flash_attention2"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-large"
DEVICE = "cuda"
BATCH_SIZE = 16
DATALOADER_SHUFFLE = True
LEARNING_RATE = 1e-6
EPOCHS = 10
MAX_GRAD_NORM = 5.0
OUTPUT_DIR = f"./outputs/{TEST_NAME}"
CONTEXT_LENGTH = 12
STRIDE = 12 if DATASET == "biosignal" else 1
ACCUMULATE_STEPS = 2
NUM_CHANNELS = 19 if DATASET == "biosignal" else 24

task_name = "classification" if DATASET == "biosignal" else "forecasting"


def train_model(model, dataloader, optimizer, loss_fn, task_name, scheduler, scaler):
    model.train()
    accumulate_losses = 0.0
    accumulate_steps = 0

    for i, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            moment_outputs = model.forward_moment(timeseries=batch["x_scaled"])
            llama_outputs = model.forward_llama(llama_inputs=batch["patch_report_tokenized"])
            if task_name == "classification":
                targets = batch["y"].long().squeeze(-1)
            elif task_name == "forecasting":
                targets = batch["y_scaled"].float()

            outputs = model.combine_outputs(llama_outputs, moment_outputs)
            loss = loss_fn(outputs, targets)

            accumulate_losses += loss
            accumulate_steps += 1

        if i % ACCUMULATE_STEPS == 0 or i == len(dataloader) - 1:
            accumulate_losses = accumulate_losses / accumulate_steps

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
            scaler.step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            wandb.log(
                {
                    "train/total_loss": accumulate_losses.item(),
                    "lr": scheduler.get_last_lr()[0],
                },
            )
            accumulate_losses = 0.0
            accumulate_steps = 0


@torch.no_grad()
def eval_model(model, dataloader, loss_fn, task_name):
    model.eval()
    total_losses = []
    accuracy_list = []
    for batch in tqdm(dataloader):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            moment_outputs = model.forward_moment(timeseries=batch["x_scaled"])
            llama_outputs = model.forward_llama(llama_inputs=batch["patch_report_tokenized"])
            if task_name == "classification":
                targets = batch["y"].long().squeeze(-1)
            elif task_name == "forecasting":
                targets = batch["y_scaled"].float()

            pred = model.combine_outputs(llama_outputs, moment_outputs)
            loss = loss_fn(pred, targets)
            if task_name == "classification":
                accuracy = (pred.argmax(dim=-1) == targets).float().mean()
                accuracy_list.append(accuracy.item())
        total_losses.append(loss.item())
    wandb.log(
        {
            "eval/total_loss": sum(total_losses) / len(total_losses),
            "eval/accuracy": sum(accuracy_list) / len(accuracy_list) if task_name == "classification" else 0.0,
        }
    )


def save(epoch, model: BTSModel, optimizer, scheduler):
    output_dir = Path(OUTPUT_DIR) / f"epoch_{epoch+1}"
    model.save(output_dir)
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")
    torch.save(scheduler.state_dict(), output_dir / "scheduler.pth")


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
        # prefetch_factor=3,
        # num_workers=1,
        pin_memory=True,
        collate_fn=tokenizer_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=DATALOADER_SHUFFLE,
        # prefetch_factor=3,
        # num_workers=1,
        pin_memory=True,
        collate_fn=tokenizer_collate_fn,
    )
    model = BTSModel(
        llama_model_name=LLAMA_MODEL_NAME,
        moment_model_name=MOMENT_MODEL_NAME,
        task_name=task_name,
        device=DEVICE,
        num_channels=NUM_CHANNELS,
        moment_reduction_method="concat",
    )
    model.to(DEVICE)
    model.print_num_trainable_parameters()

    loss_fn = None
    if task_name == "classification":
        loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    elif task_name == "forecasting":
        loss_fn = nn.MSELoss().to(DEVICE)
    else:
        raise ValueError("Unsupported task name")

    steps_per_epoch = math.ceil(len(train_dataloader) / ACCUMULATE_STEPS)
    total_steps = steps_per_epoch * EPOCHS

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    max_lr = 1e-5
    total_steps = math.ceil(len(train_dataloader) / ACCUMULATE_STEPS) * EPOCHS
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

    # scheduler = get_scheduler(
    #     # name='constant_with_warmup',
    #     name="linear",
    #     optimizer=optimizer,
    #     num_training_steps=total_steps,
    #     num_warmup_steps=steps_per_epoch,
    # )

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
            scheduler=scheduler,
        )

        save(
            epoch,
            model,
            optimizer,
            scheduler,
        )

        eval_model(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            task_name=task_name,
        )


if __name__ == "__main__":
    main()
