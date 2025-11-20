from tqdm import tqdm
from pathlib import Path
import math

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_scheduler


from src.model import BTSModel
from src.dataset import init_dataset
from src.utils import to_device, set_seed

import warnings

warnings.filterwarnings(
    "ignore", "None of the inputs have requires_grad=True. Gradients will be None"
)


DATASET = "biosignal"  # "manufacturing" or "biosignal"

WANDB_MODE = "online"
TEST_NAME = f"{DATASET}_ensemble_experiment"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
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
ACCUMULATE_STEPS = 4
NUM_CHANNELS = 19 if DATASET == "biosignal" else 24

task_name = "classification" if DATASET == "biosignal" else "forecasting"


def train_model(model, dataloader, optimizer, loss_fn, task_name, scheduler, scaler):
    model.train()
    losses = []
    accumulate_losses = 0.0
    accumulate_steps = 0

    accumulate_llama_losses = 0.0
    accumulate_moment_losses = 0.0
    for i, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            moment_outputs = model.forward_moment(timeseries=batch["x_scaled"])
            llama_outputs = model.forward_llama(reports=batch["patch_report"])
            if task_name == "classification":
                moment_outputs = moment_outputs.logits
                # B, 2
                targets = batch["y"].long().squeeze(-1)
                outputs = llama_outputs + moment_outputs
            elif task_name == "forecasting":
                # B, 1
                moment_outputs = moment_outputs.forecast[:, -1]
                targets = batch["y_scaled"].float()
                outputs = (llama_outputs + moment_outputs) / 2
            loss = loss_fn(outputs, targets)
            accumulate_losses += loss
            accumulate_steps += 1
            with torch.no_grad():
                accumulate_llama_losses += loss_fn(moment_outputs, targets)
                accumulate_moment_losses += loss_fn(llama_outputs, targets)

        if i % ACCUMULATE_STEPS == 0 or i == len(dataloader) - 1:
            accumulate_losses = accumulate_losses / accumulate_steps
            accumulate_llama_losses = accumulate_llama_losses / accumulate_steps
            accumulate_moment_losses = accumulate_moment_losses / accumulate_steps

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            parameters = list(model.llama_head.parameters()) + list(
                model.moment_model.head.parameters()
            )
            torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
            scaler.step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            losses.append(accumulate_losses.item())
            wandb.log(
                {
                    "train/total_loss": accumulate_losses.item(),
                    "train/llama_loss": accumulate_llama_losses.item(),
                    "train/moment_loss": accumulate_moment_losses.item(),
                    "moment_lr": scheduler.get_last_lr()[0],
                },
            )
            accumulate_losses = 0.0
            accumulate_llama_losses = 0.0
            accumulate_moment_losses = 0.0
            accumulate_steps = 0


@torch.no_grad()
def eval_model(model, dataloader, loss_fn, task_name):
    model.eval()
    total_losses = []
    moment_losses = []
    llama_losses = []
    for batch in tqdm(dataloader):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            llama_outputs = model.forward_llama(reports=batch["patch_report"])
            moment_outputs = model.forward_moment(timeseries=batch["x_scaled"])
            if task_name == "classification":
                targets = batch["y"].long().squeeze(-1)

                llama_loss = loss_fn(llama_outputs, targets)
                moment_loss = loss_fn(moment_outputs.logits, targets)
                logits = (llama_outputs + moment_outputs.logits) / 2
                loss = loss_fn(logits, targets)

            elif task_name == "forecasting":
                targets_scaled = batch["y_scaled"].float()

                llama_pred = llama_outputs
                llama_loss = loss_fn(llama_pred, targets_scaled)

                moment_pred = moment_outputs.forecast[:, -1]
                moment_loss = loss_fn(moment_pred, targets_scaled)

                pred = (moment_pred + llama_pred) / 2
                loss = loss_fn(pred, targets_scaled)
        total_losses.append(loss.item())
        moment_losses.append(moment_loss.item())
        llama_losses.append(llama_loss.item())
    wandb.log(
        {
            "eval/softvote_loss": sum(total_losses) / len(total_losses),
            "eval/moment_loss": sum(moment_losses) / len(moment_losses),
            "eval/llama_loss": sum(llama_losses) / len(llama_losses),
        }
    )


def save(epoch, model, optimizer, scheduler):
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
        prefetch_factor=3,
        num_workers=1,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=DATALOADER_SHUFFLE,
        prefetch_factor=3,
        num_workers=1,
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
        loss_fn = nn.CrossEntropyLoss()
    elif task_name == "forecasting":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Unsupported task name")

    steps_per_epoch = math.ceil(len(train_dataloader) / ACCUMULATE_STEPS)
    total_steps = steps_per_epoch * EPOCHS

    model_parameters = list(model.llama_head.parameters()) + list(
        model.moment_model.head.parameters()
    )
    optimizer = torch.optim.Adam(model_parameters, lr=LEARNING_RATE)

    scheduler = get_scheduler(
        # name='constant_with_warmup',
        name="linear",
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=steps_per_epoch,
    )

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
