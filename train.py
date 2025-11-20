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

warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")


DATASET = "biosignal"  # "manufacturing" or "biosignal"

WANDB_MODE = "online"
TEST_NAME = f"{DATASET}"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-large"
DEVICE = "cuda"
BATCH_SIZE = 16
DATALOADER_SHUFFLE = True
LLAMA_LEARNING_RATE = 1e-6
MOMENT_LEARNING_RATE = 1e-6
EPOCHS = 10
MAX_GRAD_NORM = 5.0
OUTPUT_DIR = f"./outputs/{TEST_NAME}"
CONTEXT_LENGTH = 12
STRIDE = 12 if DATASET == "biosignal" else 1
LLAMA_ACCUMULATE_STEPS = 2
MOMENT_ACCUMULATE_STEPS = 2

task_name = "classification" if DATASET == "biosignal" else "forecasting"


def train_llama_model(model, dataloader, optimizer, loss_fn, task_name, scheduler, scaler):
    model.train()
    losses = []
    accumulate_losses = 0.0
    accumulate_steps = 0
    for i, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            llama_outputs = model.forward_llama(reports=batch["patch_report"])
            if task_name == "classification":
                targets = batch["y"].long().squeeze(-1)
            elif task_name == "forecasting":
                targets = batch["y_scaled"].float()
            loss = loss_fn(llama_outputs, targets)
        accumulate_losses += loss
        accumulate_steps += 1
        if i % LLAMA_ACCUMULATE_STEPS == 0 or i == len(dataloader) - 1:
            accumulate_losses = accumulate_losses / accumulate_steps
            scaler.scale(accumulate_losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.llama_head.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            losses.append(accumulate_losses.item())
            wandb.log(
                {
                    "train_llama_loss": accumulate_losses.item(),
                    "llama_lr": scheduler.get_last_lr()[0],
                }
            )
            accumulate_losses = 0.0
            accumulate_steps = 0
    return sum(losses) / len(losses)


def train_moment_model(model, dataloader, optimizer, loss_fn, task_name, scheduler, scaler):
    model.train()
    losses = []
    accumulate_losses = 0.0
    accumulate_steps = 0
    for i, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            moment_outputs = model.forward_moment(timeseries=batch["x_scaled"])
            if task_name == "classification":
                moment_outputs = moment_outputs.logits
                targets = batch["y_scaled"].long().squeeze(-1)
            elif task_name == "forecasting":
                moment_outputs = moment_outputs.forecast[:, -1]
                targets = batch["y_scaled"].float()
            loss = loss_fn(moment_outputs, targets)
        accumulate_losses += loss
        accumulate_steps += 1
        if i % MOMENT_ACCUMULATE_STEPS == 0 or i == len(dataloader) - 1:
            accumulate_losses = accumulate_losses / accumulate_steps
            scaler.scale(accumulate_losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.moment_model.head.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            losses.append(accumulate_losses.item())
            wandb.log(
                {
                    "train_moment_loss": accumulate_losses.item(),
                    "moment_lr": scheduler.get_last_lr()[0],
                }
            )
            accumulate_losses = 0.0
            accumulate_steps = 0
    return sum(losses) / len(losses)


@torch.no_grad()
def eval_model(model, dataloader, loss_fn, task_name):
    model.eval()
    softvote_losses = []
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
                llama_probs = nn.functional.softmax(llama_outputs, dim=-1)

                moment_loss = loss_fn(moment_outputs.logits, targets)
                moment_probs = nn.functional.softmax(moment_outputs.logits, dim=-1)

                assert llama_probs.shape == moment_probs.shape
                logits = llama_outputs + moment_outputs.logits
                logits = logits / 2
                loss = loss_fn(logits, targets)

            elif task_name == "forecasting":
                targets_scaled = batch["y_scaled"].float()

                llama_pred = llama_outputs
                llama_loss = loss_fn(llama_pred, targets_scaled)

                moment_pred = moment_outputs.forecast[:, -1]
                moment_loss = loss_fn(moment_pred, targets_scaled)

                pred = moment_pred + llama_pred
                pred = pred / 2
                loss = loss_fn(pred, targets_scaled)
        softvote_losses.append(loss.item())
        moment_losses.append(moment_loss.item())
        llama_losses.append(llama_loss.item())
    wandb.log(
        {
            "eval_softvote_loss": sum(softvote_losses) / len(softvote_losses),
            "eval_moment_loss": sum(moment_losses) / len(moment_losses),
            "eval_llama_loss": sum(llama_losses) / len(llama_losses),
        }
    )
    return sum(softvote_losses) / len(softvote_losses)


def save(epoch, model, llama_optimizer, moment_optimizer, llama_scheduler, moment_scheduler):
    output_dir = Path(OUTPUT_DIR) / f"epoch_{epoch+1}"
    model.save(output_dir)
    # save optmizer
    torch.save(llama_optimizer.state_dict(), output_dir / "llama_optimizer.pth")
    torch.save(moment_optimizer.state_dict(), output_dir / "moment_optimizer.pth")
    # scheduler state
    torch.save(llama_scheduler.state_dict(), output_dir / "llama_scheduler.pth")
    torch.save(moment_scheduler.state_dict(), output_dir / "moment_scheduler.pth")


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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=DATALOADER_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=DATALOADER_SHUFFLE)

    model = BTSModel(
        llama_model_name=LLAMA_MODEL_NAME,
        moment_model_name=MOMENT_MODEL_NAME,
        task_name=task_name,
        device=DEVICE,
        moment_reduction_method="mean",
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
    llama_num_update_steps_per_epoch = math.ceil(len(train_dataloader) / LLAMA_ACCUMULATE_STEPS)
    llama_total_steps = llama_num_update_steps_per_epoch * EPOCHS

    llama_optimizer = torch.optim.AdamW(model.llama_head.parameters(), lr=LLAMA_LEARNING_RATE)

    llama_scheduler = get_scheduler(
        # name='constant_with_warmup',
        name="linear",
        optimizer=llama_optimizer,
        num_training_steps=llama_total_steps,
        num_warmup_steps=llama_num_update_steps_per_epoch,
    )

    moment_optimizer = torch.optim.Adam(model.moment_model.head.parameters(), lr=MOMENT_LEARNING_RATE)
    max_lr = 1e-4
    moment_num_update_steps_per_epoch = math.ceil(len(train_dataloader) / MOMENT_ACCUMULATE_STEPS)
    moment_total_steps = moment_num_update_steps_per_epoch * EPOCHS

    moment_scheduler = OneCycleLR(
        moment_optimizer,
        max_lr=max_lr,
        total_steps=moment_total_steps,
        pct_start=0.3,
    )

    for epoch in range(EPOCHS):
        wandb.log({"epoch": epoch + 1})
        print(f"Epoch {epoch + 1}/{EPOCHS} - Training Llama Model")
        train_llama_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=llama_optimizer,
            loss_fn=loss_fn,
            task_name=task_name,
            scaler=scaler,
            scheduler=llama_scheduler,
        )

        print(f"Epoch {epoch + 1}/{EPOCHS} - Training Moment Model")
        train_moment_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=moment_optimizer,
            loss_fn=loss_fn,
            task_name=task_name,
            scaler=scaler,
            scheduler=moment_scheduler,
        )
        wandb.log({"moment_lr": moment_scheduler.get_last_lr()[0]})

        save(
            epoch,
            model,
            llama_optimizer,
            moment_optimizer,
            llama_scheduler,
            moment_scheduler,
        )
        eval_model(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            task_name=task_name,
        )


if __name__ == "__main__":
    main()
