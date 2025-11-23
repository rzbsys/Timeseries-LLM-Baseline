from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model import BTSModel
from src.dataset import init_dataset, collate_fn
from src.utils import to_device, set_seed


DATASET = "biosignal"  # "manufacturing" or "biosignal"
# CHECKPOINT_PATH = "./outputs.old/manufacturing_ensemble_experiment.overfit/epoch_3"
CHECKPOINT_PATH = "./outputs.old/biosignal_ensemble_experiment/epoch_1"
OUTPUT_DIR = f"./outputs/eval_{DATASET}_ensemble1"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-small"
CONTEXT_LENGTH = 12
STRIDE = 12 if DATASET == "biosignal" else 1
BATCH_SIZE = 16
DATALOADER_SHUFFLE = True
DEVICE = "cuda:0"
NUM_CHANNELS = 19 if DATASET == "biosignal" else 24
task_name = "classification" if DATASET == "biosignal" else "forecasting"
if isinstance(OUTPUT_DIR, str):
    OUTPUT_DIR = Path(OUTPUT_DIR)
if isinstance(CHECKPOINT_PATH, str):
    CHECKPOINT_PATH = Path(CHECKPOINT_PATH)


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
    return {
        "avg_loss": sum(total_losses) / len(total_losses),
        "avg_accuracy": sum(accuracy_list) / len(accuracy_list) if accuracy_list else None,
    }


def main():

    if OUTPUT_DIR.exists():
        raise ValueError(f"Output directory {OUTPUT_DIR} already exists.")
    if not CHECKPOINT_PATH.exists():
        raise ValueError(f"Checkpoint path {CHECKPOINT_PATH} does not exist.")

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenizer_collate_fn = lambda batch: collate_fn(tokenizer, batch)
    test_dataset = init_dataset(
        dataset_name=DATASET,
        split="test",
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
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
        task_name=task_name,
        device=DEVICE,
        num_channels=NUM_CHANNELS,
        moment_reduction_method="concat",
    )
    model.to(DEVICE)
    model.print_num_trainable_parameters()
    model.load(CHECKPOINT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sigmas = [0.0, 0.05, 0.1, 1, 3, 5]
    reversed(sigmas)
    results = []
    for sigma in sigmas:
        print(f"Evaluating with random sigma: {sigma}")
        if DATASET == "biosignal":
            for ds in test_dataset.datasets:
                ds.config.random_sigma = sigma
        else:
            test_dataset.config.random_sigma = sigma

        metrics = eval_model(
            model=model,
            dataloader=test_dataloader,
            loss_fn=nn.CrossEntropyLoss() if task_name == "classification" else nn.MSELoss(),
            task_name=task_name,
        )

        results.append(metrics)
        print(f"Sigma: {sigma}, Metrics: {metrics}")

    with open(OUTPUT_DIR / "eval_metrics.txt", "w") as f:
        for metric in metrics:
            f.write(f"{metric}\n")
    print("Evaluation metrics:")
    for metric in metrics:
        print(metric)


if __name__ == "__main__":
    main()
