from pathlib import Path
from tqdm import tqdm

import torch

from .dataset import BTSTimeSeriesDataset, DatasetConfig, collate_fn
from .spec import *


MANUFATURING_DATASET = "/home/rzbsys/nas/CodeGen2025/Codes/Minguk/bts_dataset/data/manufacturing/SAMYANG.csv"
BIOSIGNAL_DATASET = "/home/rzbsys/nas/CodeGen2025/Codes/Minguk/bts_dataset/data/biosignal/sub/"


def init_dataset(
    dataset_name: str,
    split: str = "train",
    context_length: int = 12,
    stride: int = 1,
    random_sigma: float = 0.0,
) -> BTSTimeSeriesDataset:
    if dataset_name == "manufacturing":
        dataset_config = DatasetConfig(
            dataset_path=MANUFATURING_DATASET,
            column_to_train=SAMYANG_DEFAULT_TRAIN_COLUMNS,
            column_to_predict=SAMYANG_DEFAULT_TARGET_COLUMNS,
            preprocess_fns=SAMYANG_DEFAULT_PREPROCESS_FNS,
            data_kwargs={
                "parse_dates": [
                    "TimeStamp",
                ]
            },
            mode=split,
            context_length=context_length,
            shuffle=False,
            stride=stride,
            x_standardization=True,
            y_standardization=True,
            patch_validate_fns=SAMYANG_DEFAULT_PATCH_VALIDATE_FNS,
            y_offset=1,
            patch_report_fn=SAMYANG_DEFAULT_REPORT_FN,
            train_ratio=0.8,
            val_ratio=0.0,
            test_ratio=0.2,
            random_sigma=random_sigma,
        )
        dataset = BTSTimeSeriesDataset(config=dataset_config)
    elif dataset_name == "biosignal":
        dataset_config = DatasetConfig(
            dataset_path=BIOSIGNAL_DATASET,
            column_to_train=BIOSIGNAL_DEFAULT_TRAIN_COLUMNS,
            column_to_predict=BIOSIGNAL_DEFAULT_TARGET_COLUMNS,
            preprocess_fns=BIOSIGNAL_DEFAULT_PREPROCESS_FNS,
            mode=split,
            context_length=context_length,
            stride=stride,
            x_standardization=True,
            y_standardization=True,
            patch_validate_fns=BIOSIGNAL_DEFAULT_PATCH_VALIDATE_FNS,
            verbose=False,
            shuffle=True,
            patch_report_fn=BIOSIGNAL_DEFAULT_REPORT_FN,
            train_ratio=0.8,
            val_ratio=0.0,
            test_ratio=0.2,
            seed=42,
            random_sigma=random_sigma,
        )
        dataset = BTSTimeSeriesDataset(config=dataset_config)
    else:
        raise ValueError("Invalid dataset_name value. Choose 'manufacturing' or 'biosignal'.")

    return dataset


__all__ = [
    "init_dataset",
]
