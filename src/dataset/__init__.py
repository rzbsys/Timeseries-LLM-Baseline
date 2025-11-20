from pathlib import Path
from tqdm import tqdm

import torch

from .dataset import BTSTimeSeriesDataset, DatasetConfig, ConcatDataets
from .spec import *


MANUFATURING_DATASET = "/home/rzbsys/nas/CodeGen2025/Codes/Minguk/bts_dataset/data/manufacturing/SAMYANG.csv"
BIOSIGNAL_DATASET = "/home/rzbsys/nas/CodeGen2025/Codes/Minguk/bts_dataset/data/biosignal/sub/"


def init_dataset(dataset_name: str, split: str = "train", context_length: int = 12, stride: int = 1) -> BTSTimeSeriesDataset:
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
            stride=stride,
            x_standardization=True,
            y_standardization=True,
            patch_validate_fns=SAMYANG_DEFAULT_PATCH_VALIDATE_FNS,
            y_offset=1,
            patch_report_fn=SAMYANG_DEFAULT_REPORT_FN,
            train_ratio=0.8,
            val_ratio=0.0,
            test_ratio=0.2,
        )
        dataset = BTSTimeSeriesDataset(config=dataset_config)
    elif dataset_name == "biosignal":
        dataset_pathes = list(Path(BIOSIGNAL_DATASET).glob("*"))
        dataset_pathes = [path for path in dataset_pathes if "sub1_" in path.name]
        # dataset_pathes = [path for path in dataset_pathes if 20 <= int(path.name.split("_trial")[-1].split(".")[0]) <= 40]

        datasets = []

        for dataset_path in tqdm(dataset_pathes):
            dataset_config = DatasetConfig(
                dataset_path=dataset_path,
                column_to_train=BIOSIGNAL_DEFAULT_TRAIN_COLUMNS,
                column_to_predict=BIOSIGNAL_DEFAULT_TARGET_COLUMNS,
                preprocess_fns=BIOSIGNAL_DEFAULT_PREPROCESS_FNS,
                mode=split,
                context_length=context_length,
                stride=stride,
                x_standardization=True,
                y_standardization=True,
                patch_validate_fns=BIOSIGNAL_DEFAULT_PATCH_VALIDATE_FNS,
                # skip_validate=True,
                verbose=False,
                patch_report_fn=BIOSIGNAL_DEFAULT_REPORT_FN,
                train_ratio=0.8,
                val_ratio=0.0,
                test_ratio=0.2,
            )

            ds = BTSTimeSeriesDataset(config=dataset_config)
            datasets.append(ds)

        dataset = ConcatDataets(datasets)
    else:
        raise ValueError("Invalid dataset_name value. Choose 'manufacturing' or 'biosignal'.")

    return dataset


__all__ = [
    "init_dataset",
]
