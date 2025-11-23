from typing import Callable, List, Dict, Literal, Union, Any, Optional
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import pandas as pd
from dataclasses import dataclass
import random

from src.dataset.utils import *
from src.model.llama.utils import tokenize, create_data_format

GENERATE_RESULT_PROMPT = """You are an expert time-series prediction system.
Your input is a natural-language report describing one or more time series.
Identify and leverage column (feature) details and inter-column relationships mentioned in the report to interpret patterns (trend, seasonality, anomalies, missing data, etc.), then produce a binary prediction based on the report's evidence.

1. Extract time granularity, key variables, units, aggregation rules, baselines/thresholds from the report.
2. If causal/correlational relationships between columns are described, use them as evidence.
3. Analyze this report and summarize the features for prediction."""


@dataclass
class DatasetConfig:
    dataset_path: Union[str, Path]
    dataset_name: str = "BTSTimeSeriesDataset"
    data_kwargs: Dict = None
    preprocess_fns: List[Callable] = None
    column_to_train: List[str] = None
    column_to_predict: str = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    x_standardization: bool = True
    y_standardization: bool = True
    mode: Literal["train", "val", "test"] = "train"
    context_length: int = 12
    stride: int = 1
    y_horizon: int = 1
    y_offset: int = 0
    patch_validate_fns: List[Callable] = None
    skip_validate: bool = False
    verbose: bool = True
    patch_report_fn: Callable = None
    shuffle: bool = True
    seed: int = 42
    load_dataset: bool = True
    generate_system_prompt: str = GENERATE_RESULT_PROMPT
    random_sigma: float = 0.0

    def __post_init__(self):
        assert self.mode in ["train", "val", "test"], "Mode must be 'train', 'val', or 'test'."
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        else:
            self.dataset_path = self.dataset_path
        assert self.dataset_path.exists(), f"Dataset path {self.dataset_path} does not exist."
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
        assert self.column_to_predict is not None, "Target column must be specified."
        assert len(self.column_to_train) > 0, "At least one column to extract must be specified."
        # assert self.column_to_predict not in self.column_to_train, "Target column should not be in columns to extract."
        assert self.y_horizon > 0 and self.y_offset >= 0, "y_horizon must be positive and y_offset must be non-negative."

        if self.preprocess_fns is None:
            self.preprocess_fns = []
        if self.data_kwargs is None:
            self.data_kwargs = {}
        if self.patch_validate_fns is None:
            self.patch_validate_fns = []


class BTSTimeSeriesDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.scaler_x = None
        self.scaler_y = None
        self.rnd_state = random.Random(self.config.seed)

        # CSV 읽는 부분
        if config.load_dataset:
            self.origin_data = load_dataset(self.config.dataset_path, **self.config.data_kwargs)
        else:
            return

        # 전처리 함수 하나씩 실행시키는 부분
        preprocess_data = self.origin_data.copy()
        for fn in tqdm(self.config.preprocess_fns, desc="Preprocessing Data", total=len(self.config.preprocess_fns), disable=not self.config.verbose):
            preprocess_data = fn(preprocess_data)
        preprocess_data = preprocess_data.reset_index(drop=True)

        # skip idx 계산(y가 valid하지 않았을 때를 고려하기 위함임)
        if not self.config.skip_validate:
            self.__valid_indices = calc_valid_indices(
                preprocess_data,
                context_length=self.config.context_length,
                stride=self.config.stride,
                validate_fns=self.config.patch_validate_fns,
                verbose=self.config.verbose,
                y_horizon=self.config.y_horizon,
                y_offset=self.config.y_offset,
            )
        else:
            max_valid_indicate = len(preprocess_data) - self.config.context_length + 1 - self.config.y_offset - self.config.y_horizon + 1
            self.__valid_indices = [list(range(i, i + self.config.context_length)) for i in range(0, max_valid_indicate, self.config.stride)]
        if self.config.shuffle:
            self.rnd_state.shuffle(self.__valid_indices)
        train_data, val_data, test_data = split_data(self.__valid_indices, train_ratio=self.config.train_ratio, val_ratio=self.config.val_ratio, test_ratio=self.config.test_ratio)
        self.dataset = preprocess_data.copy()

        # 데이터 분할 및 표준화 부분
        train_index = [idx for indices in train_data for idx in indices]
        if self.config.x_standardization:
            # 우선 train data에 fit 시켜야해서 분할 먼저
            column_to_train_wo_target = [col for col in self.config.column_to_train if col != self.config.column_to_predict]
            train_x = preprocess_data.iloc[train_index][column_to_train_wo_target].values
            preprocess_data, scaler_x = standardize_data(preprocess_data, train_x, column_to_train_wo_target)
            self.scaler_x = scaler_x

        if self.config.y_standardization:
            train_y = preprocess_data.iloc[train_index][[self.config.column_to_predict]].values
            preprocess_data, scaler_y = standardize_data(preprocess_data, train_y, [self.config.column_to_predict])
            self.scaler_y = scaler_y

        if self.config.x_standardization or self.config.y_standardization:
            cols = set([*self.config.column_to_train, self.config.column_to_predict])
            self.scaled_dataset = preprocess_data[list(cols)]
        else:
            self.scaled_dataset = None

        # 데이터셋 선택 부분
        if self.config.mode == "train":
            self.indicates = train_data
        elif self.config.mode == "val":
            self.indicates = val_data
        else:
            self.indicates = test_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_indicates = self.indicates[idx]

        y_min = x_indicates[-1] + self.config.y_offset
        y_max = y_min + self.config.y_horizon

        y_indicates = list(range(y_min, y_max))

        x = torch.tensor(self.dataset.iloc[x_indicates][self.config.column_to_train].values).T.float()
        y = torch.tensor(self.dataset.iloc[y_indicates][self.config.column_to_predict].values).T.float()

        if self.config.random_sigma > 0.0:
            random_noise = torch.randn_like(x) * self.config.random_sigma
            x += random_noise

        output = {"x": x, "y": y, "x_columns": self.config.column_to_train, "y_column": self.config.column_to_predict}

        patch_report = None
        if self.config.patch_report_fn is not None:
            sliced_data = self.dataset.iloc[x_indicates].copy()
            if self.config.random_sigma > 0.0:
                sliced_data = torch.tensor(sliced_data[self.config.column_to_train].values).T.float() + random_noise
                sliced_data = pd.DataFrame(sliced_data.T.numpy(), columns=self.config.column_to_train)
            patch_report = self.config.patch_report_fn(sliced_data)
            output["patch_report"] = patch_report
        if self.config.x_standardization:
            x_s = torch.tensor(self.scaled_dataset.iloc[x_indicates][self.config.column_to_train].values).T.float()
            if self.config.random_sigma > 0.0:
                x_s += random_noise
            output["x_scaled"] = x_s
        if self.config.y_standardization:
            y_s = torch.tensor(self.scaled_dataset.iloc[y_indicates][self.config.column_to_predict].values).T.float()

            output["y_scaled"] = y_s
        return output

    def __len__(self) -> int:
        return len(self.indicates)


def collate_fn(tokenizer, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            collated[key] = [item[key] for item in batch]

    if "patch_report" in collated:
        messages = [
            create_data_format(
                [GENERATE_RESULT_PROMPT, report],
                roles=["system", "user"],
            )
            for report in collated["patch_report"]
        ]

        patch_report_tokenized = tokenize(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        collated["patch_report_tokenized"] = dict(patch_report_tokenized)
    return collated


class ConcatDataets(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.dataset_sizes = [len(ds) for ds in datasets]

    def __len__(self) -> int:
        return sum(self.dataset_sizes)

    def __getitem__(self, idx: int) -> Any:
        for ds_idx, size in enumerate(self.dataset_sizes):
            if idx < size:
                return self.datasets[ds_idx][idx]
            idx -= size
        raise IndexError("Index out of range")
