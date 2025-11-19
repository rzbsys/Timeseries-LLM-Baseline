from typing import Any, List, Callable
import os

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.contrib.concurrent import thread_map


# @lru_cache(maxsize=3)
def load_dataset(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".pkl":
        df = pd.read_pickle(path, **kwargs)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return df


def split_data(data: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, y_horizen: int = 1, y_offset: int = 0) -> tuple[List[int], List[int], List[int]]:
    n = len(data) - y_horizen - y_offset + 1
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data


def standardize_data(data: pd.DataFrame, target_to_fit: pd.DataFrame, vars) -> pd.DataFrame:
    scaler = StandardScaler().fit(target_to_fit)
    data_copy = data.copy()
    data_copy.loc[:, vars] = scaler.transform(data[vars].values)
    return data_copy, scaler


def calc_valid_indices(
    dataset: pd.DataFrame,
    context_length: int,
    stride: int,
    validate_fns: List[Callable],
    parallel: bool = False,
    verbose: bool = True,
) -> list[list[int]]:
    start_idx = np.arange(0, len(dataset) - context_length + 1, stride, dtype=np.int64)
    valid_indices = []
    skip_cnt = 0

    def _check(start: int) -> tuple[int, bool]:
        end = start + context_length
        df = dataset.iloc[start:end]
        results = True if not validate_fns else all(fn(df) for fn in validate_fns)
        return start, end, results

    if parallel:
        workers = max(1, os.cpu_count() or 1)
        results = thread_map(_check, start_idx, max_workers=workers, desc=f"Calculating valid indices ({workers} workers)")
        for start, end, is_valid in results:
            if is_valid:
                indicates = list(range(start, end))
                valid_indices.append(indicates)
            else:
                skip_cnt += 1
    else:
        for start in tqdm(start_idx, desc="Calculating valid indices", disable=not verbose):
            start, end, results = _check(start)
            if results:
                indicates = list(range(start, end))
                valid_indices.append(indicates)
            else:
                skip_cnt += 1
    if verbose:
        print(f"Skipped {skip_cnt} sequences due to validation functions.")
    return valid_indices


def create_data_format(messages: List[str], roles: List[str]):
    assert len(messages) == len(roles), "Messages and roles must have the same length."
    data = []
    for msg, role in zip(messages, roles):
        data.append({"role": role, "content": msg})
    return data


def tokenize(tokenizer: Any, texts: List[str], max_length: int) -> dict[str, Any]:

    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
