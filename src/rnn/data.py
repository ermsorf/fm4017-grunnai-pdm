import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data_handling import select_data_files


@dataclass(frozen=True)
class RNNSplit:
    """A chronological split of windowed RNN inputs and targets."""

    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class RNNData:
    """Windowed RNN data and the files used to construct it."""

    train: RNNSplit
    validation: RNNSplit
    test: RNNSplit
    predictor_files: tuple[Path, ...]
    reference_files: tuple[Path, ...]


def _read_sensor_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} is missing the required 'timestamp' column")

    value_columns = [column for column in frame if column.endswith("_value")]
    if len(value_columns) != 1:
        raise ValueError(f"{path} must contain exactly one '*_value' column")

    value_column = value_columns[0]
    prefix = value_column.removesuffix("_value")
    mask_column = prefix + "_mask"
    age_column = prefix + "_age"
    required_columns = {value_column, mask_column, age_column}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    result = frame[["timestamp", value_column, mask_column, age_column]].copy()
    result.columns = ["timestamp", "value", "mask", "age"]
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    return result


def _load_selected_files(
    predictor_files: Iterable[str | Path],
    reference_files: Iterable[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictor_paths = tuple(Path(path) for path in predictor_files)
    reference_paths = tuple(Path(path) for path in reference_files)
    if not predictor_paths:
        raise ValueError("At least one predictor file must be selected")
    if not reference_paths:
        raise ValueError("At least one reference file must be selected")

    predictor_frames = [_read_sensor_file(path) for path in predictor_paths]
    reference_frames = [_read_sensor_file(path) for path in reference_paths]
    predictors = pd.concat(
        [frame.set_index("timestamp")[["value", "mask", "age"]] for frame in predictor_frames],
        axis=1,
        join="inner",
    )
    references = pd.concat(
        [frame.set_index("timestamp")[["value"]] for frame in reference_frames],
        axis=1,
        join="inner",
    )
    if predictors.empty or references.empty:
        raise ValueError("Selected files have no timestamps in common")
    common_timestamps = predictors.index.intersection(references.index)
    predictors = predictors.loc[common_timestamps]
    references = references.loc[common_timestamps]
    predictors.columns = [
        f"{path.stem}_{suffix}"
        for path in predictor_paths
        for suffix in ("value", "mask", "age")
    ]
    references.columns = [path.stem for path in reference_paths]
    return predictors, references


def _make_split(
    predictors: pd.DataFrame,
    references: pd.DataFrame,
    start: int,
    stop: int,
    sequence_length: int,
) -> RNNSplit:
    inputs = predictors.iloc[start:stop].to_numpy(dtype="float32")
    targets = references.iloc[start:stop].to_numpy(dtype="float32")
    if len(inputs) < sequence_length:
        empty_inputs = torch.empty((0, sequence_length, predictors.shape[1]), dtype=torch.float32)
        empty_targets = torch.empty((0, references.shape[1]), dtype=torch.float32)
        return RNNSplit(empty_inputs, empty_targets)

    windows = [inputs[index:index + sequence_length] for index in range(len(inputs) - sequence_length + 1)]
    window_targets = [targets[index + sequence_length - 1] for index in range(len(targets) - sequence_length + 1)]
    return RNNSplit(
        torch.from_numpy(np.stack(windows)),
        torch.from_numpy(np.stack(window_targets)),
    )


def load_rnn_data(
    data_folder: str | Path,
    *,
    predictor_contains: str | Iterable[str] | None = None,
    predictor_exclude: str | Iterable[str] | None = None,
    predictor_matches: str | None = None,
    reference_contains: str | Iterable[str] | None = None,
    reference_exclude: str | Iterable[str] | None = None,
    reference_matches: str | None = None,
    sequence_length: int = 10,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> RNNData:
    """Select sensor files and create chronological train/validation/test windows."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must leave a positive test split")

    predictor_files = tuple(select_data_files(
        data_folder,
        filename_contains=predictor_contains,
        filename_exclude=predictor_exclude,
        filename_matches=predictor_matches,
    ))
    reference_files = tuple(select_data_files(
        data_folder,
        filename_contains=reference_contains,
        filename_exclude=reference_exclude,
        filename_matches=reference_matches,
    ))
    predictors, references = _load_selected_files(predictor_files, reference_files)
    split_train = int(len(predictors) * train_fraction)
    split_validation = split_train + int(len(predictors) * validation_fraction)
    return RNNData(
        train=_make_split(predictors, references, 0, split_train, sequence_length),
        validation=_make_split(predictors, references, split_train, split_validation, sequence_length),
        test=_make_split(predictors, references, split_validation, len(predictors), sequence_length),
        predictor_files=predictor_files,
        reference_files=reference_files,
    )


def load_rnn_evaluation_data(
    data_folder: str | Path,
    *,
    predictor_contains: str | Iterable[str] | None = None,
    predictor_exclude: str | Iterable[str] | None = None,
    predictor_matches: str | None = None,
    reference_contains: str | Iterable[str] | None = None,
    reference_exclude: str | Iterable[str] | None = None,
    reference_matches: str | None = None,
    sequence_length: int = 10,
) -> RNNSplit:
    """Load all chronological windows from a separate evaluation dataset."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    predictor_files = tuple(select_data_files(
        data_folder,
        filename_contains=predictor_contains,
        filename_exclude=predictor_exclude,
        filename_matches=predictor_matches,
    ))
    reference_files = tuple(select_data_files(
        data_folder,
        filename_contains=reference_contains,
        filename_exclude=reference_exclude,
        filename_matches=reference_matches,
    ))
    predictors, references = _load_selected_files(predictor_files, reference_files)
    return _make_split(predictors, references, 0, len(predictors), sequence_length)
