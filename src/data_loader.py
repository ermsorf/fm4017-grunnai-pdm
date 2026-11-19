"""Shared data loading utilities for model training and evaluation."""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Iterable

from src.data_handling import select_data_files


@dataclass(frozen=True)
class RNNSplit:
    """A chronological split of windowed RNN inputs and targets."""

    inputs: torch.Tensor
    targets: torch.Tensor
    timestamps: pd.DatetimeIndex = field(
        default_factory=lambda: pd.DatetimeIndex([])
    )


@dataclass(frozen=True)
class RNNData:
    """Windowed RNN data and the files used to construct it."""

    train: RNNSplit
    validation: RNNSplit
    test: RNNSplit
    predictor_files: tuple[Path, ...]
    reference_files: tuple[Path, ...]


@dataclass(frozen=True)
class TabularSplit:
    """A chronological split of one-row-per-timestamp model inputs."""

    inputs: pd.DataFrame
    targets: pd.DataFrame


@dataclass(frozen=True)
class TabularData:
    """Tabular sensor data and the files used to construct it."""

    train: TabularSplit
    validation: TabularSplit
    test: TabularSplit
    predictor_files: tuple[Path, ...]
    reference_files: tuple[Path, ...]


def _read_sensor_file(
    path: Path,
    cache: dict[Path, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if cache is not None and path in cache:
        return cache[path]

    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} is missing the required 'timestamp' column")

    value_columns = [column for column in frame if column.endswith("value")]
    if len(value_columns) != 1:
        raise ValueError(f"{path} must contain exactly one '*_value' column")

    value_column = value_columns[0]
    prefix = value_column.removesuffix("_value")
    mask_column = prefix + "_mask"
    age_column = prefix + "_age"
    available_columns = ["timestamp", value_column]
    feature_names = ["timestamp", "value"]
    for column, feature_name in (
        (mask_column, "mask"),
        (age_column, "age"),
    ):
        if column in frame.columns:
            available_columns.append(column)
            feature_names.append(feature_name)

    result = frame[available_columns].copy()
    result.columns = feature_names
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    if cache is not None:
        cache[path] = result
    return result


def _load_selected_files(
    predictor_files: Iterable[str | Path],
    reference_files: Iterable[str | Path],
    predictor_features: tuple[str, ...],
    cache: dict[Path, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictor_paths = tuple(Path(path) for path in predictor_files)
    reference_paths = tuple(Path(path) for path in reference_files)
    if not predictor_paths:
        raise ValueError("At least one predictor file must be selected")
    if not reference_paths:
        raise ValueError("At least one reference file must be selected")

    predictor_frames = [_read_sensor_file(path, cache) for path in predictor_paths]
    reference_frames = [_read_sensor_file(path, cache) for path in reference_paths]
    predictor_columns = []
    predictor_column_names = []
    for path, frame in zip(predictor_paths, predictor_frames):
        available_features = [
            feature for feature in predictor_features if feature in frame.columns
        ]
        if not available_features:
            raise ValueError(
                f"{path} does not contain any requested predictor features"
            )
        predictor_columns.append(
            frame.set_index("timestamp")[available_features]
        )
        predictor_column_names.extend(
            f"{path.stem}_{feature}" for feature in available_features
        )
    predictors = pd.concat(
        predictor_columns,
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
    predictors.columns = predictor_column_names
    references.columns = [path.stem for path in reference_paths]
    return predictors, references


def _make_split(
    predictors: pd.DataFrame,
    references: pd.DataFrame,
    start: int,
    stop: int,
    sequence_length: int,
    exclusion_values: pd.DataFrame | None = None,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int = 0,
    exclude_after_samples: int = 0,
    exclude_before: str | pd.Timedelta | None = None,
    exclude_after: str | pd.Timedelta | None = None,
) -> RNNSplit:
    inputs = np.array(
        predictors.iloc[start:stop].to_numpy(dtype="float32"),
        dtype="float32",
        order="C",
        copy=True,
    )
    targets = np.array(
        references.iloc[start:stop].to_numpy(dtype="float32"),
        dtype="float32",
        order="C",
        copy=True,
    )
    if len(inputs) < sequence_length:
        empty_inputs = torch.empty((0, sequence_length, predictors.shape[1]), dtype=torch.float32)
        empty_targets = torch.empty((0, references.shape[1]), dtype=torch.float32)
        return RNNSplit(empty_inputs, empty_targets, pd.DatetimeIndex([]))

    selected_values = exclusion_values.iloc[start:stop] if exclusion_values is not None else None
    valid_rows = np.ones(len(inputs), dtype=bool)
    if selected_values is not None and exclude_below:
        for signal, minimum in exclude_below.items():
            values = selected_values[signal].to_numpy(dtype="float32")
            valid_rows &= values >= float(minimum)
        valid_rows = _apply_exclusion_buffers(
            valid_rows,
            predictors.index[start:stop],
            before_samples=exclude_before_samples,
            after_samples=exclude_after_samples,
            before=exclude_before,
            after=exclude_after,
        )

    window_count = len(inputs) - sequence_length + 1
    if np.all(valid_rows):
        valid_starts = None
    else:
        valid_starts = np.flatnonzero(
            np.lib.stride_tricks.sliding_window_view(
                valid_rows, sequence_length
            ).all(axis=1)
        )

    if valid_starts is not None and len(valid_starts) == 0:
        return RNNSplit(
            torch.empty((0, sequence_length, predictors.shape[1]), dtype=torch.float32),
            torch.empty((0, references.shape[1]), dtype=torch.float32),
            pd.DatetimeIndex([]),
        )

    input_tensor = torch.from_numpy(inputs)
    target_tensor = torch.from_numpy(targets)
    windowed_inputs = torch.as_strided(
        input_tensor,
        size=(window_count, sequence_length, inputs.shape[1]),
        stride=(inputs.shape[1], inputs.shape[1], 1),
    )
    windowed_targets = target_tensor[sequence_length - 1:]
    if valid_starts is not None:
        starts = torch.from_numpy(valid_starts).long()
        windowed_inputs = windowed_inputs.index_select(0, starts)
        windowed_targets = windowed_targets.index_select(0, starts)
        timestamp_positions = valid_starts + sequence_length - 1
    else:
        timestamp_positions = np.arange(sequence_length - 1, len(inputs))

    return RNNSplit(
        windowed_inputs,
        windowed_targets,
        pd.DatetimeIndex(references.index[start + timestamp_positions]),
    )


def _make_tabular_split(
    predictors: pd.DataFrame,
    references: pd.DataFrame,
    start: int,
    stop: int,
    exclusion_values: pd.DataFrame | None = None,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int = 0,
    exclude_after_samples: int = 0,
    exclude_before: str | pd.Timedelta | None = None,
    exclude_after: str | pd.Timedelta | None = None,
) -> TabularSplit:
    inputs = predictors.iloc[start:stop].copy()
    targets = references.iloc[start:stop].copy()
    if exclusion_values is not None and exclude_below:
        valid_rows = pd.Series(True, index=inputs.index)
        selected_values = exclusion_values.iloc[start:stop]
        for signal, minimum in exclude_below.items():
            valid_rows &= selected_values[signal] >= float(minimum)
        valid_rows = _apply_exclusion_buffers(
            valid_rows.to_numpy(),
            inputs.index,
            before_samples=exclude_before_samples,
            after_samples=exclude_after_samples,
            before=exclude_before,
            after=exclude_after,
        )
        inputs = inputs.loc[valid_rows]
        targets = targets.loc[valid_rows]
    return TabularSplit(inputs=inputs, targets=targets)


def _apply_exclusion_buffers(
    valid_rows: np.ndarray,
    timestamps: pd.Index,
    *,
    before_samples: int,
    after_samples: int,
    before: str | pd.Timedelta | None,
    after: str | pd.Timedelta | None,
) -> np.ndarray:
    """Expand invalid rows by asymmetric sample-count and time buffers."""
    if before_samples < 0 or after_samples < 0:
        raise ValueError("Exclusion sample buffers must be non-negative")
    before_delta = pd.Timedelta(0) if before is None else pd.Timedelta(before)
    after_delta = pd.Timedelta(0) if after is None else pd.Timedelta(after)
    if before_delta < pd.Timedelta(0) or after_delta < pd.Timedelta(0):
        raise ValueError("Exclusion time buffers must be non-negative")

    invalid = ~np.asarray(valid_rows, dtype=bool)
    expanded = invalid.copy()
    invalid_positions = np.flatnonzero(invalid)
    for position in invalid_positions:
        start = max(0, position - before_samples)
        stop = min(len(expanded), position + after_samples + 1)
        expanded[start:stop] = True

    if before_delta or after_delta:
        timestamp_index = pd.DatetimeIndex(timestamps)
        if not timestamp_index.is_monotonic_increasing:
            raise ValueError("Exclusion timestamps must be sorted")
        for position in invalid_positions:
            left = timestamp_index[position] - before_delta
            right = timestamp_index[position] + after_delta
            start = timestamp_index.searchsorted(left, side="left")
            stop = timestamp_index.searchsorted(right, side="right")
            expanded[start:stop] = True
    return ~expanded


def _load_exclusion_values(
    data_folder: str | Path,
    exclude_below: Mapping[str, float] | None,
    common_timestamps: pd.Index,
    cache: dict[Path, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Load threshold indicators without adding them to model inputs."""
    if not exclude_below:
        return None

    result = pd.DataFrame(index=common_timestamps)
    for signal in exclude_below:
        matching_files = select_data_files(data_folder, filename_contains=signal)
        if not matching_files:
            raise ValueError(f"No signal file matches exclusion threshold {signal!r}")

        values = pd.concat(
            [
                _read_sensor_file(path, cache).set_index("timestamp")["value"]
                for path in matching_files
            ],
            axis=1,
        ).reindex(common_timestamps).min(axis=1, skipna=False)
        normalization_path = Path(data_folder) / "normalization_parameters.json"
        if normalization_path.is_file():
            with normalization_path.open(encoding="utf-8") as file:
                parameters = json.load(file)
            file_parameters = [
                parameters.get(path.name)
                for path in matching_files
            ]
            if all(parameter is not None for parameter in file_parameters):
                # The indicator file values are normalized; thresholds remain
                # expressed in the original physical units.
                values = pd.concat(
                    [
                        _read_sensor_file(path, cache).set_index("timestamp")["value"]
                        * parameter["std"]
                        + parameter["mean"]
                        for path, parameter in zip(matching_files, file_parameters)
                    ],
                    axis=1,
                ).reindex(common_timestamps).min(axis=1, skipna=False)
        result[signal] = values
    return result


def _load_normalization_parameters(
    path: str | Path,
) -> dict[str, dict[str, float]]:
    normalization_path = Path(path)
    if not normalization_path.is_file():
        raise FileNotFoundError(
            f"Normalization parameters not found: {normalization_path}"
        )
    with normalization_path.open(encoding="utf-8") as file:
        parameters = json.load(file)
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Normalization parameters must be an object: {normalization_path}"
        )
    return parameters


def _convert_to_training_normalization(
    predictors: pd.DataFrame,
    references: pd.DataFrame,
    predictor_files: tuple[Path, ...],
    reference_files: tuple[Path, ...],
    source_normalization_parameters: Mapping[str, Mapping[str, float]],
    training_normalization_parameters: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert normalized values from another period to the training scale."""
    predictors = predictors.copy()
    references = references.copy()
    for path in (*predictor_files, *reference_files):
        source = source_normalization_parameters.get(path.name)
        training = training_normalization_parameters.get(path.name)
        if source is None or training is None:
            raise ValueError(f"Missing normalization parameters for {path.name}")
        source_std = float(source["std"])
        training_std = float(training["std"])
        if source_std <= 0 or training_std <= 0:
            raise ValueError(f"Normalization standard deviations must be positive for {path.name}")
        values = (
            predictors[path.stem + "_value"]
            if path in predictor_files
            else references[path.stem]
        )
        converted = (
            values * source_std + float(source["mean"]) - float(training["mean"])
        ) / training_std
        if path in predictor_files:
            predictors[path.stem + "_value"] = converted
        else:
            references[path.stem] = converted
    return predictors, references


def _validate_thresholds(
    exclude_below: Mapping[str, float] | None,
) -> None:
    if not exclude_below:
        return
    for signal, minimum in exclude_below.items():
        try:
            numeric_minimum = float(minimum)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Threshold for {signal!r} must be numeric") from error
        if not np.isfinite(numeric_minimum):
            raise ValueError(f"Threshold for {signal!r} must be finite")


def _validate_predictor_features(
    predictor_features: Iterable[str],
) -> tuple[str, ...]:
    features = tuple(predictor_features)
    supported_features = {"value", "mask", "age"}
    if not features:
        raise ValueError("predictor_features must contain at least one feature")
    if any(feature not in supported_features for feature in features):
        raise ValueError(
            "predictor_features may only contain 'value', 'mask', or 'age'"
        )
    if len(set(features)) != len(features):
        raise ValueError("predictor_features must not contain duplicates")
    return features


def load_rnn_data(
    data_folder: str | Path,
    *,
    validation_data_folder: str | Path | None = None,
    training_normalization_parameters: str | Path | None = None,
    predictor_contains: str | Iterable[str] | None = None,
    predictor_exclude: str | Iterable[str] | None = None,
    predictor_matches: str | None = None,
    reference_contains: str | Iterable[str] | None = None,
    reference_exclude: str | Iterable[str] | None = None,
    reference_matches: str | None = None,
    sequence_length: int = 10,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.195,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int = 0,
    exclude_after_samples: int = 0,
    exclude_before: str | pd.Timedelta | None = None,
    exclude_after: str | pd.Timedelta | None = None,
    predictor_features: Iterable[str] = ("value", "mask", "age"),
) -> RNNData:
    """Select sensor files and create train/validation/test windows.

    ``exclude_below`` maps a case-insensitive signal-file-name fragment to its
    minimum allowed value. ``exclude_before`` and ``exclude_after`` accept
    pandas duration strings such as ``"5min"`` or ``"2h"`` and expand each
    invalid period asymmetrically. Sample buffers can be supplied separately.
    Indicator files are loaded separately and are not included in the model
    inputs or targets. When ``validation_data_folder`` is
    provided, all of ``data_folder`` is used for training and all of the
    validation folder is used for validation; the test split is empty.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must leave a positive test split")
    _validate_thresholds(exclude_below)
    predictor_features = _validate_predictor_features(predictor_features)

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
    file_cache: dict[Path, pd.DataFrame] = {}
    predictors, references = _load_selected_files(
        predictor_files, reference_files, predictor_features, file_cache
    )
    exclusion_values = _load_exclusion_values(
        data_folder, exclude_below, predictors.index, file_cache
    )
    if validation_data_folder is not None:
        validation_folder = Path(validation_data_folder)
        validation_predictor_files = tuple(select_data_files(
            validation_folder,
            filename_contains=predictor_contains,
            filename_exclude=predictor_exclude,
            filename_matches=predictor_matches,
        ))
        validation_reference_files = tuple(select_data_files(
            validation_folder,
            filename_contains=reference_contains,
            filename_exclude=reference_exclude,
            filename_matches=reference_matches,
        ))
        if (
            [path.name for path in validation_predictor_files]
            != [path.name for path in predictor_files]
            or [path.name for path in validation_reference_files]
            != [path.name for path in reference_files]
        ):
            raise ValueError("Training and validation folders must contain the same selected sensor files")
        validation_file_cache: dict[Path, pd.DataFrame] = {}
        validation_predictors, validation_references = _load_selected_files(
            validation_predictor_files,
            validation_reference_files,
            predictor_features,
            validation_file_cache,
        )
        source_parameters = _load_normalization_parameters(
            validation_folder / "normalization_parameters.json"
        )
        training_parameters = _load_normalization_parameters(
            training_normalization_parameters
            or Path(data_folder) / "normalization_parameters.json"
        )
        validation_predictors, validation_references = _convert_to_training_normalization(
            validation_predictors,
            validation_references,
            predictor_files,
            reference_files,
            source_parameters,
            training_parameters,
        )
        validation_exclusion_values = _load_exclusion_values(
            validation_folder,
            exclude_below,
            validation_predictors.index,
            validation_file_cache,
        )
        return RNNData(
            train=_make_split(
                predictors, references, 0, len(predictors), sequence_length,
                exclusion_values, exclude_below, exclude_before_samples,
                exclude_after_samples, exclude_before, exclude_after
            ),
            validation=_make_split(
                validation_predictors, validation_references, 0,
                len(validation_predictors), sequence_length,
                validation_exclusion_values, exclude_below,
                exclude_before_samples, exclude_after_samples,
                exclude_before, exclude_after,
            ),
            test=RNNSplit(
                torch.empty((0, sequence_length, predictors.shape[1]), dtype=torch.float32),
                torch.empty((0, references.shape[1]), dtype=torch.float32),
                pd.DatetimeIndex([]),
            ),
            predictor_files=predictor_files,
            reference_files=reference_files,
        )
    split_train = int(len(predictors) * train_fraction)
    split_validation = split_train + int(len(predictors) * validation_fraction)
    return RNNData(
        train=_make_split(
            predictors, references, 0, split_train, sequence_length,
            exclusion_values, exclude_below, exclude_before_samples,
            exclude_after_samples, exclude_before, exclude_after
        ),
        validation=_make_split(
            predictors, references, split_train, split_validation, sequence_length,
            exclusion_values, exclude_below, exclude_before_samples,
            exclude_after_samples, exclude_before, exclude_after
        ),
        test=_make_split(
            predictors, references, split_validation, len(predictors), sequence_length,
            exclusion_values, exclude_below, exclude_before_samples,
            exclude_after_samples, exclude_before, exclude_after
        ),
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
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int = 0,
    exclude_after_samples: int = 0,
    exclude_before: str | pd.Timedelta | None = None,
    exclude_after: str | pd.Timedelta | None = None,
    predictor_features: Iterable[str] = ("value", "mask", "age"),
) -> RNNSplit:
    """Load all chronological windows from a separate evaluation dataset."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    _validate_thresholds(exclude_below)
    predictor_features = _validate_predictor_features(predictor_features)

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
    file_cache: dict[Path, pd.DataFrame] = {}
    predictors, references = _load_selected_files(
        predictor_files, reference_files, predictor_features, file_cache
    )
    exclusion_values = _load_exclusion_values(
        data_folder, exclude_below, predictors.index, file_cache
    )
    return _make_split(
        predictors, references, 0, len(predictors), sequence_length,
        exclusion_values, exclude_below, exclude_before_samples,
        exclude_after_samples, exclude_before, exclude_after
    )


def load_tabular_data(
    data_folder: str | Path,
    *,
    predictor_contains: str | Iterable[str] | None = None,
    predictor_exclude: str | Iterable[str] | None = None,
    predictor_matches: str | None = None,
    reference_contains: str | Iterable[str] | None = None,
    reference_exclude: str | Iterable[str] | None = None,
    reference_matches: str | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    exclude_below: Mapping[str, float] | None = None,
    predictor_features: Iterable[str] = ("value",),
) -> TabularData:
    """Select sensor files for non-recurrent, per-timestep prediction.

    Each input row contains the selected predictor features at one timestamp,
    and the target is the reference value at that same timestamp. Unlike
    :func:`load_rnn_data`, no windows or sequence dimension are created.
    """
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must leave a positive test split")
    _validate_thresholds(exclude_below)
    predictor_features = _validate_predictor_features(predictor_features)

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
    file_cache: dict[Path, pd.DataFrame] = {}
    predictors, references = _load_selected_files(
        predictor_files, reference_files, predictor_features, file_cache
    )
    exclusion_values = _load_exclusion_values(
        data_folder, exclude_below, predictors.index, file_cache
    )
    split_train = int(len(predictors) * train_fraction)
    split_validation = split_train + int(len(predictors) * validation_fraction)
    return TabularData(
        train=_make_tabular_split(
            predictors, references, 0, split_train, exclusion_values, exclude_below
        ),
        validation=_make_tabular_split(
            predictors, references, split_train, split_validation,
            exclusion_values, exclude_below,
        ),
        test=_make_tabular_split(
            predictors, references, split_validation, len(predictors),
            exclusion_values, exclude_below,
        ),
        predictor_files=predictor_files,
        reference_files=reference_files,
    )
