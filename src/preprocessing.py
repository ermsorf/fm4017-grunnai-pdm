### Preprocessing functions.
import shutil
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re

import os
import numpy as np
import pandas as pd
import src.config
from src.data_handling import load_signals, select_data_files

"""
Each sensor only stores a new sample when its value has changed by more than a specified threshold. As a result, the raw sensor data is irregularly sampled.
We want to convert this into a regular time series with one sample every 10 seconds. To do this, we assign each measured sample to the next 10-second timestamp (ceiling), and forward-fill that value until the next measured sample is recorded.

For example, the raw data:

2024-10-24 19:00:04.490, 0.4147472381591797
2024-10-24 19:00:33.790, 3.1029648780822754

becomes:

timestamp,               sensor_1_val,          sensor_1_mask, sensor_1_age
2024-10-24 19:00:10.000, 0.4147472381591797,    1,             0
2024-10-24 19:00:20.000, 0.4147472381591797,    0,             1
2024-10-24 19:00:30.000, 0.4147472381591797,    0,             2
2024-10-24 19:00:40.000, 3.1029648780822754,    1,             0


Here:

sensor_1_val is the measured value or the last known value carried forward.
sensor_1_mask is 1 when the value comes from an actual measurement assigned to that timestamp, and 0 when it is forward-filled.
sensor_1_age is the number of 10-second intervals since the last measured value. Thus, it is 0 for a measured value, 1 after one interval, 2 after two intervals, etc.

If multiple raw samples fall within the same 10-second interval, only the **last sample in that interval** is used. The samples are assigned to intervals using the ceiling timestamp. For example, samples occurring between 19:00:30 and 19:00:40 are assigned to the 19:00:40 timestamp, with the latest sample taking precedence.

The final dataframe has the following structure, where the sensor name is derived from the filename stem:

timestamp, sensor_1_val, sensor_1_mask, sensor_1_age
"""



def step_interpolate_files(
    file_paths: Iterable[str | Path],
    output_folder: str | Path | None = None,
    interval="10s",
    start_time="2024-06-02 00:00:00",
) -> list[Path]:
    """Convert irregular sensor files to regularly sampled, forward-filled signals.

    All output files start at start_time. The last measurement at or before
    start_time initializes forward-filling, but is not marked as a new
    measurement unless its timestamp equals start_time.
    """
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_time)
    written_files: list[Path] = []

    for file_path in file_paths:
        source = Path(file_path)
        signal = load_signals([source])[source]
        signal = signal.dropna(subset=["timestamp", "value"]).sort_values(
            "timestamp",
            kind="mergesort",
        )

        if signal.empty:
            raise ValueError(f"Signal file contains no valid samples: {source}")

        signal["timestamp"] = signal["timestamp"].dt.ceil(interval)
        signal = signal.drop_duplicates("timestamp", keep="last").set_index("timestamp")

        # Keep the last measurement at or before the requested start time.
        previous = signal.loc[signal.index <= start].tail(1)
        future = signal.loc[signal.index > start]

        if previous.empty:
            raise ValueError(
                f"No measurement at or before {start} for sensor: {source}"
            )

        end = future.index.max() if not future.empty else start

        timestamps = pd.date_range(
            start=start,
            end=end,
            freq=interval,
            name="timestamp",
        )

        # Combine the initial value with future measurements.
        combined = pd.concat([previous, future])
        combined = combined[~combined.index.duplicated(keep="last")]

        result = combined[["value"]].reindex(
            combined.index.union(timestamps)
        ).sort_index()

        measured = result["value"].notna()
        result["value"] = result["value"].ffill()
        result["mask"] = measured.astype("int64")
        result["age"] = (
            (~measured).groupby(measured.cumsum()).cumcount().astype("int64")
        )

        # Keep only the common timeline.
        result = result.reindex(timestamps)

        sensor_name = source.stem
        result = result.rename(
            columns={
                "value": f"{sensor_name}_value",
                "mask": f"{sensor_name}_mask",
                "age": f"{sensor_name}_age",
            }
        ).reset_index()

        destination = destination_folder / source.name
        result.to_csv(destination, index=False)
        written_files.append(destination)

    return written_files


def create_time_features(
    process_folder: str | Path,
    *,
    timestamp_source: str | Path | None = None,
) -> list[Path]:
    """Create cyclic day, week, and year signals from processed timestamps.

    The generated signals use the same ``timestamp``, ``*_value``,
    ``*_mask``, and ``*_age`` schema as step-interpolated sensor files.  Each
    cyclic component is written as its own file so it can be selected by the
    existing RNN data loader.
    """
    process_path = Path(process_folder)
    if not process_path.is_dir():
        raise FileNotFoundError(f"Process folder does not exist: {process_path}")

    if timestamp_source is not None:
        source_path = Path(timestamp_source)
        if not source_path.is_absolute():
            source_path = process_path / source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Timestamp source does not exist: {source_path}")
        candidate_paths = [source_path]
    else:
        candidate_paths = sorted(
            path
            for path in process_path.glob("*.csv")
            if not path.name.startswith("time_")
        )

    if not candidate_paths:
        raise ValueError(f"No CSV files available as a timestamp source in {process_path}")

    timestamps: pd.Series | None = None
    source_error: Exception | None = None
    for source_path in candidate_paths:
        try:
            frame = pd.read_csv(source_path, usecols=["timestamp"])
            parsed = pd.to_datetime(frame["timestamp"], errors="raise")
        except (OSError, ValueError, KeyError, TypeError) as error:
            source_error = error
            continue
        if parsed.empty:
            source_error = ValueError(f"Timestamp source contains no rows: {source_path}")
            continue
        timestamps = parsed
        break

    if timestamps is None:
        detail = f": {source_error}" if source_error is not None else ""
        raise ValueError(f"No CSV file contains a usable timestamp column{detail}")

    timestamp_index = pd.DatetimeIndex(timestamps)
    if timestamp_index.has_duplicates:
        raise ValueError("Timestamp source contains duplicate timestamps")
    if not timestamp_index.is_monotonic_increasing:
        raise ValueError("Timestamp source timestamps must be sorted")

    seconds_in_day = (
        timestamp_index.hour * 3600
        + timestamp_index.minute * 60
        + timestamp_index.second
        + timestamp_index.microsecond / 1_000_000
    )
    day_phase = seconds_in_day / (24 * 3600)
    week_phase = (
        timestamp_index.dayofweek + day_phase
    ) / 7
    year_phase = (
        timestamp_index.dayofyear - 1 + day_phase
    ) / np.where(timestamp_index.is_leap_year, 366, 365)

    phases = {
        "day": day_phase,
        "week": week_phase,
        "year": year_phase,
    }
    written_files: list[Path] = []
    for period, phase in phases.items():
        for component, values in (
            ("sin", np.sin(2 * np.pi * phase)),
            ("cos", np.cos(2 * np.pi * phase)),
        ):
            stem = f"time_{period}_{component}"
            output = pd.DataFrame({
                "timestamp": timestamp_index,
                f"{stem}_value": values,
                f"{stem}_mask": 1,
                f"{stem}_age": 0,
            })
            destination = process_path / f"{stem}.csv"
            output.to_csv(destination, index=False)
            written_files.append(destination)

    normalization_params_path = process_path / "normalization_parameters.json"
    if normalization_params_path.is_file():
        with normalization_params_path.open(encoding="utf-8") as file:
            normalization_parameters = json.load(file)
        if not isinstance(normalization_parameters, dict):
            raise ValueError(
                f"Normalization parameters must be an object: {normalization_params_path}"
            )
    else:
        normalization_parameters = {}

    for path in written_files:
        normalization_parameters[path.name] = {"mean": 0.0, "std": 1.0}

    with normalization_params_path.open("w", encoding="utf-8") as file:
        json.dump(normalization_parameters, file, indent=4)

    return written_files


def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose signal value is zero, negative, or invalid.

    The timestamp is intentionally not included in the comparison because
    pandas cannot compare datetime values with zero.
    """
    if "value" not in df.columns:
        raise ValueError("Expected a 'value' column in the signal dataframe")

    values = pd.to_numeric(df["value"], errors="coerce")
    return df.loc[values >= 0].copy()

def remove_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose signal value is zero, negative, or invalid.

    The timestamp is intentionally not included in the comparison because
    pandas cannot compare datetime values with zero.
    """
    if "value" not in df.columns:
        raise ValueError("Expected a 'value' column in the signal dataframe")

    values = pd.to_numeric(df["value"], errors="coerce")
    return df.loc[values > 0].copy()


def detect_outliers(
    values: pd.Series,
    *,
    window: int = 11,
    threshold: float = 6.0,
) -> pd.Series:
    """Detect isolated spikes with a rolling median absolute deviation.

    The returned boolean series has the same index as ``values``. A point is
    an outlier when its robust z-score exceeds ``threshold``. A zero-MAD
    window is treated as constant, so a non-constant point in that window is
    still detected.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer greater than or equal to 3")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    rolling_median = numeric.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).median()
    deviations = (numeric - rolling_median).abs()
    mad = deviations.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).median()

    robust_z = 0.6745 * (numeric - rolling_median) / mad.replace(0, np.nan)
    outliers = robust_z.abs().gt(threshold)
    constant_window_outliers = mad.eq(0) & deviations.gt(0)
    return (outliers | constant_window_outliers).fillna(False).astype(bool)


def smooth_signal(
    df: pd.DataFrame,
    value_column: str = "value",
    *,
    method: str = "median",
    window: int = 5,
    detect_spikes: bool = True,
    outlier_window: int = 11,
    outlier_threshold: float = 6.0,
) -> pd.DataFrame:
    """Optionally replace spikes and smooth one signal column.

    ``method`` can be ``"median"``, ``"mean"``, or ``"ewm"``. Spike
    replacement uses linear interpolation, which avoids pulling a rolling
    mean toward a large isolated value. Non-numeric values are left as NaN
    during processing and the input dataframe is never modified.
    """
    if value_column not in df.columns:
        raise ValueError(f"Expected a '{value_column}' column in the dataframe")
    if window < 1:
        raise ValueError("window must be greater than zero")
    if method not in {"median", "mean", "ewm"}:
        raise ValueError("method must be one of: 'median', 'mean', 'ewm'")

    result = df.copy()
    values = pd.to_numeric(result[value_column], errors="coerce").astype(float)
    if detect_spikes:
        spikes = detect_outliers(
            values,
            window=outlier_window,
            threshold=outlier_threshold,
        )
        values = values.mask(spikes).interpolate(limit_direction="both")

    if method == "median":
        smoothed = values.rolling(window=window, center=True, min_periods=1).median()
    elif method == "mean":
        smoothed = values.rolling(window=window, center=True, min_periods=1).mean()
    else:
        smoothed = values.ewm(span=window, adjust=False, min_periods=1).mean()

    result[value_column] = smoothed
    return result


def smooth_signal_files(
    input_paths: Iterable[str | Path],
    *,
    value_column: str = "value",
    output_folder: str | Path | None = None,
    method: str = "median",
    window: int = 5,
    detect_spikes: bool = True,
    outlier_window: int = 11,
    outlier_threshold: float = 6.0,
) -> list[Path]:
    """Smooth CSV files with a normal header and preserve their filenames."""
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for input_path in input_paths:
        source = Path(input_path)
        frame = pd.read_csv(source)
        if value_column in frame.columns:
            value_columns = [value_column]
        else:
            value_columns = [
                column for column in frame.columns if column.endswith("_value")
            ]
            if not value_columns:
                raise ValueError(
                    f"Expected a '{value_column}' column or a '*_value' column in {source}"
                )

        smoothed = frame.copy()
        for column in value_columns:
            smoothed = smooth_signal(
                smoothed,
                value_column=column,
                method=method,
                window=window,
                detect_spikes=detect_spikes,
                outlier_window=outlier_window,
                outlier_threshold=outlier_threshold,
            )
        destination = destination_folder / source.name
        smoothed.to_csv(destination, index=False)
        written_files.append(destination)
    return written_files

def remove_outliers_values(
    df: pd.DataFrame,
    *,
    window: int = 11,
    threshold: float = 6.0,
) -> pd.DataFrame:
    """Remove robustly detected outlier rows from a raw signal dataframe."""
    if "value" not in df.columns:
        raise ValueError("Expected a 'value' column in the signal dataframe")
    numeric = pd.to_numeric(df["value"], errors="coerce")
    outliers = detect_outliers(numeric, window=window, threshold=threshold)
    return df.loc[~outliers].copy()


def remove_outliers(
    input_paths: Iterable[str | Path],
    output_folder: str | Path | None = None,
    *,
    window: int = 11,
    threshold: float = 6.0,
) -> list[Path]:
    """Remove isolated spikes from selected raw signal CSV files."""
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for csv_file in input_paths:
        source = Path(csv_file)
        signal = remove_outliers_values(
            load_signals([source])[source],
            window=window,
            threshold=threshold,
        )
        destination = destination_folder / source.name
        signal.to_csv(destination, index=False)
        written_files.append(destination)
    return written_files


def combine_nozzle_openings(
    input_paths: Iterable[str | Path],
    output_folder: str | Path | None = None,
    *,
    output_name: str = "Scada.GRUN.AGG2.TURB2.NALER.M_NAL_MEAN.csv",
) -> Path:
    """Combine NALER signals into one mean opening signal.

    Nozzle files are independently sampled. At each timestamp where any
    nozzle changes, each nozzle's latest known opening is carried forward and
    the mean is calculated across the readings available so far. Missing
    readings are not treated as closed nozzles.
    """
    sources = [Path(path) for path in input_paths]
    if not sources:
        raise ValueError("At least one nozzle signal file is required")

    signals: list[pd.Series] = []
    for source in sources:
        signal = load_signals([source])[source].dropna(
            subset=["timestamp", "value"]
        )
        signal = signal.sort_values("timestamp", kind="mergesort")
        signal = signal.drop_duplicates("timestamp", keep="last")
        if signal.empty:
            raise ValueError(f"No valid samples found in nozzle signal: {source}")
        values = signal.set_index("timestamp")["value"].rename(source.stem)
        signals.append(values)

    aligned = pd.concat(signals, axis=1, sort=False).sort_index().ffill()
    combined = aligned.mean(axis=1, skipna=True).dropna()
    if combined.empty:
        raise ValueError("No overlapping valid samples found in nozzle signals")

    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = destination_folder / output_name
    pd.DataFrame({
        "timestamp": combined.index,
        "value": combined.to_numpy(),
    }).to_csv(destination, index=False)
    return destination


def create_naler_boolean_files(
    input_paths: Iterable[str | Path],
    output_folder: str | Path,
    *,
    threshold: float = 5.0,
) -> list[Path]:
    """Create enabled/disabled signals from NALER opening measurements.

    A value at or above ``threshold`` is enabled (``1``); a value below it is
    disabled (``0``). The output uses ``synth.NALER.NAL<n>_BOOL.csv`` names
    and the standard ``timestamp,value`` signal schema.
    """
    if not np.isfinite(threshold):
        raise ValueError("NALER boolean threshold must be finite")

    destination_folder = Path(output_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for csv_file in input_paths:
        source = Path(csv_file)
        match = re.search(r"(?:^|[._])NAL(\d+)(?:POS|_BOOL)?$", source.stem)
        if match is None:
            raise ValueError(
                f"Could not determine NALER number from filename: {source.name}"
            )

        signal = load_signals([source])[source].dropna(
            subset=["timestamp", "value"]
        )
        if signal.empty:
            raise ValueError(f"No valid samples found in NALER signal: {source}")

        signal = signal.sort_values("timestamp", kind="mergesort")
        signal = signal.drop_duplicates("timestamp", keep="last")
        signal["value"] = (signal["value"] >= threshold).astype("int64")

        destination = destination_folder / f"synth.NALER.NAL{match.group(1)}_BOOL.csv"
        signal.to_csv(destination, index=False)
        written_files.append(destination)

    return written_files


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fill and standardize columns ending in ``_value``.

    This keeps the dataframe-oriented API used by the example training
    scripts. Use :func:`smooth_signal` before this step when the input has
    isolated spikes.
    """
    if data.empty:
        return data.copy()

    result = data.copy()
    value_columns = [
        column
        for column in result.columns
        if pd.to_numeric(result[column], errors="coerce").notna().any()
    ]
    for column in value_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.interpolate(limit_direction="both")
        result[column] = values
        mean = values.mean()
        std = values.std()
        scale = std if std and np.isfinite(std) else 1.0
        result[f"normalized_{column}"] = (values - mean) / scale
    return result.ffill().bfill()


def remove_negative(
    input_paths: Iterable[str | Path],
    output_folder: str | Path | None = None,
) -> list[Path]:
    """Clean selected signal CSV files and write their original filenames."""
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    written_files = []
    for csv_file in input_paths:
        csv_file = Path(csv_file)
        signal = remove_negative_values(load_signals([csv_file])[csv_file])
        destination = destination_folder / csv_file.name
        signal.to_csv(destination, index=False)
        written_files.append(destination)

    return written_files

def remove_zeros(
    input_paths: Iterable[str | Path],
    output_folder: str | Path | None = None,
) -> list[Path]:
    """Clean selected signal CSV files and write their original filenames."""
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    written_files = []
    for csv_file in input_paths:
        csv_file = Path(csv_file)
        signal = remove_zero_values(load_signals([csv_file])[csv_file])
        destination = destination_folder / csv_file.name
        signal.to_csv(destination, index=False)
        written_files.append(destination)

    return written_files


def z_normalize(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Z-normalize the specified value column in the dataframe."""
    if value_column not in df.columns:
        raise ValueError(f"Expected a '{value_column}' column in the dataframe")

    mean = df[value_column].mean()
    std = df[value_column].std()
    df[value_column] = (df[value_column] - mean) / std
    params = {"mean": mean, "std": std}
    return df, params

def z_normalize_files(
    input_paths: Iterable[str | Path],
    value_column: str,
    output_folder: str | Path | None = None,
) -> list[Path]:
    """Z-normalize the specified value column in selected signal CSV files."""
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    normalization_parameters = {}

    written_files = []
    for csv_file in input_paths:
        csv_file = Path(csv_file)
        signal, params = z_normalize(load_signals([csv_file])[csv_file], value_column)
        normalization_parameters[csv_file.name] = params
        destination = destination_folder / csv_file.name
        signal.to_csv(destination, index=False)
        written_files.append(destination)
    normalization_params_path = destination_folder / "normalization_parameters.json"
    with open(normalization_params_path, "w") as f:
        import json
        json.dump(normalization_parameters, f, indent=4)
    return written_files


def normalize_nozzle_files(
    input_paths: Iterable[str | Path],
    output_folder: str | Path | None = None,
) -> list[Path]:
    """Scale nozzle openings from their physical 0-100 range to 0-1.

    Unlike z-normalization, this uses the same fixed scale for every nozzle so
    equal physical openings have equal model inputs. The fixed scale is also
    recorded in the shared normalization metadata for period conversion.
    """
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

    normalization_params_path = destination_folder / "normalization_parameters.json"
    if normalization_params_path.is_file():
        with normalization_params_path.open(encoding="utf-8") as file:
            normalization_parameters = json.load(file)
        if not isinstance(normalization_parameters, dict):
            raise ValueError(
                f"Normalization parameters must be an object: {normalization_params_path}"
            )
    else:
        normalization_parameters = {}

    written_files = []
    for csv_file in input_paths:
        csv_file = Path(csv_file)
        signal = load_signals([csv_file])[csv_file]
        signal["value"] = signal["value"] / 100.0
        normalization_parameters[csv_file.name] = {"mean": 0.0, "std": 100.0}
        destination = destination_folder / csv_file.name
        signal.to_csv(destination, index=False)
        written_files.append(destination)

    with normalization_params_path.open("w", encoding="utf-8") as file:
        json.dump(normalization_parameters, file, indent=4)
    return written_files



if __name__ == "__main__":

    print("Starting preprocessing...")
    process_folder = os.path.join(src.config.DATA_PROCESSED, "2025-06-02_2025-10-31_2NI")
    csv_files = select_data_files(
        src.config.DATA_RAW_2025,
        filename_contains=["AGG2", "Hydrocord"],
        filename_exclude=["Hydrocord.Eta"],
    )
    for csv_file in csv_files:
        shutil.copy(csv_file, os.path.join(process_folder, csv_file.name))

    files_remove_negative = select_data_files(
        process_folder,
        filename_contains=["AGG2", "Hydrocord"],
    )
    print("Removing negative")
    remove_negative(
        files_remove_negative,
        output_folder=process_folder,
    )
    files_remove_zeros = select_data_files(
        process_folder,
        filename_contains=["AGG2", "Hydrocord"],
        filename_exclude=["VIBR", "NALER"],
    )
    print("Removing zeros")
    remove_zeros(
        files_remove_zeros,
        output_folder=process_folder,
    )
    files = select_data_files(
        process_folder,
        filename_contains=["AGG2", "Hydrocord"],
    )

    # print("Removing outliers.")
    # remove_outliers(
    #     files,
    #     output_folder=process_folder,
    #     window=11,
    #     threshold=5.0,
    # )
    naler_files = select_data_files(
        process_folder,
        filename_contains=["NALER"],
    )
    print("Creating NALER boolean files.")
    create_naler_boolean_files(
        naler_files,
        output_folder=process_folder,
        threshold=5.0,
    )

    files = select_data_files(
        process_folder,
        filename_contains=["AGG2", "Hydrocord"],
    )
    print("Step-interpolating.")
    files = select_data_files(
        process_folder,
        filename_contains=["AGG2", "Hydrocord", "synth"],
    )
    step_interpolate_files(files, output_folder=process_folder, interval="2s", start_time="2025-06-02 00:00:00")

    print("Z-normalization.")
    z_normalize_files(
        select_data_files(
            process_folder,
            filename_contains=["AGG2", "Hydrocord"],
            filename_exclude=["NALER"],
        ),
        value_column="value",
        output_folder=process_folder,
    )
    normalize_nozzle_files(
        select_data_files(
            process_folder,
            filename_contains=["NALER"],
        ),
        output_folder=process_folder,
    )



    # print("Smoothing after step interpolation.")
    # smooth_signal_files(
    #     files,
    #     output_folder=process_folder,
    #     method="median",
    #     window=11,
    #     detect_spikes=True,
    # )

    # print("Creating time features.")
    # create_time_features(process_folder)



    print("Preprocessing Complete.")





    