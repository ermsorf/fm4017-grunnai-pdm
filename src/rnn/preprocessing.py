### Preprocessing functions.
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import src.config
from src.data_loader import load_signals, select_data_files

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


def step_interpolate_files(file_paths: Iterable[str | Path], output_folder: str | Path | None = None) -> list[Path]:
    """Convert irregular sensor files to ten-second, forward-filled signals.

    Each input file is expected to contain the raw ``timestamp`` and ``value``
    columns produced by :func:`src.data_loader.load_signal`. Samples are
    assigned to the ceiling of their ten-second interval; when several samples
    share an interval, the last one in timestamp order is retained.
    """
    destination_folder = (
        Path(output_folder) if output_folder is not None else Path(__file__).parent / "data"
    )
    destination_folder.mkdir(parents=True, exist_ok=True)

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

        signal["timestamp"] = signal["timestamp"].dt.ceil("10s")
        signal = signal.drop_duplicates("timestamp", keep="last").set_index("timestamp")

        timestamps = pd.date_range(
            start=signal.index.min(),
            end=signal.index.max(),
            freq="10s",
            name="timestamp",
        )
        result = signal[["value"]].reindex(timestamps)
        measured = result["value"].notna()
        result["value"] = result["value"].ffill()
        result["mask"] = measured.astype("int64")
        result["age"] = (
            (~measured).groupby(measured.cumsum()).cumcount().astype("int64")
        )

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








if __name__ == "__main__":
    
    csv_files = select_data_files(
        src.config.DATA_RAW_2024,
        filename_contains=["AGG2", "Hydrocord"],
        filename_exclude=["Hydrocord.Eta"],
    )
    for csv_file in csv_files:
        shutil.copy(csv_file, Path(__file__).parent / "data" / csv_file.name)

    files_remove_negative = select_data_files(
        Path(__file__).parent / "data",
        filename_contains=["AGG2", "Hydrocord"],
    )

    remove_negative(
        files_remove_negative,
        output_folder=Path(__file__).parent / "data",
    )
    files_remove_zeros = select_data_files(
        Path(__file__).parent / "data",
        filename_contains=["AGG2", "Hydrocord"],
        filename_exclude=["VIBR", "NALER"],
    )
    remove_zeros(
        files_remove_zeros,
        output_folder=Path(__file__).parent / "data",
    )
    print("Done removing negative and zero values from selected CSV files.")

    files = select_data_files(
        Path(__file__).parent / "data",
        filename_contains=["AGG2", "Hydrocord"],
    )
    step_interpolate_files(files, output_folder=Path(__file__).parent / "data" / "step_interp_10s")
    print("Done step-interpolating selected CSV files to 10-second intervals.")
    





    