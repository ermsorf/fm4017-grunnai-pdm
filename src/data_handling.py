from collections.abc import Iterable
from pathlib import Path
import re

import pandas as pd


def _as_keywords(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def select_data_files(
    data_path: str | Path,
    *,
    filename_contains: str | Iterable[str] | None = None,
    filename_exclude: str | Iterable[str] | None = None,
    filename_matches: str | re.Pattern[str] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Returns a list of CSV files in a directory that match the given filters.

    Include keywords use OR semantics. Exclude keywords and the regular
    expression are applied after inclusion, and all of those filters must
    pass. A file path is treated as a one-file selection and is still checked
    against the supplied filters.
    """
    path = Path(data_path)
    if path.is_file():
        candidates = [path] if path.suffix.lower() == ".csv" else []
    elif path.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        candidates = sorted(path.glob(pattern))
    else:
        raise FileNotFoundError(f"Data path does not exist: {path}")

    includes = _as_keywords(filename_contains)
    excludes = _as_keywords(filename_exclude)
    matcher = re.compile(filename_matches) if isinstance(filename_matches, str) else filename_matches

    return [
        file_path
        for file_path in candidates
        if (not includes or any(keyword in file_path.name for keyword in includes))
        and not any(keyword in file_path.name for keyword in excludes)
        and (matcher is None or matcher.search(file_path.name) is not None)
    ]


def load_signal(file_path: str | Path) -> pd.DataFrame:
    """Load one signal CSV into normalized timestamp/value columns."""
    signal = pd.read_csv(
        file_path,
        header=None,
        skiprows=2,
        names=["timestamp", "value"],
    )
    signal["timestamp"] = pd.to_datetime(signal["timestamp"], errors="coerce")
    signal["value"] = pd.to_numeric(signal["value"], errors="coerce")
    return signal


def load_signals(file_paths: Iterable[str | Path]) -> dict[Path, pd.DataFrame]:
    """Load selected signal files once, keyed by their paths."""
    return {
        Path(file_path): load_signal(file_path)
        for file_path in file_paths
    }
