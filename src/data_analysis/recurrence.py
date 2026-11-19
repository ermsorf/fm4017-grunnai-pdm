"""Non-training diagnostics for choosing a recurrent-model history length.

The functions in this module measure whether observations at earlier lags still
contain information about the present.  They are diagnostics, not forecasts:
they cannot establish that an RNN will perform well, but they can show when a
longer sequence is unlikely to add useful information.

Lag zero means the current observation.  Therefore, for a model whose window
ends at the target timestamp, ``lag=1`` is the first genuinely historical
observation.  This matches the window construction in :mod:`src.data_loader`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data_loader import RNNData, load_rnn_data


def _as_numeric_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    result = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if result.empty:
        raise ValueError(f"{name} contains no finite numeric observations")
    return result.astype(float)


def _paired_values(
    left: pd.Series,
    right: pd.Series,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError("lag must be non-negative")
    left_values = _as_numeric_series(left, "left")
    right_values = _as_numeric_series(right, "right")
    left_values, right_values = left_values.align(right_values, join="inner")
    if lag:
        left_values = left_values.iloc[:-lag]
        right_values = right_values.iloc[lag:]
    size = min(len(left_values), len(right_values))
    if size == 0:
        return np.array([]), np.array([])
    return left_values.iloc[:size].to_numpy(), right_values.iloc[:size].to_numpy()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _discretize(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if np.ptp(values) == 0:
        return np.zeros(len(values), dtype=int)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def _mutual_information(left: np.ndarray, right: np.ndarray, bins: int) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    left_bins = _discretize(left, bins)
    right_bins = _discretize(right, bins)
    counts = np.zeros((left_bins.max() + 1, right_bins.max() + 1), dtype=float)
    np.add.at(counts, (left_bins, right_bins), 1)
    probabilities = counts / counts.sum()
    marginal_left = probabilities.sum(axis=1, keepdims=True)
    marginal_right = probabilities.sum(axis=0, keepdims=True)
    expected = marginal_left * marginal_right
    nonzero = (probabilities > 0) & (expected > 0)
    return float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero] / expected[nonzero])))


def recurrence_profile(
    signal: pd.Series,
    max_lag: int,
    *,
    bins: int = 16,
) -> pd.DataFrame:
    """Measure linear and nonlinear self-dependence for each lag."""
    values = _as_numeric_series(signal, "signal")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    rows = []
    for lag in range(max_lag + 1):
        current, previous = _paired_values(values, values, lag)
        rows.append({
            "lag": lag,
            "n_pairs": len(current),
            "autocorrelation": _correlation(current, previous),
            "mutual_information": _mutual_information(current, previous, bins),
        })
    return pd.DataFrame(rows)


def lagged_information_profile(
    predictors: pd.DataFrame,
    target: pd.Series,
    max_lag: int,
    *,
    bins: int = 16,
    n_shuffles: int = 0,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Measure information in each predictor lag about the current target.

    ``n_shuffles`` optionally adds a 95th-percentile mutual-information null
    threshold.  Shuffling is a statistical check, not model training.
    """
    if not isinstance(predictors, pd.DataFrame):
        raise TypeError("predictors must be a pandas DataFrame")
    if predictors.empty:
        raise ValueError("predictors must contain at least one column")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if n_shuffles < 0:
        raise ValueError("n_shuffles must be non-negative")
    target_values = _as_numeric_series(target, "target")
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(random_state)

    for column in predictors.columns:
        predictor = _as_numeric_series(predictors[column], f"predictors[{column!r}]")
        for lag in range(max_lag + 1):
            lagged, current_target = _paired_values(predictor, target_values, lag)
            mutual_information = _mutual_information(lagged, current_target, bins)
            row: dict[str, float | int | str] = {
                "predictor": str(column),
                "lag": lag,
                "n_pairs": len(lagged),
                "correlation": _correlation(lagged, current_target),
                "mutual_information": mutual_information,
            }
            if n_shuffles:
                shuffled = [
                    _mutual_information(lagged, rng.permutation(current_target), bins)
                    for _ in range(n_shuffles)
                ]
                row["shuffle_mi_95"] = float(np.quantile(shuffled, 0.95))
                row["mi_above_shuffle_95"] = int(mutual_information > row["shuffle_mi_95"])
            rows.append(row)
    return pd.DataFrame(rows)


def effective_memory(
    profile: pd.DataFrame,
    *,
    value_column: str = "autocorrelation",
    threshold: float = 0.1,
    consecutive_lags: int = 3,
) -> int | None:
    """Return the last meaningful lag, or ``None`` if the profile does not decay.

    A lag is considered no longer meaningful when the absolute value of the
    selected metric stays below ``threshold`` for ``consecutive_lags`` lags.
    """
    if value_column not in profile or "lag" not in profile:
        raise ValueError(f"profile must contain 'lag' and {value_column!r}")
    if threshold < 0 or consecutive_lags < 1:
        raise ValueError("threshold must be non-negative and consecutive_lags positive")
    values = profile.sort_values("lag")[value_column].abs().fillna(0).to_numpy()
    lags = profile.sort_values("lag")["lag"].to_numpy()
    for index in range(len(values) - consecutive_lags + 1):
        if np.all(values[index:index + consecutive_lags] < threshold):
            return max(0, int(lags[index] - 1))
    return None


def plot_profile(profile: pd.DataFrame, *, value_column: str = "autocorrelation") -> None:
    """Plot a profile returned by :func:`recurrence_profile`."""
    if value_column not in profile:
        raise ValueError(f"profile does not contain {value_column!r}")
    ax = profile.plot(x="lag", y=value_column, marker="o", grid=True, legend=False)
    ax.set(xlabel="Lag (samples)", ylabel=value_column.replace("_", " ").title())
    plt.tight_layout()
    plt.show()


def _windowed_lag_profile(
    data: RNNData,
    predictor_names: list[str],
    reference_names: list[str],
    max_lag: int,
    *,
    bins: int,
    n_shuffles: int,
    random_state: int | None,
) -> pd.DataFrame:
    """Measure lagged dependence without flattening overlapping RNN windows."""
    splits = [data.train, data.validation, data.test]
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(random_state)

    for predictor_index, predictor_name in enumerate(predictor_names):
        feature_index = predictor_index * 3  # value, mask, age
        for reference_index, reference_name in enumerate(reference_names):
            for lag in range(max_lag + 1):
                predictor_values = torch.cat([
                    split.inputs[:, -1 - lag, feature_index] for split in splits
                ]).numpy()
                target_values = torch.cat([
                    split.targets[:, reference_index] for split in splits
                ]).numpy()
                mutual_information = _mutual_information(
                    predictor_values, target_values, bins
                )
                row: dict[str, float | int | str] = {
                    "predictor": predictor_name,
                    "reference": reference_name,
                    "lag": lag,
                    "n_pairs": len(predictor_values),
                    "correlation": _correlation(predictor_values, target_values),
                    "mutual_information": mutual_information,
                }
                if n_shuffles:
                    shuffled = [
                        _mutual_information(
                            predictor_values,
                            rng.permutation(target_values),
                            bins,
                        )
                        for _ in range(n_shuffles)
                    ]
                    row["shuffle_mi_95"] = float(np.quantile(shuffled, 0.95))
                    row["mi_above_shuffle_95"] = int(
                        mutual_information > row["shuffle_mi_95"]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def analyze_rnn_history(
    data_folder: str | Path,
    *,
    predictor_contains: str | Iterable[str] | None = None,
    predictor_exclude: str | Iterable[str] | None = None,
    reference_contains: str | Iterable[str] | None = None,
    reference_exclude: str | Iterable[str] | None = None,
    max_lag: int = 60,
    bins: int = 16,
    n_shuffles: int = 0,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Check the same predictor/reference selection used by the RNN or LSTM.

    ``max_lag=60`` means the loader creates 61-step windows and the report
    checks the current input plus 60 previous samples.  No model is trained.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    data = load_rnn_data(
        data_folder,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        sequence_length=max_lag + 1,
    )
    predictor_names = [path.stem for path in data.predictor_files]
    reference_names = [path.stem for path in data.reference_files]
    profile = _windowed_lag_profile(
        data,
        predictor_names,
        reference_names,
        max_lag,
        bins=bins,
        n_shuffles=n_shuffles,
        random_state=random_state,
    )
    print(f"Predictor files: {', '.join(predictor_names)}")
    print(f"Reference files: {', '.join(reference_names)}")
    print(f"Window length checked: {max_lag + 1} samples")
    print(profile.to_string(index=False))
    return profile


if __name__ == "__main__":
    # Use the same folder and file filters as src/rnn/main.py or src/lstm/model.py.
    DATA_FOLDER = Path(
        "data/processed/2024-06-02_2024-10-31_stepinterp_10s"
    )
    PREDICTOR_CONTAINS = ["VANNSTR", "NALER"]
    PREDICTOR_EXCLUDE = ["VIBR", "TMP"]
    REFERENCE_CONTAINS = ["VIBR"]
    MAX_LAG = 60

    analyze_rnn_history(
        DATA_FOLDER,
        predictor_contains=PREDICTOR_CONTAINS,
        predictor_exclude=PREDICTOR_EXCLUDE,
        reference_contains=REFERENCE_CONTAINS,
        max_lag=MAX_LAG,
        n_shuffles=100,
    )