"""Correlation and exploratory soft-sensor analysis for Generator 2.

The command line entry point compares VIBR2 with operating variables and
evaluates small, time-ordered regression models.  It accepts both raw exports
and the regular step-interpolated exports.

Examples
--------
python -m src.data_analysis.correlation --source processed
python -m src.data_analysis.correlation --source raw --resample 10s
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW_FOLDER = ROOT / "data" / "raw" / "2025-06-01_2025-10-31"
PROCESSED_FOLDER = ROOT / "data" / "processed" / "2025-06-02_2025-10-31_stepinterp_10s"
DEFAULT_OUTPUT = ROOT / "plots" / "correlation_vibr2"

SIGNALS = {
    "vibr2": "Scada.GRUN.AGG2.G2.LAGER.M_VIBR2.csv",
    "mw_power": "Scada.GRUN.AGG2.G2.MV.M_P_MW.csv",
    "flow": "Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR.csv",
    "wicket": "Scada.GRUN.AGG2.TURB2.PADRAG.M_WCKT_POS.csv",
    "speed": "Scada.GRUN.AGG2.TURB2.REG.M_TURT.csv",
    "water_pressure": "Scada.GRUN.AGG2.VANN.SJAKT.M_TRYKK.csv",
    "magnetization_current": "Scada.GRUN.AGG2.G2.MAGN.M_I.csv",
    "bearing_temp_1": "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP1.csv",
    "bearing_temp_4": "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP4.csv",
    "bearing_temp_5": "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP5.csv",
    "bearing_temp_6": "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP6.csv",
    "oil_temp_1": "Scada.GRUN.AGG2.G2.LAGER.M_OLTMP1.csv",
    "oil_temp_2": "Scada.GRUN.AGG2.G2.LAGER.M_OLTMP2.csv",
}

NOZZLES = [
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL1.csv",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL2.csv",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL3.csv",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL4.csv",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL5.csv",
]


def nozzle_force_amplitude(
    nozzle_data: pd.DataFrame,
    nozzle_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Estimate resultant force without assuming nozzle identities.

    The five values are treated as force magnitudes at five occupied vertices
    of a regular hexagon. Every assignment of the signals to those vertices is
    evaluated, so the output expresses geometric uncertainty rather than
    silently claiming that (for example) nozzle 1 opposes nozzle 4.

    The force directions are a geometric proxy. They should be replaced by
    measured jet directions if those are available.
    """
    columns = list(nozzle_columns or [f"nozzle_{index}" for index in range(1, 6)])
    if len(columns) != 5 or any(column not in nozzle_data for column in columns):
        raise ValueError("Exactly five nozzle columns are required")

    values = nozzle_data[columns].to_numpy(dtype=float)
    angles = np.deg2rad(np.arange(5) * 60.0)
    x_weights = np.cos(angles)
    y_weights = np.sin(angles)
    amplitudes = np.empty((len(values), 120))

    for assignment_index, assignment in enumerate(permutations(range(5))):
        assigned = values[:, assignment]
        resultant_x = assigned @ x_weights
        resultant_y = assigned @ y_weights
        amplitudes[:, assignment_index] = np.hypot(resultant_x, resultant_y)

    return pd.DataFrame(
        {
            "nozzle_force_min": amplitudes.min(axis=1),
            "nozzle_force_median": np.median(amplitudes, axis=1),
            "nozzle_force_mean": amplitudes.mean(axis=1),
            "nozzle_force_max": amplitudes.max(axis=1),
            "nozzle_force_uncertainty": amplitudes.std(axis=1),
        },
        index=nozzle_data.index,
    )


def read_signal(path: str | Path, name: str) -> pd.Series:
    """Read either a raw export or a processed ``*_value`` export."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "timestamp" in header and any(column.endswith("_value") for column in header):
        frame = pd.read_csv(path, usecols=["timestamp", next(c for c in header if c.endswith("_value"))])
        frame.columns = ["timestamp", name]
    else:
        frame = pd.read_csv(path, header=None, skiprows=2, usecols=[0, 1])
        frame.columns = ["timestamp", name]

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return (
        frame.dropna()
        .drop_duplicates("timestamp")
        .set_index("timestamp")[name]
        .sort_index()
    )


def load_aligned_signals(
    folder: str | Path,
    signal_files: dict[str, str],
    *,
    interval: str = "10s",
    max_age_steps: int | None = None,
) -> pd.DataFrame:
    """Load signals onto one time grid.

    Raw data is aggregated to the grid without forward filling.  Processed
    step-held data is already on the grid; ``max_age_steps`` can mask values
    that have been held for too long using the corresponding ``*_age`` column.
    """
    folder = Path(folder)
    columns = []
    for name, filename in signal_files.items():
        path = folder / filename
        if not path.exists():
            continue
        series = read_signal(path, name)
        if "timestamp" in pd.read_csv(path, nrows=0).columns:
            if max_age_steps is not None:
                metadata = pd.read_csv(path, usecols=lambda c: c.endswith("_age"))
                if not metadata.empty:
                    age = pd.to_numeric(metadata.iloc[:, 0], errors="coerce")
                    age.index = pd.to_datetime(
                        pd.read_csv(path, usecols=["timestamp"])["timestamp"],
                        errors="coerce",
                    )
                    series = series.mask(age.reindex(series.index) > max_age_steps)
        else:
            series = series.resample(interval).mean()
        columns.append(series.rename(name))

    if not columns:
        raise FileNotFoundError(f"No requested signal files found in {folder}")
    return pd.concat(columns, axis=1, join="inner").dropna()


def add_nozzle_features(data: pd.DataFrame, folder: str | Path, *, interval: str) -> pd.DataFrame:
    """Add total and imbalance features for the five Generator 2 nozzles."""
    nozzle_files = {f"nozzle_{index + 1}": filename for index, filename in enumerate(NOZZLES)}
    nozzle_data = load_aligned_signals(folder, nozzle_files, interval=interval)
    result = data.join(nozzle_data, how="inner")
    nozzle_columns = list(nozzle_files)
    physical_nozzles = nozzle_data.copy()
    parameters_path = Path(folder) / "normalization_parameters.json"
    if parameters_path.exists():
        with parameters_path.open(encoding="utf-8") as parameters_file:
            parameters = json.load(parameters_file)
        for column, filename in nozzle_files.items():
            parameter = parameters.get(filename)
            if parameter is not None:
                physical_nozzles[column] = (
                    physical_nozzles[column] * parameter["std"] + parameter["mean"]
                )
    result["nozzle_total"] = physical_nozzles.sum(axis=1)
    result["nozzle_mean"] = physical_nozzles.mean(axis=1)
    result["nozzle_spread"] = physical_nozzles.max(axis=1) - physical_nozzles.min(axis=1)
    result = result.join(nozzle_force_amplitude(physical_nozzles, nozzle_columns))
    return result


def correlation_table(data: pd.DataFrame, target: str = "vibr2") -> pd.DataFrame:
    """Return Pearson and Spearman correlations against ``target``."""
    rows = []
    for column in data.columns:
        if column == target:
            continue
        pair = data[[target, column]].dropna()
        rows.append({
            "signal": column,
            "n": len(pair),
            "pearson": pair[target].corr(pair[column], method="pearson"),
            "spearman": pair[target].corr(pair[column], method="spearman"),
        })
    return pd.DataFrame(rows).sort_values("spearman", key=lambda values: values.abs(), ascending=False)


def fit_pressure_log_curve(
    data: pd.DataFrame,
    *,
    pressure_column: str = "water_pressure",
    target: str = "vibr2",
    max_rows: int = 100_000,
) -> tuple[dict[str, float], pd.Series]:
    """Fit ``VIBR2 = a * log(b - pressure) + c`` with ``a > 0``.

    ``b`` is constrained above the largest observed pressure so the logarithm
    remains defined. The returned series is the fitted VIBR2 value at each
    input row. This is an exploratory descriptive feature, not a causal model.
    """
    pair = data[[pressure_column, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 10:
        raise ValueError("At least 10 pressure/VIBR2 observations are required")
    if len(pair) > max_rows:
        pair = pair.iloc[np.linspace(0, len(pair) - 1, max_rows, dtype=int)]

    pressure = pair[pressure_column].to_numpy(dtype=float)
    vibration = pair[target].to_numpy(dtype=float)
    pressure_range = max(float(np.ptp(pressure)), 1.0)
    pressure_max = float(data[pressure_column].dropna().max())
    pressure_min = float(np.min(pressure))
    initial_b = pressure_max + pressure_range
    initial_a = max(float(np.std(vibration)), 1e-3) / max(
        abs(np.log(initial_b - pressure_min) - np.log(initial_b - pressure_max)),
        1e-3,
    )
    initial_c = float(np.median(vibration - initial_a * np.log(initial_b - pressure)))

    # Optimize log(a) and log(b - max(x)); this enforces both constraints.
    def unpack(parameters: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.exp(parameters[0])),
            pressure_max + float(np.exp(parameters[1])),
            float(parameters[2]),
        )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        a, b, c = unpack(parameters)
        return a * np.log(b - pressure) + c - vibration

    result = least_squares(
        residuals,
        x0=np.array([np.log(initial_a), np.log(initial_b - pressure_max), initial_c]),
        loss="soft_l1",
        f_scale=max(float(np.std(vibration)), 1e-3),
        max_nfev=2_000,
    )
    if not result.success:
        raise ValueError(f"Pressure log-curve fitting failed: {result.message}")

    a, b, c = unpack(result.x)
    fitted = pd.Series(
        a * np.log(b - data[pressure_column]),
        index=data.index,
        name="water_pressure_log_fit",
    ) + c
    return {"a": a, "b": b, "c": c}, fitted


def evaluate_soft_sensor(
    data: pd.DataFrame,
    *,
    target: str = "vibr2",
    features: Iterable[str] | None = None,
    polynomial_degree: int = 2,
    splits: int = 5,
    max_rows: int = 100_000,
) -> tuple[pd.DataFrame, Pipeline]:
    """Evaluate an ElasticNet soft sensor with chronological cross-validation."""
    selected = list(features or [column for column in data.columns if column != target])
    selected = [column for column in selected if column in data.columns]
    if not selected:
        raise ValueError("No soft-sensor features are present in the aligned data")
    clean = data[[target, *selected]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) > max_rows:
        clean = clean.iloc[np.linspace(0, len(clean) - 1, max_rows, dtype=int)]
    if len(clean) < max(20, splits + 1):
        raise ValueError("Not enough aligned observations for time-ordered validation")

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=polynomial_degree, include_bias=False)),
        ("regression", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000)),
    ])
    n_splits = min(splits, len(clean) - 1)
    predictions = np.full(len(clean), np.nan)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_indices, test_indices in splitter.split(clean[selected]):
        fold_model = Pipeline(model.steps)
        fold_model.fit(clean[selected].iloc[train_indices], clean[target].iloc[train_indices])
        predictions[test_indices] = fold_model.predict(clean[selected].iloc[test_indices])

    valid = np.isfinite(predictions)
    metrics = pd.DataFrame([{
        "features": ",".join(selected),
        "n": int(valid.sum()),
        "mae": mean_absolute_error(clean[target].iloc[valid], predictions[valid]),
        "rmse": np.sqrt(mean_squared_error(clean[target].iloc[valid], predictions[valid])),
        "r2": r2_score(clean[target].iloc[valid], predictions[valid]),
    }])
    model.fit(clean[selected], clean[target])
    return metrics, model


def plot_results(
    data: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
    *,
    max_points: int = 20_000,
) -> None:
    """Save correlations and scatter plots with VIBR2 on every x-axis."""
    output.parent.mkdir(parents=True, exist_ok=True)
    top = correlations.head(8).sort_values("spearman")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top["signal"], top["spearman"])
    axis.set_title("VIBR2 rank correlation")
    axis.set_xlabel("Spearman correlation")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)

    signals = list(correlations["signal"])
    sampled = data[["vibr2", *signals]].dropna()
    if len(sampled) > max_points:
        sampled = sampled.iloc[np.linspace(0, len(sampled) - 1, max_points, dtype=int)]
    columns = 2
    rows = int(np.ceil(len(signals) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), squeeze=False)
    for axis, signal in zip(axes.flat, signals):
        axis.scatter(sampled["vibr2"], sampled[signal], s=5, alpha=0.2)
        axis.set_title(f"VIBR2 vs {signal}")
        axis.set_xlabel("VIBR2")
        axis.set_ylabel(signal)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(signals):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output.with_name(f"{output.stem}_scatter.png"), dpi=160)
    plt.close(figure)


def plot_soft_sensor_results(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, Pipeline],
    output: Path,
    *,
    max_points: int = 20_000,
) -> None:
    """Plot actual VIBR2 on x against each fitted soft-sensor estimate on y."""
    plot_data = data.copy()
    for name, features in feature_sets.items():
        if name not in models:
            continue
        available = [feature for feature in features if feature in plot_data]
        valid = plot_data[["vibr2", *available]].dropna()
        if valid.empty:
            continue
        if len(valid) > max_points:
            valid = valid.iloc[np.linspace(0, len(valid) - 1, max_points, dtype=int)]
        plot_data.loc[valid.index, f"{name}_prediction"] = models[name].predict(valid[available])

    prediction_columns = [column for column in plot_data if column.endswith("_prediction")]
    if not prediction_columns:
        return
    figure, axes = plt.subplots(
        int(np.ceil(len(prediction_columns) / 2)),
        2,
        figsize=(14, max(4, 4 * int(np.ceil(len(prediction_columns) / 2)))),
        squeeze=False,
    )
    for axis, prediction_column in zip(axes.flat, prediction_columns):
        name = prediction_column.removesuffix("_prediction")
        valid = plot_data[["vibr2", prediction_column]].dropna()
        axis.scatter(valid["vibr2"], valid[prediction_column], s=5, alpha=0.2)
        limits = np.array([valid.min().min(), valid.max().max()])
        axis.plot(limits, limits, "r--", linewidth=1, label="ideal")
        axis.set_title(f"VIBR2 vs soft sensor: {name}")
        axis.set_xlabel("Actual VIBR2")
        axis.set_ylabel("Predicted VIBR2")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    for axis in axes.flat[len(prediction_columns):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output.with_name(f"{output.stem}_soft_sensors.png"), dpi=160)
    plt.close(figure)


def plot_pressure_log_curve(
    data: pd.DataFrame,
    parameters: dict[str, float],
    output: Path,
    *,
    max_points: int = 20_000,
) -> None:
    """Plot VIBR2 against pressure with the fitted log curve overlaid."""
    valid = data[["water_pressure", "vibr2"]].dropna()
    if len(valid) > max_points:
        valid = valid.iloc[np.linspace(0, len(valid) - 1, max_points, dtype=int)]
    x = np.linspace(valid["water_pressure"].min(), valid["water_pressure"].max(), 400)
    y = parameters["a"] * np.log(parameters["b"] - x) + parameters["c"]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(valid["water_pressure"], valid["vibr2"], s=5, alpha=0.2, label="observations")
    axis.plot(x, y, color="red", linewidth=2, label="a log(b - x) + c")
    axis.set_title("VIBR2 vs water pressure: constrained log fit")
    axis.set_xlabel("Water pressure")
    axis.set_ylabel("VIBR2")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output.with_name(f"{output.stem}_pressure_log_fit.png"), dpi=160)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("processed", "raw"), default="processed")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--resample", default="10s")
    parser.add_argument("--max-age-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.folder or (PROCESSED_FOLDER if args.source == "processed" else RAW_FOLDER)
    data = load_aligned_signals(
        folder,
        SIGNALS,
        interval=args.resample,
        max_age_steps=args.max_age_steps,
    )
    try:
        data = add_nozzle_features(data, folder, interval=args.resample)
    except FileNotFoundError:
        pass

    pressure_parameters = None
    try:
        pressure_parameters, pressure_fit = fit_pressure_log_curve(data)
        data["water_pressure_log_fit"] = pressure_fit
        data["water_pressure_log_residual"] = data["vibr2"] - pressure_fit
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name("vibr2_pressure_log_parameters.json").write_text(
            json.dumps(pressure_parameters, indent=2),
            encoding="utf-8",
        )
    except ValueError as error:
        print(f"water_pressure_log_fit: {error}")

    correlations = correlation_table(data)
    correlations.to_csv(args.output.with_name("vibr2_correlations.csv"), index=False)
    feature_sets = {
        "mw_only": ["mw_power"],
        "mw_and_hydraulic": ["mw_power", "flow", "wicket", "water_pressure"],
        "mw_and_pressure_log_feature": [
            "mw_power", "water_pressure", "water_pressure_log_fit",
        ],
        "mw_hydraulic_and_machine": [
            "mw_power", "flow", "wicket", "speed", "water_pressure",
            "bearing_temp_1", "bearing_temp_4", "bearing_temp_5", "bearing_temp_6",
            "oil_temp_1", "oil_temp_2", "magnetization_current",
        ],
        "mw_plus_mapping_agnostic_nozzle_force": [
            "mw_power", "nozzle_total", "nozzle_spread",
            "nozzle_force_min", "nozzle_force_median", "nozzle_force_max",
            "nozzle_force_uncertainty",
        ],
    }
    results = []
    models = {}
    for name, features in feature_sets.items():
        try:
            metrics, model = evaluate_soft_sensor(data, features=features)
            metrics.insert(0, "model", name)
            results.append(metrics)
            models[name] = model
        except ValueError as error:
            print(f"{name}: {error}")
    if results:
        pd.concat(results, ignore_index=True).to_csv(
            args.output.with_name("vibr2_soft_sensor_results.csv"), index=False
        )
    plot_results(data, correlations, args.output.with_suffix(".png"))
    if pressure_parameters is not None:
        plot_pressure_log_curve(data, pressure_parameters, args.output)
    plot_soft_sensor_results(data, feature_sets, models, args.output)
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
