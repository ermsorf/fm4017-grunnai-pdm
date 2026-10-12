"""Train and evaluate XGBoost regressors on step-interpolated signal files."""


from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score, r2_score
from xgboost import XGBRegressor

import src.config


def load_data(data_folder: Path) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(data_folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    frames: list[pd.DataFrame] = []
    target_columns: list[str] = []
    for file in files:
        frame = pd.read_csv(file)
        if "timestamp" not in frame:
            raise ValueError(f"{file} has no 'timestamp' column")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
        value_columns = [column for column in frame if column.endswith("_value")]
        if not value_columns:
            raise ValueError(f"{file} has no '*_value' column")
        if "VIBR" in file.name:
            target_columns.extend(value_columns)
        else:
            frames.append(frame)

    if not frames or not target_columns:
        raise ValueError("Expected both non-VIBR predictor files and VIBR target files")

    data = frames[0]
    for frame in frames[1:]:
        data = data.merge(frame, on="timestamp", how="inner")
    for file in sorted(data_folder.glob("*VIBR*.csv")):
        target = pd.read_csv(file)
        target["timestamp"] = pd.to_datetime(target["timestamp"], errors="coerce")
        target = target.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
        data = data.merge(target, on="timestamp", how="inner")

    data = data.sort_values("timestamp").reset_index(drop=True)
    target_columns = sorted(set(target_columns).intersection(data.columns))
    return data, target_columns


def score(y_true: pd.Series, prediction: np.ndarray) -> tuple[float, float, float]:
    valid = y_true.notna().to_numpy() & np.isfinite(prediction)
    actual = y_true.to_numpy()[valid]
    predicted = prediction[valid]
    if len(actual) < 2:
        return float("nan"), float("nan"), float("nan")
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return (
        float(r2_score(actual, predicted)),
        float(explained_variance_score(actual, predicted)),
        correlation,
    )


def train_target(
    data: pd.DataFrame,
    target: str,
    train_end: int,
    validation_end: int,
) -> None:
    feature_columns = [
        column
        for column in data.columns
        if column != "timestamp" and "VIBR" not in column
    ]
    feature_columns = [column for column in feature_columns if column in data]
    if not feature_columns:
        raise ValueError("No predictor columns remain after excluding vibration values")

    x = pd.DataFrame(
        {
            column: pd.to_numeric(data[column], errors="coerce").astype("float32")
            for column in feature_columns
        },
        index=data.index,
    )
    y = pd.to_numeric(data[target], errors="coerce")
    usable = y.notna()
    x, y = x.loc[usable], y.loc[usable]
    train_end = min(train_end, len(x))
    validation_end = min(validation_end, len(x))
    if train_end < 2 or validation_end <= train_end or validation_end >= len(x):
        raise ValueError(f"Not enough usable rows to split target {target}")

    x_train, x_validation, x_test = (
        x.iloc[:train_end],
        x.iloc[train_end:validation_end],
        x.iloc[validation_end:],
    )
    y_train, y_validation, y_test = (
        y.iloc[:train_end],
        y.iloc[train_end:validation_end],
        y.iloc[validation_end:],
    )
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_validation = imputer.transform(x_validation)
    x_test = imputer.transform(x_test)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=32,
        learning_rate=0.0001,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)

    print(f"\n{target} ({len(feature_columns)} features)")
    print("split        R2       explained_variance  correlation")
    for name, actual, features in (
        ("validation", y_validation, x_validation),
        ("test", y_test, x_test),
    ):
        r2, variance, correlation = score(actual, model.predict(features))
        print(f"{name:<12} {r2:8.4f} {variance:19.4f} {correlation:12.4f}")


def main() -> None:
    data_folder = (
        Path(src.config.DATA_PROCESSED)
        / "2024-06-02_2024-10-31_stepinterp_10s"
    )
    data, targets = load_data(data_folder)
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    print(f"rows={len(data)}, targets={len(targets)}")
    for target in targets:
        train_target(data, target, train_end, validation_end)


if __name__ == "__main__":
    main()
