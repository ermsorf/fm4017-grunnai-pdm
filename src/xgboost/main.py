"""Train and evaluate XGBoost on per-timestep sensor-to-vibration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

import src.config
from src.data_loader import TabularData, load_tabular_data


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


def _features(split: pd.DataFrame) -> pd.DataFrame:
    return split.apply(pd.to_numeric, errors="coerce").astype("float32")


def variable_influence(
    model: XGBRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate each variable's influence using permutation importance.

    The returned ``mean_importance`` is the mean decrease in R2 after
    shuffling one variable. Negative values indicate that shuffling the
    variable improved the score for that repeat.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    numeric_features = _features(features)
    numeric_target = pd.to_numeric(target, errors="coerce")
    valid_rows = numeric_target.notna()
    numeric_features = numeric_features.loc[valid_rows]
    numeric_target = numeric_target.loc[valid_rows]
    if len(numeric_target) < 2:
        raise ValueError("At least two usable rows are required")

    result = permutation_importance(
        model,
        numeric_features,
        numeric_target,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return (
        pd.DataFrame(
            {
                "variable": numeric_features.columns,
                "mean_importance": result.importances_mean,
                "std_importance": result.importances_std,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def train_target(
    data: TabularData,
    target: str,
    *,
    config: src.config.XGBoostModelConfig = src.config.XGBOOST_MODEL_CONFIG,
) -> XGBRegressor:
    """Train one regressor for one vibration reference signal."""
    if target not in data.train.targets.columns:
        raise ValueError(f"Unknown target {target!r}")

    x_train = _features(data.train.inputs)
    x_validation = _features(data.validation.inputs)
    x_test = _features(data.test.inputs)
    y_train = pd.to_numeric(data.train.targets[target], errors="coerce")
    y_validation = pd.to_numeric(data.validation.targets[target], errors="coerce")
    y_test = pd.to_numeric(data.test.targets[target], errors="coerce")

    train_rows = y_train.notna()
    validation_rows = y_validation.notna()
    test_rows = y_test.notna()
    x_train, y_train = x_train.loc[train_rows], y_train.loc[train_rows]
    x_validation, y_validation = x_validation.loc[validation_rows], y_validation.loc[validation_rows]
    x_test, y_test = x_test.loc[test_rows], y_test.loc[test_rows]
    if len(x_train) < 2 or x_validation.empty or x_test.empty:
        raise ValueError(f"Not enough usable rows to split target {target}")

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_validation = imputer.transform(x_validation)
    x_test = imputer.transform(x_test)

    model = XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=config.objective,
        tree_method=config.tree_method,
        random_state=config.random_state,
        device=config.device,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    print(f"\n{target} ({x_train.shape[1]} features)")
    print("split        R2       explained_variance  correlation")
    for name, actual, features in (
        ("validation", y_validation, x_validation),
        ("test", y_test, x_test),
    ):
        r2, variance, correlation = score(actual, model.predict(features))
        print(f"{name:<12} {r2:8.4f} {variance:19.4f} {correlation:12.4f}")
    return model


def run(
    data_folder: str | Path | None = None,
    *,
    config: src.config.XGBoostModelConfig = src.config.XGBOOST_MODEL_CONFIG,
) -> dict[str, XGBRegressor]:
    """Load selected sensor files and train one model per vibration target."""
    folder = config.training_data_folder if data_folder is None else data_folder
    data = load_tabular_data(
        folder,
        predictor_contains=config.predictor_contains,
        predictor_exclude=config.predictor_exclude,
        predictor_features=config.predictor_features,
        reference_contains=config.reference_contains,
        reference_exclude=config.reference_exclude,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        exclude_below=config.exclude_below,
    )
    targets = list(data.train.targets.columns)
    print(
        f"rows={len(data.train.inputs) + len(data.validation.inputs) + len(data.test.inputs)}, "
        f"targets={len(targets)}"
    )
    return {
        target: train_target(data, target, config=config)
        for target in targets
    }


if __name__ == "__main__":
    run()
