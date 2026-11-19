"""Train XGBoost models and measure sensor influence with permutation importance.

Run from the repository root:

    python -m src.xgboost.variable_influence

The script uses the XGBoost settings in ``src.config`` and writes one ranked
CSV file per target plus a combined CSV to ``plots/xgboost_variable_influence``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

import src.config
from src.data_loader import TabularData, load_tabular_data
from src.xgboost.main import _features, score


def build_model(config: src.config.XGBoostModelConfig) -> XGBRegressor:
    """Create an XGBoost regressor from the project configuration."""
    return XGBRegressor(
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


def analyze_target(
    data: TabularData,
    target: str,
    config: src.config.XGBoostModelConfig,
    *,
    repeats: int,
) -> pd.DataFrame:
    """Train one target model and return ranked test-set variable influence."""
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
        raise ValueError(f"Not enough usable rows to analyze target {target!r}")

    imputer = SimpleImputer(strategy="median")
    x_train_array = imputer.fit_transform(x_train)
    x_validation_array = imputer.transform(x_validation)
    x_test_array = imputer.transform(x_test)

    model = build_model(config)
    model.fit(
        x_train_array,
        y_train,
        eval_set=[(x_validation_array, y_validation)],
        verbose=False,
    )
    prediction = model.predict(x_test_array)
    r2, variance, correlation = score(y_test, prediction)
    print(
        f"{target}: test R2={r2:.4f}, explained variance={variance:.4f}, "
        f"correlation={correlation:.4f}"
    )

    importance = permutation_importance(
        model,
        x_test_array,
        y_test,
        scoring="r2",
        n_repeats=repeats,
        random_state=config.random_state,
    )
    return (
        pd.DataFrame(
            {
                "target": target,
                "variable": x_test.columns,
                "mean_importance": importance.importances_mean,
                "std_importance": importance.importances_std,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=None)
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("plots") / "xgboost_variable_influence",
    )
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    config = src.config.XGBOOST_MODEL_CONFIG
    data_folder = config.training_data_folder if args.data_folder is None else args.data_folder
    data = load_tabular_data(
        data_folder,
        predictor_contains=config.predictor_contains,
        predictor_exclude=config.predictor_exclude,
        predictor_features=config.predictor_features,
        reference_contains=config.reference_contains,
        reference_exclude=config.reference_exclude,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        exclude_below=config.exclude_below,
    )

    args.output_folder.mkdir(parents=True, exist_ok=True)
    reports = [
        analyze_target(data, target, config, repeats=args.repeats)
        for target in data.train.targets.columns
    ]
    combined = pd.concat(reports, ignore_index=True)
    for target, report in combined.groupby("target", sort=False):
        report.to_csv(args.output_folder / f"{target}_variable_influence.csv", index=False)
    combined.to_csv(args.output_folder / "all_targets_variable_influence.csv", index=False)
    print(f"\nWrote variable influence reports to {args.output_folder}")


if __name__ == "__main__":
    main()
