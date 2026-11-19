"""Evaluate a saved GRU checkpoint and save predictions."""

from collections.abc import Mapping
from pathlib import Path

import src.config

from src.gru.gru_model import GRUModel
from src.recurrent_evaluation import (
    RecurrentEvaluation,
    evaluate_recurrent,
    print_variable_influence,
    plot_predictions,
    save_predictions,
)

GRUEvaluation = RecurrentEvaluation


def evaluate(
    model_path: str | Path,
    data_folder: str | Path,
    *,
    config: src.config.RecurrentModelConfig = src.config.SEQUENCE_MODEL_CONFIG,
    sequence_length: int | None = None,
    batch_size: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    predictor_contains: str | list[str] | tuple[str, ...] | None = None,
    predictor_exclude: str | list[str] | tuple[str, ...] | None = None,
    predictor_features: tuple[str, ...] | list[str] | None = None,
    reference_contains: str | list[str] | tuple[str, ...] | None = None,
    reference_exclude: str | list[str] | tuple[str, ...] | None = None,
    training_normalization_parameters: str | Path | None = None,
    evaluation_normalization_parameters: str | Path | None = None,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int | None = None,
    exclude_after_samples: int | None = None,
    exclude_before: str | None = None,
    exclude_after: str | None = None,
) -> GRUEvaluation:
    return evaluate_recurrent(
        GRUModel,
        model_path,
        data_folder,
        config=config,
        checkpoint_key="gru.weight_ih_l0",
        model_name="GRU",
        sequence_length=sequence_length,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        predictor_features=predictor_features,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        training_normalization_parameters=training_normalization_parameters,
        evaluation_normalization_parameters=evaluation_normalization_parameters,
        exclude_below=exclude_below,
        exclude_before_samples=exclude_before_samples,
        exclude_after_samples=exclude_after_samples,
        exclude_before=exclude_before,
        exclude_after=exclude_after,
    )


if __name__ == "__main__":
    result = evaluate(
        Path(src.config.MODEL_SAVE_PATH) / "gru" / "vibr1" / "gru_vibr1.pth",
        Path(src.config.SEQUENCE_MODEL_CONFIG.evaluation_data_folder),
    )
    plot_path = Path(src.config.MODEL_SAVE_PATH) / "gru_predictions.png"
    data_path = Path(src.config.MODEL_SAVE_PATH) / "gru_predictions.csv"
    plot_predictions(result, plot_path, show=True)
    save_predictions(result, data_path)
    print(f"Samples: {result.sample_count}")
    print(f"MSE: {result.mse:.6f}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"MAE: {result.mae:.6f}")
    print(f"R²: {result.r2:.6f}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved predictions to {data_path}")
    for metrics in result.per_target:
        print(f"Target {metrics['name']}: {metrics}")
    print_variable_influence(result.variable_influence)
