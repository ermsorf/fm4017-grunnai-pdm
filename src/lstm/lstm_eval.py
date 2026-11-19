"""Evaluate a saved LSTM checkpoint and save predictions."""

from collections.abc import Mapping
from pathlib import Path

import src.config

from src.lstm.lstm_model import LSTMModel
from src.recurrent_evaluation import (
    RecurrentEvaluation,
    evaluate_recurrent,
    plot_predictions,
    save_predictions,
)

LSTMEvaluation = RecurrentEvaluation


def plot_predictions_vs_inputs(
    result: LSTMEvaluation,
    output_path: str | Path,
    *,
    input_contains: tuple[str, ...] = ("VANNSTR", "NALER", "MW"),
    max_samples: int | None = None,
    show: bool = False,
) -> None:
    """Save a vertically stacked plot of selected inputs and LSTM predictions."""
    import matplotlib.pyplot as plt

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or None")
    selected_indices = tuple(
        index
        for index, name in enumerate(result.predictor_names)
        if any(keyword.casefold() in name.casefold() for keyword in input_contains)
    )
    if not selected_indices:
        raise ValueError(
            "No predictor filenames matched the requested input filters: "
            f"{input_contains}"
        )

    sample_count = min(result.sample_count, max_samples or result.sample_count)
    figure, axes = plt.subplots(
        len(selected_indices) + len(result.target_names),
        1,
        figsize=(14, 3 * (len(selected_indices) + len(result.target_names))),
        squeeze=False,
        sharex=True,
    )
    timestamps = result.timestamps[:sample_count]
    for axis_index, predictor_index in enumerate(selected_indices):
        axis = axes[axis_index, 0]
        axis.plot(
            timestamps,
            result.predictor_values[:sample_count, predictor_index].numpy(),
            label="Input",
            color="tab:blue",
        )
        axis.set_ylabel("Input")
        axis.set_title(result.predictor_names[predictor_index])
        axis.grid(alpha=0.25)
        axis.legend()

    target_axis_start = len(selected_indices)
    for target_index, name in enumerate(result.target_names):
        axis = axes[target_axis_start + target_index, 0]
        axis.plot(
            timestamps,
            result.targets[:sample_count, target_index].numpy(),
            label="Actual",
            color="tab:green",
        )
        axis.plot(
            timestamps,
            result.predictions[:sample_count, target_index].numpy(),
            label="Predicted",
            color="tab:orange",
        )
        axis.set_ylabel("Target")
        axis.set_title(f"LSTM prediction: {name}")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel("Timestamp")
    figure.autofmt_xdate()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)


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
) -> LSTMEvaluation:
    return evaluate_recurrent(
        LSTMModel,
        model_path,
        data_folder,
        config=config,
        checkpoint_key="lstm.weight_ih_l0",
        model_name="LSTM",
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
        Path(src.config.MODEL_SAVE_PATH) / "lstm" /"lstm_vibr1.pth",
        Path(src.config.SEQUENCE_MODEL_CONFIG.evaluation_data_folder),
    )
    plot_path = Path(src.config.MODEL_SAVE_PATH) / "lstm_predictions.png"
    input_plot_path = Path(src.config.MODEL_SAVE_PATH) / "lstm_predictions_vs_inputs.png"
    data_path = Path(src.config.MODEL_SAVE_PATH) / "lstm_predictions.csv"
    plot_predictions(result, plot_path, show=True)
    # plot_predictions_vs_inputs(result, input_plot_path, show=True)
    save_predictions(result, data_path)
    print(f"Samples: {result.sample_count}")
    print(f"MSE: {result.mse:.6f}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"MAE: {result.mae:.6f}")
    print(f"R²: {result.r2:.6f}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved input comparison plot to {input_plot_path}")
    print(f"Saved predictions to {data_path}")
    for metrics in result.per_target:
        print(f"Target {metrics['name']}: {metrics}")
