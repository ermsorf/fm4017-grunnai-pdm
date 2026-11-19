"""Evaluate a saved LSTM checkpoint and plot predictions against actual values."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import torch

from src.data_handling import select_data_files
from src.lstm.model import LSTMModel
from src.rnn.data import load_rnn_evaluation_data


@dataclass(frozen=True)
class LSTMEvaluation:
    """Regression metrics and values from an LSTM evaluation run."""

    sample_count: int
    mse: float
    rmse: float
    mae: float
    r2: float
    target_names: tuple[str, ...]
    predictions: torch.Tensor
    targets: torch.Tensor
    per_target: tuple[dict[str, float | str], ...]


def _load_normalization_parameters(
    path: str | Path,
) -> dict[str, dict[str, float]]:
    parameters_path = Path(path)
    if not parameters_path.is_file():
        raise FileNotFoundError(f"Normalization parameters not found: {parameters_path}")
    with parameters_path.open(encoding="utf-8") as file:
        parameters = json.load(file)
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Normalization parameters must be an object: {parameters_path}")
    return parameters


def _denormalize(
    values: torch.Tensor,
    parameters: list[dict[str, float]],
) -> torch.Tensor:
    means = torch.tensor([item["mean"] for item in parameters], dtype=values.dtype)
    stds = torch.tensor([item["std"] for item in parameters], dtype=values.dtype)
    return values * stds + means


def _normalize(
    values: torch.Tensor,
    parameters: list[dict[str, float]],
) -> torch.Tensor:
    means = torch.tensor([item["mean"] for item in parameters], dtype=values.dtype)
    stds = torch.tensor([item["std"] for item in parameters], dtype=values.dtype)
    if torch.any(stds <= 0):
        raise ValueError("Normalization standard deviations must be positive")
    return (values - means) / stds


def evaluate(
    model_path: str | Path,
    data_folder: str | Path,
    *,
    sequence_length: int = 4,
    batch_size: int = 1024,
    hidden_size: int = 32,
    num_layers: int = 4,
    predictor_contains: str | list[str] | None = None,
    predictor_exclude: str | list[str] | None = None,
    reference_contains: str | list[str] | None = None,
    reference_exclude: str | list[str] | None = None,
    training_normalization_parameters: str | Path | None = None,
    evaluation_normalization_parameters: str | Path | None = None,
) -> LSTMEvaluation:
    """Evaluate an LSTM model and report metrics in the original signal units."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    evaluation_parameters_path = (
        evaluation_normalization_parameters
        or Path(data_folder) / "normalization_parameters.json"
    )
    training_parameters_path = (
        training_normalization_parameters or evaluation_parameters_path
    )
    evaluation_parameters = _load_normalization_parameters(evaluation_parameters_path)
    training_parameters = _load_normalization_parameters(training_parameters_path)

    predictor_files = tuple(select_data_files(
        data_folder,
        filename_contains=predictor_contains,
        filename_exclude=predictor_exclude,
    ))
    reference_files = tuple(select_data_files(
        data_folder,
        filename_contains=reference_contains,
        filename_exclude=reference_exclude,
    ))
    selected_files = (*predictor_files, *reference_files)
    missing = [
        path.name
        for path in selected_files
        if path.name not in evaluation_parameters
        or path.name not in training_parameters
    ]
    if missing:
        raise ValueError(f"Missing normalization parameters for: {sorted(set(missing))}")

    evaluation_data = load_rnn_evaluation_data(
        data_folder,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        sequence_length=sequence_length,
    )
    if len(evaluation_data.inputs) == 0:
        raise ValueError("The evaluation dataset contains no complete sequences")

    predictor_parameters = [evaluation_parameters[path.name] for path in predictor_files]
    training_predictor_parameters = [
        training_parameters[path.name] for path in predictor_files
    ]
    reference_parameters = [evaluation_parameters[path.name] for path in reference_files]
    training_reference_parameters = [
        training_parameters[path.name] for path in reference_files
    ]

    inputs = evaluation_data.inputs.clone()
    for index, (evaluation_parameter, training_parameter) in enumerate(
        zip(predictor_parameters, training_predictor_parameters)
    ):
        value_slice = inputs[:, :, index * 3:index * 3 + 1]
        inputs[:, :, index * 3:index * 3 + 1] = _normalize(
            _denormalize(value_slice, [evaluation_parameter]),
            [training_parameter],
        )

    model = LSTMModel(
        input_size=inputs.shape[2],
        hidden_size=hidden_size,
        output_size=len(reference_files),
        num_layers=num_layers,
    )
    model.load_model(model_path)
    prediction_batches = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        prediction_batches.append(model.predict(batch).cpu())
    predictions = _denormalize(
        torch.cat(prediction_batches, dim=0),
        training_reference_parameters,
    )
    targets = _denormalize(evaluation_data.targets, reference_parameters)
    errors = predictions - targets

    target_mse = torch.mean(errors.square(), dim=0)
    target_mae = torch.mean(errors.abs(), dim=0)
    target_variance = torch.sum(
        (targets - targets.mean(dim=0)).square(),
        dim=0,
    )
    target_r2 = 1 - torch.sum(errors.square(), dim=0) / target_variance.clamp_min(
        torch.finfo(targets.dtype).eps
    )
    per_target = tuple(
        {
            "name": reference_files[index].name,
            "mse": float(target_mse[index]),
            "rmse": float(torch.sqrt(target_mse[index])),
            "mae": float(target_mae[index]),
            "r2": float(target_r2[index]),
        }
        for index in range(targets.shape[1])
    )
    mse = float(torch.mean(errors.square()))
    return LSTMEvaluation(
        sample_count=len(targets),
        mse=mse,
        rmse=mse**0.5,
        mae=float(torch.mean(errors.abs())),
        r2=float(
            1
            - torch.sum(errors.square())
            / torch.sum((targets - targets.mean()).square()).clamp_min(
                torch.finfo(targets.dtype).eps
            )
        ),
        target_names=tuple(path.name for path in reference_files),
        predictions=predictions,
        targets=targets,
        per_target=per_target,
    )


def plot_predictions(
    result: LSTMEvaluation,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
) -> None:
    """Save one predicted-versus-actual plot for each target."""
    import matplotlib.pyplot as plt

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or None")
    sample_count = min(result.sample_count, max_samples or result.sample_count)
    figure, axes = plt.subplots(
        len(result.target_names),
        1,
        figsize=(12, 4 * len(result.target_names)),
        squeeze=False,
    )
    for index, name in enumerate(result.target_names):
        axis = axes[index, 0]
        axis.plot(
            result.targets[:sample_count, index].numpy(),
            label="Actual",
            linewidth=1,
        )
        axis.plot(
            result.predictions[:sample_count, index].numpy(),
            label="Predicted",
            linewidth=1,
        )
        axis.set_title(name)
        axis.set_xlabel("Evaluation sample")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    import src.config

    result = evaluate(
        Path(src.config.MODEL_SAVE_PATH) / "lstm_model.pth",
        Path(src.config.DATA_PROCESSED) / "2024-06-02_2024-10-31_stepinterp_2s",
        batch_size=1024,
        training_normalization_parameters=Path(src.config.DATA_PROCESSED)
        / "2024-06-02_2024-10-31_stepinterp_2s"
        / "normalization_parameters.json",
        predictor_contains=["VANNSTR", "NALER", "MW"],
        predictor_exclude=["VIBR", "TMP"],
        reference_contains=["VIBR"],
    )
    plot_path = Path(src.config.MODEL_SAVE_PATH) / "lstm_predictions.png"
    plot_predictions(result, plot_path)
    print(f"Samples: {result.sample_count}")
    print(f"MSE: {result.mse:.6f}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"MAE: {result.mae:.6f}")
    print(f"R²: {result.r2:.6f}")
    print(f"Saved plot to {plot_path}")
    for metrics in result.per_target:
        print(f"Target {metrics['name']}: {metrics}")
