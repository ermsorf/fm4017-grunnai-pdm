"""Shared mechanics for model-specific recurrent evaluators."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import pandas as pd
import torch

from src.data_handling import select_data_files
from src.data_loader import load_rnn_evaluation_data


class RecurrentModel(Protocol):
    def load_model(self, path: str | Path) -> None: ...
    def predict(self, inputs: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class RecurrentEvaluation:
    sample_count: int
    mse: float
    rmse: float
    mae: float
    r2: float
    target_names: tuple[str, ...]
    predictor_names: tuple[str, ...]
    predictor_values: torch.Tensor
    timestamps: pd.DatetimeIndex
    predictions: torch.Tensor
    targets: torch.Tensor
    per_target: tuple[dict[str, float | str], ...]
    variable_influence: tuple[dict[str, float | str], ...]


def load_normalization_parameters(path: str | Path) -> dict[str, dict[str, float]]:
    parameters_path = Path(path)
    if not parameters_path.is_file():
        raise FileNotFoundError(f"Normalization parameters not found: {parameters_path}")
    with parameters_path.open(encoding="utf-8") as file:
        parameters = json.load(file)
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Normalization parameters must be an object: {parameters_path}")
    return parameters


def denormalize(
    values: torch.Tensor,
    parameters: list[dict[str, float]],
) -> torch.Tensor:
    means = torch.tensor([item["mean"] for item in parameters], dtype=values.dtype)
    stds = torch.tensor([item["std"] for item in parameters], dtype=values.dtype)
    return values * stds + means


def normalize(
    values: torch.Tensor,
    parameters: list[dict[str, float]],
) -> torch.Tensor:
    means = torch.tensor([item["mean"] for item in parameters], dtype=values.dtype)
    stds = torch.tensor([item["std"] for item in parameters], dtype=values.dtype)
    if torch.any(stds <= 0):
        raise ValueError("Normalization standard deviations must be positive")
    return (values - means) / stds


def checkpoint_input_size(
    model_path: str | Path,
    *,
    checkpoint_key: str,
    model_name: str,
) -> int:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    input_weights = state_dict.get(checkpoint_key)
    if not isinstance(input_weights, torch.Tensor) or input_weights.ndim != 2:
        raise ValueError(
            f"{model_name} checkpoint does not contain a valid {checkpoint_key} tensor: "
            f"{model_path}"
        )
    return input_weights.shape[1]


def calculate_variable_influence(
    model: RecurrentModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictor_names: tuple[str, ...],
    target_parameters: list[dict[str, float]],
    *,
    batch_size: int,
    baseline_mse: float,
) -> tuple[dict[str, float | str], ...]:
    """Measure each predictor's effect as the MSE increase after permutation."""
    if len(inputs) == 0:
        raise ValueError("Cannot calculate variable influence for empty inputs")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not predictor_names:
        raise ValueError("Cannot calculate variable influence without predictors")
    if inputs.shape[2] % len(predictor_names) != 0:
        raise ValueError("Input features are not evenly grouped by predictor")

    features_per_predictor = inputs.shape[2] // len(predictor_names)
    generator = torch.Generator(device="cpu").manual_seed(0)
    influence: list[dict[str, float | str]] = []
    for predictor_index, predictor_name in enumerate(predictor_names):
        permutation = torch.randperm(len(inputs), generator=generator)
        permuted_inputs = inputs.clone()
        start = predictor_index * features_per_predictor
        stop = start + features_per_predictor
        permuted_inputs[:, :, start:stop] = inputs[permutation, :, start:stop]
        permuted_predictions = torch.cat(
            [
                model.predict(permuted_inputs[start_index:start_index + batch_size]).cpu()
                for start_index in range(0, len(inputs), batch_size)
            ],
            dim=0,
        )
        permuted_predictions = denormalize(
            permuted_predictions, target_parameters
        )
        permuted_mse = float(torch.mean((permuted_predictions - targets).square()))
        influence.append(
            {
                "name": predictor_name,
                "mse_increase": permuted_mse - baseline_mse,
                "permuted_mse": permuted_mse,
            }
        )
    return tuple(sorted(influence, key=lambda item: float(item["mse_increase"]), reverse=True))


def print_variable_influence(
    influence: tuple[dict[str, float | str], ...],
) -> None:
    """Print predictors ranked by permutation-induced MSE increase."""
    print("Variable influence (permutation MSE increase):")
    for rank, metrics in enumerate(influence, start=1):
        print(
            f"{rank:>2}. {metrics['name']}: "
            f"MSE increase={float(metrics['mse_increase']):.6f} "
            f"(permuted MSE={float(metrics['permuted_mse']):.6f})"
        )


def evaluate_recurrent(
    model_class: type[RecurrentModel],
    model_path: str | Path,
    data_folder: str | Path,
    *,
    config,
    checkpoint_key: str,
    model_name: str,
    sequence_length: int | None,
    batch_size: int | None,
    hidden_size: int | None,
    num_layers: int | None,
    predictor_contains,
    predictor_exclude,
    predictor_features,
    reference_contains,
    reference_exclude,
    training_normalization_parameters,
    evaluation_normalization_parameters,
    exclude_below,
    exclude_before_samples=None,
    exclude_after_samples=None,
    exclude_before=None,
    exclude_after=None,
) -> RecurrentEvaluation:
    sequence_length = config.sequence_length if sequence_length is None else sequence_length
    batch_size = config.batch_size if batch_size is None else batch_size
    hidden_size = config.hidden_size if hidden_size is None else hidden_size
    num_layers = config.num_layers if num_layers is None else num_layers
    predictor_contains = config.predictor_contains if predictor_contains is None else predictor_contains
    predictor_exclude = config.predictor_exclude if predictor_exclude is None else predictor_exclude
    predictor_features_was_overridden = predictor_features is not None
    predictor_features = config.predictor_features if predictor_features is None else predictor_features
    reference_contains = config.reference_contains if reference_contains is None else reference_contains
    reference_exclude = config.reference_exclude if reference_exclude is None else reference_exclude
    training_normalization_parameters = (
        config.training_normalization_parameters
        if training_normalization_parameters is None
        else training_normalization_parameters
    )
    evaluation_normalization_parameters = (
        config.evaluation_normalization_parameters
        if evaluation_normalization_parameters is None
        else evaluation_normalization_parameters
    )
    exclude_below = config.exclude_below if exclude_below is None else exclude_below
    exclude_before_samples = (
        config.exclude_before_samples
        if exclude_before_samples is None
        else exclude_before_samples
    )
    exclude_after_samples = (
        config.exclude_after_samples
        if exclude_after_samples is None
        else exclude_after_samples
    )
    exclude_before = config.exclude_before if exclude_before is None else exclude_before
    exclude_after = config.exclude_after if exclude_after is None else exclude_after
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    evaluation_path = evaluation_normalization_parameters or Path(data_folder) / "normalization_parameters.json"
    training_path = training_normalization_parameters or evaluation_path
    evaluation_parameters = load_normalization_parameters(evaluation_path)
    training_parameters = load_normalization_parameters(training_path)
    predictor_files = tuple(select_data_files(
        data_folder, filename_contains=predictor_contains, filename_exclude=predictor_exclude
    ))
    reference_files = tuple(select_data_files(
        data_folder, filename_contains=reference_contains, filename_exclude=reference_exclude
    ))
    checkpoint_size = checkpoint_input_size(
        model_path, checkpoint_key=checkpoint_key, model_name=model_name
    )
    configured_size = len(predictor_files) * len(predictor_features)
    if not predictor_features_was_overridden and checkpoint_size != configured_size:
        legacy_features = ("value", "mask", "age")
        if checkpoint_size == len(predictor_files) * len(legacy_features):
            predictor_features = legacy_features
        else:
            raise ValueError(
                f"Checkpoint expects {checkpoint_size} input features, but configured "
                f"evaluation data provides {configured_size}. Pass predictor_features "
                "explicitly for this checkpoint."
            )
    selected_files = (*predictor_files, *reference_files)
    missing = [
        path.name for path in selected_files
        if path.name not in evaluation_parameters or path.name not in training_parameters
    ]
    if missing:
        raise ValueError(f"Missing normalization parameters for: {sorted(set(missing))}")

    evaluation_data = load_rnn_evaluation_data(
        data_folder,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        predictor_features=predictor_features,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        sequence_length=sequence_length,
        exclude_below=exclude_below,
        exclude_before_samples=exclude_before_samples,
        exclude_after_samples=exclude_after_samples,
        exclude_before=exclude_before,
        exclude_after=exclude_after,
    )
    if len(evaluation_data.inputs) == 0:
        raise ValueError("The evaluation dataset contains no complete sequences")

    predictor_parameters = [evaluation_parameters[path.name] for path in predictor_files]
    training_predictor_parameters = [training_parameters[path.name] for path in predictor_files]
    reference_parameters = [evaluation_parameters[path.name] for path in reference_files]
    training_reference_parameters = [training_parameters[path.name] for path in reference_files]
    predictor_names = tuple(path.name for path in predictor_files)
    inputs = evaluation_data.inputs.clone()
    value_index = predictor_features.index("value") if "value" in predictor_features else None
    features_per_predictor = len(predictor_features)
    predictor_values = torch.empty(
        (len(inputs), len(predictor_files)),
        dtype=inputs.dtype,
    )
    if value_index is not None:
        for index, (evaluation_parameter, training_parameter) in enumerate(
            zip(predictor_parameters, training_predictor_parameters)
        ):
            column = index * features_per_predictor + value_index
            values = inputs[:, :, column:column + 1]
            predictor_values[:, index] = denormalize(
                values[:, -1, :], [evaluation_parameter]
            ).squeeze(1)
            inputs[:, :, column:column + 1] = normalize(
                denormalize(values, [evaluation_parameter]), [training_parameter]
            )
    else:
        predictor_values = torch.empty((len(inputs), 0), dtype=inputs.dtype)

    model = model_class(
        input_size=inputs.shape[2],
        hidden_size=hidden_size,
        output_size=len(reference_files),
        num_layers=num_layers,
    )
    model.load_model(model_path)
    predictions = torch.cat(
        [model.predict(inputs[start:start + batch_size]).cpu()
         for start in range(0, len(inputs), batch_size)],
        dim=0,
    )
    predictions = denormalize(predictions, training_reference_parameters)
    targets = denormalize(evaluation_data.targets, reference_parameters)
    errors = predictions - targets
    target_mse = torch.mean(errors.square(), dim=0)
    target_mae = torch.mean(errors.abs(), dim=0)
    target_variance = torch.sum((targets - targets.mean(dim=0)).square(), dim=0)
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
    variable_influence = calculate_variable_influence(
        model,
        inputs,
        targets,
        predictor_names,
        training_reference_parameters,
        batch_size=batch_size,
        baseline_mse=mse,
    )
    return RecurrentEvaluation(
        sample_count=len(targets),
        mse=mse,
        rmse=mse ** 0.5,
        mae=float(torch.mean(errors.abs())),
        r2=float(1 - torch.sum(errors.square()) / torch.sum(
            (targets - targets.mean()).square()
        ).clamp_min(torch.finfo(targets.dtype).eps)),
        target_names=tuple(path.name for path in reference_files),
        predictor_names=predictor_names,
        predictor_values=predictor_values,
        timestamps=evaluation_data.timestamps,
        predictions=predictions,
        targets=targets,
        per_target=per_target,
        variable_influence=variable_influence,
    )


def plot_predictions(
    result: RecurrentEvaluation,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or None")
    sample_count = min(result.sample_count, max_samples or result.sample_count)
    figure, axes = plt.subplots(
        len(result.target_names), 1,
        figsize=(12, 4 * len(result.target_names)),
        squeeze=False,
    )
    timestamps = result.timestamps[:sample_count]
    for index, name in enumerate(result.target_names):
        axis = axes[index, 0]
        axis.plot(timestamps, result.targets[:sample_count, index].numpy(), label="Actual")
        axis.plot(timestamps, result.predictions[:sample_count, index].numpy(), label="Predicted")
        axis.set_title(name)
        axis.set_xlabel("Timestamp")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)


def save_predictions(result: RecurrentEvaluation, output_path: str | Path) -> None:
    values = {"timestamp": result.timestamps}
    for index, name in enumerate(result.target_names):
        values[f"{name}_actual"] = result.targets[:, index].numpy()
        values[f"{name}_predicted"] = result.predictions[:, index].numpy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(output_path, index=False)
