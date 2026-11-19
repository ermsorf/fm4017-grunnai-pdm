"""Evaluate a saved feed-forward ANN and rank input variable influence."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

import src.config
from src.ann.ann_model import ANNModel
from src.ann.ann_train import _numeric_frame, _transform
from src.data_loader import load_tabular_data


@dataclass(frozen=True)
class ANNEvaluation:
    sample_count: int
    mse: float
    rmse: float
    mae: float
    r2: float
    target_names: tuple[str, ...]
    predictor_names: tuple[str, ...]
    predictions: torch.Tensor
    targets: torch.Tensor
    variable_influence: tuple[dict[str, float | str], ...]


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float, float]:
    errors = predictions - targets
    mse = float(errors.square().mean())
    denominator = torch.sum((targets - targets.mean()).square()).clamp_min(
        torch.finfo(targets.dtype).eps
    )
    return mse, mse**0.5, float(errors.abs().mean()), float(
        1 - errors.square().sum() / denominator
    )


def evaluate(
    model_path: str | Path,
    data_folder: str | Path,
    *,
    config: src.config.ANNModelConfig = src.config.ANN_MODEL_CONFIG,
) -> ANNEvaluation:
    """Evaluate the checkpoint on the chronological held-out test split."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
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
    expected_input_size = len(data.test.inputs.columns)
    if checkpoint["input_size"] != expected_input_size:
        raise ValueError(
            "ANN checkpoint expects "
            f"{checkpoint['input_size']} inputs, but the configured data provides "
            f"{expected_input_size}; retrain the checkpoint with the current ANN configuration"
        )
    model = ANNModel(
        checkpoint["input_size"],
        checkpoint["output_size"],
        hidden_sizes=tuple(checkpoint["hidden_sizes"]),
        dropout=float(checkpoint["dropout"]),
        device="cpu",
    )
    model.load_state_dict(checkpoint["state_dict"])
    inputs = _transform(
        _numeric_frame(data.test.inputs),
        checkpoint["input_means"],
        checkpoint["input_stds"],
    )
    targets = _numeric_frame(data.test.targets)
    predictions = model.predict(inputs) * checkpoint["target_stds"] + checkpoint["target_means"]
    mse, rmse, mae, r2 = _metrics(predictions, targets)

    influence: list[dict[str, float | str]] = []
    generator = torch.Generator().manual_seed(0)
    for column, name in enumerate(data.test.inputs.columns):
        permutation = torch.randperm(len(inputs), generator=generator)
        permuted = inputs.clone()
        permuted[:, column] = inputs[permutation, column]
        permuted_predictions = (
            model.predict(permuted) * checkpoint["target_stds"] + checkpoint["target_means"]
        )
        permuted_mse = float((permuted_predictions - targets).square().mean())
        influence.append({
            "name": name,
            "mse_increase": permuted_mse - mse,
            "permuted_mse": permuted_mse,
        })

    return ANNEvaluation(
        sample_count=len(targets),
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        target_names=tuple(data.test.targets.columns),
        predictor_names=tuple(data.test.inputs.columns),
        predictions=predictions,
        targets=targets,
        variable_influence=tuple(sorted(
            influence,
            key=lambda item: float(item["mse_increase"]),
            reverse=True,
        )),
    )


def save_predictions(result: ANNEvaluation, output_path: str | Path) -> None:
    """Save test predictions; tabular splits intentionally have no timestamps."""
    values = {"sample": range(result.sample_count)}
    for index, name in enumerate(result.target_names):
        values[f"{name}_actual"] = result.targets[:, index].numpy()
        values[f"{name}_predicted"] = result.predictions[:, index].numpy()
    pd.DataFrame(values).to_csv(output_path, index=False)