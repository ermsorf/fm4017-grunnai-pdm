"""Train a feed-forward ANN on per-timestep sensor data."""

from pathlib import Path

import pandas as pd
import torch

import src.config
from src.ann.ann_model import ANNModel
from src.data_loader import TabularData, load_tabular_data


def _numeric_frame(frame: pd.DataFrame) -> torch.Tensor:
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype="float32", copy=True
    )
    return torch.from_numpy(values)


def _fit_scalers(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_means = torch.nan_to_num(torch.nanmean(inputs, dim=0), nan=0.0)
    target_means = torch.nan_to_num(torch.nanmean(targets, dim=0), nan=0.0)
    input_valid = torch.isfinite(inputs)
    target_valid = torch.isfinite(targets)
    input_centered = torch.where(input_valid, inputs - input_means, 0.0)
    target_centered = torch.where(target_valid, targets - target_means, 0.0)
    input_stds = torch.sqrt(
        input_centered.square().sum(dim=0) / input_valid.sum(dim=0).clamp_min(1)
    ).clamp_min(1e-6)
    target_stds = torch.sqrt(
        target_centered.square().sum(dim=0) / target_valid.sum(dim=0).clamp_min(1)
    ).clamp_min(1e-6)
    return input_means, input_stds, target_means, target_stds


def _transform(
    values: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
) -> torch.Tensor:
    return torch.nan_to_num(
        (values - means) / stds, nan=0.0, posinf=0.0, neginf=0.0
    )


def train(
    data: TabularData,
    *,
    config: src.config.ANNModelConfig = src.config.ANN_MODEL_CONFIG,
) -> tuple[ANNModel, dict[str, torch.Tensor]]:
    """Fit an ANN and return it with the train-fitted scaling parameters."""
    train_inputs = _numeric_frame(data.train.inputs)
    validation_inputs = _numeric_frame(data.validation.inputs)
    train_targets = _numeric_frame(data.train.targets)
    validation_targets = _numeric_frame(data.validation.targets)
    if len(train_inputs) < 2 or len(validation_inputs) == 0:
        raise ValueError("Not enough usable rows to train the ANN")

    scalers = dict(
        zip(
            ("input_means", "input_stds", "target_means", "target_stds"),
            _fit_scalers(train_inputs, train_targets),
        )
    )
    model = ANNModel(
        train_inputs.shape[1],
        train_targets.shape[1],
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        device=config.device,
    )
    model.fit(
        _transform(train_inputs, scalers["input_means"], scalers["input_stds"]),
        _transform(train_targets, scalers["target_means"], scalers["target_stds"]),
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        validation_data=(
            _transform(
                validation_inputs,
                scalers["input_means"],
                scalers["input_stds"],
            ),
            _transform(
                validation_targets,
                scalers["target_means"],
                scalers["target_stds"],
            ),
        ),
        early_stopping_patience=config.early_stopping_patience,
    )
    return model, scalers


def run(
    data_folder: str | Path | None = None,
    *,
    config: src.config.ANNModelConfig = src.config.ANN_MODEL_CONFIG,
) -> tuple[ANNModel, dict[str, torch.Tensor]]:
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
    return train(data, config=config)


if __name__ == "__main__":
    model, scalers = run()
    output = Path(src.config.MODEL_SAVE_PATH) / "ann_model.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    config = src.config.ANN_MODEL_CONFIG
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": model.network[0].in_features,
            "output_size": model.network[-1].out_features,
            "hidden_sizes": config.hidden_sizes,
            "dropout": config.dropout,
            **scalers,
        },
        output,
    )
    print(f"Saved ANN model to {output}")
