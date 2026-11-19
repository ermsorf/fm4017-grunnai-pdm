from pathlib import Path
from collections.abc import Mapping

import src.config

from src.gru.gru_model import GRUModel
from src.data_loader import load_rnn_data


def run(
    data_folder: str | Path,
    *,
    config: src.config.RecurrentModelConfig = src.config.SEQUENCE_MODEL_CONFIG,
    validation_data_folder: str | Path | None = None,
    predictor_contains: list[str] | tuple[str, ...] | None = None,
    predictor_exclude: list[str] | tuple[str, ...] | None = None,
    predictor_features: list[str] | tuple[str, ...] | None = None,
    reference_contains: list[str] | tuple[str, ...] | None = None,
    reference_exclude: list[str] | tuple[str, ...] | None = None,
    sequence_length: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    early_stopping_patience: int | None = None,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int | None = None,
    exclude_after_samples: int | None = None,
    exclude_before: str | None = None,
    exclude_after: str | None = None,
) -> GRUModel:
    """Load processed sensor data and train a GRU model."""
    sequence_length = config.sequence_length if sequence_length is None else sequence_length
    hidden_size = config.hidden_size if hidden_size is None else hidden_size
    num_layers = config.num_layers if num_layers is None else num_layers
    num_epochs = config.num_epochs if num_epochs is None else num_epochs
    batch_size = config.batch_size if batch_size is None else batch_size
    learning_rate = config.learning_rate if learning_rate is None else learning_rate
    early_stopping_patience = (
        config.early_stopping_patience
        if early_stopping_patience is None
        else early_stopping_patience
    )
    predictor_contains = (
        config.predictor_contains if predictor_contains is None else predictor_contains
    )
    predictor_exclude = (
        config.predictor_exclude if predictor_exclude is None else predictor_exclude
    )
    reference_contains = (
        config.reference_contains if reference_contains is None else reference_contains
    )
    reference_exclude = (
        config.reference_exclude if reference_exclude is None else reference_exclude
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

    predictor_features = config.predictor_features if predictor_features is None else predictor_features

    print(f"Loading data from {data_folder} with sequence length {sequence_length}")
    data = load_rnn_data(
        data_folder,
        validation_data_folder=(
            config.validation_data_folder
            if validation_data_folder is None
            else validation_data_folder
        ),
        training_normalization_parameters=config.training_normalization_parameters,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        sequence_length=sequence_length,
        predictor_features=predictor_features,
        exclude_below=exclude_below,
        exclude_before_samples=exclude_before_samples,
        exclude_after_samples=exclude_after_samples,
        exclude_before=exclude_before,
        exclude_after=exclude_after,
    )
    model = GRUModel(
        input_size=data.train.inputs.shape[2],
        hidden_size=hidden_size,
        output_size=data.train.targets.shape[1],
        num_layers=num_layers,
    )
    model.fit(
        data.train.inputs,
        data.train.targets,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_data=(data.validation.inputs, data.validation.targets),
        early_stopping_patience=early_stopping_patience,
    )
    return model


if __name__ == "__main__":
    folder = Path(src.config.DATA_PROCESSED) / "2024-06-02_2024-10-31_2NI"
    model = run(folder)
    model_path = Path(src.config.MODEL_SAVE_PATH) / "gru" / "gru_vibr1.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"Saved GRU model to {model_path}")
