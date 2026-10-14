from pathlib import Path

import src.config

from src.rnn.data import load_rnn_data
from src.rnn.model import RNNModel


def run(
    data_folder: str | Path,
    *,
    predictor_contains: list[str] | None = None,
    predictor_exclude: list[str] | None = None,
    reference_contains: list[str] | None = None,
    reference_exclude: list[str] | None = None,
    sequence_length: int = 2,
    hidden_size: int = 16,
    num_layers: int = 4,
    num_epochs: int = 20,
    batch_size: int = 256,
    early_stopping_patience: int | None = 5,
) -> RNNModel:
    print(f"Loading data from {data_folder} with sequence length {sequence_length}")
    data = load_rnn_data(
        data_folder,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        sequence_length=sequence_length,
    )
    model = RNNModel(
        input_size=data.train.inputs.shape[2],
        hidden_size=hidden_size,
        output_size=data.train.targets.shape[1],
        num_layers=num_layers,
    )
    print(f"Training RNN model with {hidden_size} hidden units and {num_layers} layers for {num_epochs} epochs")
    model.training_history = model.fit(
        data.train.inputs,
        data.train.targets,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(data.validation.inputs, data.validation.targets),
        early_stopping_patience=early_stopping_patience,
    )
    return model


if __name__ == "__main__":
    folder = Path(src.config.DATA_PROCESSED) / "2024-06-02_2024-10-31_stepinterp_10s"
    model = run(
        folder,
        # predictor_contains=["G2", "Hydrocord"],
        predictor_contains=["VANNSTR", "NALER"],
        predictor_exclude=["VIBR", "TMP"],
        reference_contains=["VIBR"],
    )
    model.save_model(Path(src.config.MODEL_SAVE_PATH) / "rnn_model.pth")
