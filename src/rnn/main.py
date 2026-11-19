from pathlib import Path
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import src.config

from src.data_loader import load_rnn_data
from src.rnn.model import RNNModel


@dataclass(frozen=True)
class RNNGridSearchTrial:
    """Result from one RNN hyperparameter combination."""

    sequence_length: int
    hidden_size: int
    num_layers: int
    validation_loss: float


@dataclass(frozen=True)
class RNNGridSearchResult:
    """Best model and validation results from an RNN grid search."""

    model: RNNModel
    best_sequence_length: int
    best_hidden_size: int
    best_num_layers: int
    trials: tuple[RNNGridSearchTrial, ...]


def run(
    data_folder: str | Path,
    *,
    validation_data_folder: str | Path | None = None,
    predictor_contains: list[str] | None = None,
    predictor_exclude: list[str] | None = None,
    reference_contains: list[str] | None = None,
    reference_exclude: list[str] | None = None,
    sequence_length: int = 4,
    hidden_size: int = 16,
    num_layers: int = 4,
    num_epochs: int = 50,
    batch_size: int = 1024,
    early_stopping_patience: int | None = 3,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int = 0,
    exclude_after_samples: int = 0,
    exclude_before: str | None = None,
    exclude_after: str | None = None,
) -> RNNModel:
    print(f"Loading data from {data_folder} with sequence length {sequence_length}")
    data = load_rnn_data(
        data_folder,
        validation_data_folder=validation_data_folder,
        predictor_contains=predictor_contains,
        predictor_exclude=predictor_exclude,
        reference_contains=reference_contains,
        reference_exclude=reference_exclude,
        sequence_length=sequence_length,
        exclude_below=exclude_below,
        exclude_before_samples=exclude_before_samples,
        exclude_after_samples=exclude_after_samples,
        exclude_before=exclude_before,
        exclude_after=exclude_after,
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


def grid_search(
    data_folder: str | Path,
    *,
    sequence_lengths: Iterable[int],
    hidden_sizes: Iterable[int],
    num_layers_values: Iterable[int],
    predictor_contains: list[str] | None = None,
    predictor_exclude: list[str] | None = None,
    reference_contains: list[str] | None = None,
    reference_exclude: list[str] | None = None,
    num_epochs: int = 20,
    batch_size: int = 1024,
    early_stopping_patience: int | None = 3,
    exclude_below: Mapping[str, float] | None = None,
    exclude_before_samples: int = 0,
    exclude_after_samples: int = 0,
    exclude_before: str | None = None,
    exclude_after: str | None = None,
) -> RNNGridSearchResult:
    """Train every hyperparameter combination and return the best model.

    The validation loss from each trial's training history is used for
    selection. Grid values are materialized once so generators are supported
    and empty or invalid grids fail before any training starts.
    """
    def validate_grid(name: str, values: Iterable[int]) -> tuple[int, ...]:
        materialized = tuple(values)
        if not materialized:
            raise ValueError(f"{name} must contain at least one value")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in materialized):
            raise ValueError(f"{name} must contain positive integers")
        return materialized

    sequence_lengths = validate_grid("sequence_lengths", sequence_lengths)
    hidden_sizes = validate_grid("hidden_sizes", hidden_sizes)
    num_layers_values = validate_grid("num_layers_values", num_layers_values)

    trials: list[RNNGridSearchTrial] = []
    best_model: RNNModel | None = None
    best_trial: RNNGridSearchTrial | None = None
    for sequence_length in sequence_lengths:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_values:
                model = run(
                    data_folder,
                    predictor_contains=predictor_contains,
                    predictor_exclude=predictor_exclude,
                    reference_contains=reference_contains,
                    reference_exclude=reference_exclude,
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    early_stopping_patience=early_stopping_patience,
                    exclude_below=exclude_below,
                    exclude_before_samples=exclude_before_samples,
                    exclude_after_samples=exclude_after_samples,
                    exclude_before=exclude_before,
                    exclude_after=exclude_after,
                )
                if model.training_history is None or not model.training_history["validation_loss"]:
                    raise ValueError("Each grid-search trial must produce validation loss values")
                trial = RNNGridSearchTrial(
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    validation_loss=min(model.training_history["validation_loss"]),
                )
                trials.append(trial)
                if best_trial is None or trial.validation_loss < best_trial.validation_loss:
                    best_trial = trial
                    best_model = model

    if best_trial is None or best_model is None:
        raise RuntimeError("RNN grid search completed without any trials")
    print(
        "Best RNN configuration: "
        f"sequence length={best_trial.sequence_length}, "
        f"hidden size={best_trial.hidden_size}, "
        f"layers={best_trial.num_layers}, "
        f"validation loss={best_trial.validation_loss:.6f}"
    )
    return RNNGridSearchResult(
        model=best_model,
        best_sequence_length=best_trial.sequence_length,
        best_hidden_size=best_trial.hidden_size,
        best_num_layers=best_trial.num_layers,
        trials=tuple(trials),
    )


if __name__ == "__main__":
    folder = Path(src.config.DATA_PROCESSED) / "2024-06-02_2024-10-31_2NI"
    model = run(
        folder,
        # predictor_contains=["G2", "Hydrocord"],
        predictor_contains=["MW", "NALER", "TURT", "VANNSTR"],
        predictor_exclude=["VIBR", "TMP"],
        reference_contains=["VIBR2"],
    )
    model.save_model(Path(src.config.MODEL_SAVE_PATH) / "rnn_model.pth")

    # result = grid_search(
    #     folder,
    #     sequence_lengths=[4,8,16,32],
    #     hidden_sizes=[16, 32, 64],
    #     num_layers_values=[2, 4, 6, 8],
    #     predictor_contains=["MW", "NALER", "TURT", "VANNSTR"],
    #     predictor_exclude=["VIBR", "TMP"],
    #     reference_contains=["VIBR1"],
    # )

    # best_model = result.model
