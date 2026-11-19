from pathlib import Path
from collections.abc import Mapping
from typing import TypedDict

import torch
import src.config


class TrainingHistory(TypedDict):
    train_loss: list[float]
    validation_loss: list[float]


class LSTMModel(torch.nn.Module):
    """LSTM regressor for windowed sensor data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.training_history: TrainingHistory | None = None
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size),
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-enabled PyTorch device is available"
            )
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.device = requested_device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        cell = torch.zeros_like(hidden)
        output, _ = self.lstm(x, (hidden, cell))
        return self.fc(output[:, -1, :])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(x.to(self.device))

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        num_epochs: int = 100,
        learning_rate: float = 0.0005,
        batch_size: int = 256,
        validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        early_stopping_patience: int | None = 4,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> TrainingHistory:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if early_stopping_patience is not None and early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative or None")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if len(x_train) != len(y_train):
            raise ValueError("training inputs and targets must have the same length")
        if len(x_train) == 0:
            raise ValueError("training data must not be empty")
        if validation_data is not None and len(validation_data[0]) != len(validation_data[1]):
            raise ValueError("validation inputs and targets must have the same length")
        if validation_data is not None and len(validation_data[0]) == 0:
            raise ValueError("validation data must not be empty")

        # criterion = torch.nn.HuberLoss(delta=0.5)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate
        )
        batches = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )
        history: TrainingHistory = {"train_loss": [], "validation_loss": []}
        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            super().train()
            total_loss = 0.0
            for inputs, targets in batches:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(self(inputs), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(inputs)
            mean_loss = total_loss / len(batches.dataset)
            history["train_loss"].append(mean_loss)

            validation_loss = None
            validation_improved = False
            if validation_data is not None:
                validation_inputs, validation_targets = validation_data
                super().eval()
                with torch.no_grad():
                    validation_loss_total = 0.0
                    for start in range(0, len(validation_inputs), batch_size):
                        stop = start + batch_size
                        validation_predictions = self(
                            validation_inputs[start:stop].to(self.device)
                        )
                        validation_loss_total += criterion(
                            validation_predictions,
                            validation_targets[start:stop].to(self.device),
                        ).item() * len(validation_predictions)
                    validation_loss = validation_loss_total / len(validation_inputs)
                history["validation_loss"].append(validation_loss)

                if validation_loss < best_validation_loss - min_delta:
                    validation_improved = True
                    best_validation_loss = validation_loss
                    best_state = {
                        name: parameter.detach().clone()
                        for name, parameter in self.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            message = f"Epoch [{epoch + 1}/{num_epochs}], Train loss: {mean_loss:.6f}"
            if validation_loss is not None:
                message += f", Validation loss: {validation_loss:.6f}"
            print(message)

            if (
                validation_data is not None
                and early_stopping_patience is not None
                and not validation_improved
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None and restore_best_weights:
            self.load_state_dict(best_state)
        self.training_history = history
        return history

    def save_model(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str | Path) -> None:
        self.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.eval()

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
) -> LSTMModel:
    """Load processed sensor data and train an LSTM model."""
    from src.data_loader import load_rnn_data

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
    predictor_features = (
        config.predictor_features if predictor_features is None else predictor_features
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
    model = LSTMModel(
        input_size=data.train.inputs.shape[2],
        hidden_size=hidden_size,
        output_size=data.train.targets.shape[1],
        num_layers=num_layers,
    )
    print(
        f"Training LSTM model with {hidden_size} hidden units and "
        f"{num_layers} layers for {num_epochs} epochs"
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
    import src.config

    model = run(Path(src.config.SEQUENCE_MODEL_CONFIG.training_data_folder))
    model_path = Path(src.config.MODEL_SAVE_PATH) / "lstm" /"lstm_vibr1.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"Saved LSTM model to {model_path}")
    