"""Feed-forward neural network for per-timestep sensor prediction."""

from pathlib import Path
from typing import TypedDict

import torch


class TrainingHistory(TypedDict):
    train_loss: list[float]
    validation_loss: list[float]


class ANNModel(torch.nn.Module):
    """A multilayer perceptron that predicts targets from one timestamp."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if input_size < 1 or output_size < 1:
            raise ValueError("input_size and output_size must be positive")
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        layers: list[torch.nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [torch.nn.Linear(previous_size, hidden_size), torch.nn.ReLU()]
            )
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(torch.nn.Linear(previous_size, output_size))
        self.network = torch.nn.Sequential(*layers)
        self.training_history: TrainingHistory | None = None

        requested_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-enabled PyTorch device is available")
        self.device = requested_device
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(inputs.to(self.device)).cpu()

    def fit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        early_stopping_patience: int | None = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> TrainingHistory:
        if len(inputs) == 0:
            raise ValueError("training data must not be empty")
        if len(inputs) != len(targets):
            raise ValueError("training inputs and targets must have the same length")
        if num_epochs < 1 or learning_rate <= 0 or batch_size < 1:
            raise ValueError("num_epochs, learning_rate, and batch_size must be positive")
        if early_stopping_patience is not None and early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative or None")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if validation_data is not None:
            if len(validation_data[0]) != len(validation_data[1]):
                raise ValueError("validation inputs and targets must have the same length")
            if len(validation_data[0]) == 0:
                raise ValueError("validation data must not be empty")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        batches = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets),
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
            for batch_inputs, batch_targets in batches:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(self(batch_inputs), batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_inputs)
            train_loss = total_loss / len(batches.dataset)
            history["train_loss"].append(train_loss)

            validation_loss = None
            validation_improved = False
            if validation_data is not None:
                super().eval()
                validation_inputs, validation_targets = validation_data
                with torch.no_grad():
                    validation_loss = criterion(
                        self(validation_inputs.to(self.device)),
                        validation_targets.to(self.device),
                    ).item()
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

            message = f"Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_loss:.4f}"
            if validation_loss is not None:
                message += f", Validation loss: {validation_loss:.4f}"
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
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
