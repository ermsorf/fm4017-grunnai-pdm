from pathlib import Path
from typing import TypedDict

import torch


class TrainingHistory(TypedDict):
    train_loss: list[float]
    validation_loss: list[float]


class RNNModel(torch.nn.Module):
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
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-enabled PyTorch device is available")
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.device = requested_device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype
        )
        output, _ = self.rnn(x, hidden)
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
        learning_rate: float = 0.001,
        batch_size: int = 256,
        validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        early_stopping_patience: int | None = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> TrainingHistory:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if early_stopping_patience is not None and early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative or None")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if validation_data is not None and len(validation_data[0]) != len(validation_data[1]):
            raise ValueError("validation inputs and targets must have the same length")
        if validation_data is not None and len(validation_data[0]) == 0:
            raise ValueError("validation data must not be empty")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        batches = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )
        history: TrainingHistory = {"train_loss": [], "validation_loss": []}
        best_validation_loss = float("inf")
        best_state = None
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
                    validation_predictions = self(validation_inputs.to(self.device))
                    validation_loss = criterion(
                        validation_predictions,
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

            message = f"Epoch [{epoch + 1}/{num_epochs}], Train loss: {mean_loss:.4f}"
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
