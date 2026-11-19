from pathlib import Path

import torch

from src.lstm.lstm_model import LSTMModel, TrainingHistory


class GRUModel(LSTMModel):
    """GRU regressor with the same training and persistence API as LSTMModel."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        device: str | torch.device | None = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            device=device,
        )
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.gru.to(self.device)
        del self.lstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        output, _ = self.gru(x, hidden)
        return self.fc(output[:, -1, :])
