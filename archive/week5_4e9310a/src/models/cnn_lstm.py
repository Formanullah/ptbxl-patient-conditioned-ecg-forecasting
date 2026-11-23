"""Temporal convolutions followed by an LSTM and a direct forecast head."""

import torch
from torch import nn


class CNNLSTMForecaster(nn.Module):
    """Extract local observed patterns before recurrent forecasting."""

    def __init__(
        self,
        static_dim=0,
        horizon=300,
        hidden=64,
        depth=1,
        dropout=0.2,
        static_embedding=16,
        head_width=128,
    ):
        super().__init__()
        self.horizon = horizon
        self.static = (
            nn.Sequential(nn.Linear(static_dim, static_embedding), nn.ReLU())
            if static_dim
            else None
        )
        channels = 12 + (static_embedding if static_dim else 0)
        self.convolutions = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        self.lstm = nn.LSTM(
            64, hidden, num_layers=depth, batch_first=True, dropout=dropout if depth > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, head_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, horizon * 12),
        )

    def forward(self, history, static=None):
        if history.ndim != 3 or history.shape[-1] != 12:
            raise ValueError("Expected history [batch,time,12]")
        if self.static is not None:
            if static is None:
                raise ValueError("This model requires static inputs")
            embedding = self.static(static)[:, None, :].expand(-1, history.shape[1], -1)
            history = torch.cat((history, embedding), dim=-1)
        features = self.convolutions(history.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(features)
        return self.head(hidden[-1]).reshape(-1, self.horizon, 12)
