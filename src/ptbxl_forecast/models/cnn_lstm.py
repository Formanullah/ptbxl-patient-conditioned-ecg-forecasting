"""CNN-LSTM direct forecaster (Phase 9): a strided convolutional stem shortens the history
4x (700 -> 175 steps) before a stacked LSTM; linear head from the last hidden state."""

from __future__ import annotations

import torch
from torch import nn


class CNNLSTMForecaster(nn.Module):
    def __init__(
        self, in_channels: int = 24, out_leads: int = 12, horizon: int = 290,
        width: int = 64, hidden: int = 128, layers: int = 2,
    ):  # fmt: skip
        super().__init__()
        self.horizon, self.out_leads = horizon, out_leads
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 7, stride=2, padding=3), nn.GELU(),
        )  # fmt: skip
        self.lstm = nn.LSTM(width, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon * out_leads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1]).view(-1, self.horizon, self.out_leads)
