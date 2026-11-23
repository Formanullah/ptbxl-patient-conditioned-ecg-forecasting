"""LSTM direct forecaster (Phase 9): the full 700-step history through an LSTM, then a
linear head from the last hidden state to all horizon steps at once."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self, in_channels: int = 24, out_leads: int = 12, horizon: int = 290,
        hidden: int = 64, layers: int = 1,
    ):  # fmt: skip
        super().__init__()
        self.horizon, self.out_leads = horizon, out_leads
        self.lstm = nn.LSTM(in_channels, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon * out_leads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (batch, time, channels) -> (batch, horizon, out_leads)."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).view(-1, self.horizon, self.out_leads)
