"""CNN-LSTM conditioned on a causal baseline forecast (Phase 10, ``sn4`` residual).

    forecast = sigmoid(gate(h)) * baseline + correction(h)

``h`` is the CNN-LSTM's summary of the history. The gate (per horizon step and lead) says
how far to trust the repeated cycles; the correction is the plain CNN-LSTM head. A
closed gate gives the Phase 9 CNN-LSTM, an open gate with zero correction gives the
baseline. Input: ``preprocessing.BaselineConditioned`` channels, i.e. the history
channels followed by ``out_leads`` channels whose last ``horizon`` steps hold the
baseline forecast.
"""

from __future__ import annotations

import torch
from torch import nn

from .cnn_lstm import CNNLSTMForecaster


class GatedBaselineCNNLSTM(nn.Module):
    def __init__(
        self, in_channels: int = 24, out_leads: int = 12, horizon: int = 290,
        width: int = 64, hidden: int = 128, layers: int = 2,
    ):  # fmt: skip
        super().__init__()
        self.in_channels, self.horizon, self.out_leads = in_channels, horizon, out_leads
        self.body = CNNLSTMForecaster(in_channels, out_leads, horizon, width, hidden, layers)
        self.gate = nn.Linear(hidden, horizon * out_leads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hist, base = x[..., : self.in_channels], x[:, -self.horizon :, self.in_channels :]
        b = self.body
        z = b.stem(hist.transpose(1, 2)).transpose(1, 2)
        out, _ = b.lstm(z)
        h = out[:, -1]
        gate = torch.sigmoid(self.gate(h)).view(-1, self.horizon, self.out_leads)
        return gate * base + b.head(h).view(-1, self.horizon, self.out_leads)
