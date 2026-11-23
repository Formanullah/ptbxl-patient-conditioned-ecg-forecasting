"""Small 1-D CNN direct forecaster (Phase 5 reference learner; about 1.8 M parameters)."""

from __future__ import annotations

import torch
from torch import nn


class CNNForecaster(nn.Module):
    def __init__(
        self, in_channels: int = 12, out_leads: int = 12, horizon: int = 290, width: int = 64
    ):
        super().__init__()
        self.horizon, self.out_leads = horizon, out_leads
        blocks, c = [], in_channels
        for _ in range(3):
            blocks += [nn.Conv1d(c, width, 7, stride=2, padding=3), nn.GELU()]
            c = width
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(8))
        self.head = nn.Linear(width * 8, horizon * out_leads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (batch, time, channels) -> (batch, horizon, out_leads)."""
        z = self.encoder(x.transpose(1, 2)).flatten(1)
        return self.head(z).view(-1, self.horizon, self.out_leads)
