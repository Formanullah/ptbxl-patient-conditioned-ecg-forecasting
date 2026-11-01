"""Dynamic encoders, late static fusion, and patient-masked NT-Xent."""

import torch
from torch import nn
from torch.nn import functional as F


class Forecaster(nn.Module):
    def __init__(
        self,
        architecture,
        static_dim,
        horizon=150,
        hidden=128,
        depth=2,
        dropout=0.2,
        static_embedding=32,
    ):
        super().__init__()
        if architecture not in ("lstm", "cnn_lstm"):
            raise ValueError("Unknown architecture")
        self.horizon = horizon
        self.cnn = (
            nn.Identity()
            if architecture == "lstm"
            else nn.Sequential(
                nn.Conv1d(12, 32, 7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.ReLU(),
                nn.AvgPool1d(2),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.ReLU(),
            )
        )
        self.rnn = nn.LSTM(
            12 if architecture == "lstm" else 64,
            hidden,
            num_layers=depth,
            dropout=dropout if depth > 1 else 0,
            batch_first=True,
        )
        self.static_net = (
            nn.Sequential(nn.Linear(static_dim, static_embedding), nn.ReLU())
            if static_dim
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + (static_embedding if static_dim else 0), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 12 * horizon),
        )
        self.projector = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 64))

    def encode(self, x):
        _, (h, _) = self.rnn(self.cnn(x).transpose(1, 2))
        return h[-1]

    def forward(self, x, static):
        h = self.encode(x)
        if self.static_net is not None:
            h = torch.cat([h, self.static_net(static)], dim=-1)
        return self.head(h).reshape(-1, 12, self.horizon)

    def project(self, x):
        return F.normalize(self.projector(self.encode(x)), dim=-1)


def build_model(cfg, architecture, static_dim):
    return Forecaster(architecture, static_dim, horizon=cfg["data"]["horizon"], **cfg["model"])


def augment(x, settings):
    gain = torch.empty((len(x), 1, 1), device=x.device).uniform_(
        settings["gain_min"], settings["gain_max"]
    )
    return x * gain + torch.randn_like(x) * settings["jitter_std"]


def nt_xent(a, b, patients, temperature=0.1):
    """Return loss and valid anchor count, excluding other same-patient views."""
    n = len(a)
    z = F.normalize(torch.cat([a, b]), dim=-1)
    patients = patients.to(z.device).repeat(2)
    idx = torch.arange(2 * n, device=z.device)
    positive = (idx + n) % (2 * n)
    different = patients[:, None] != patients[None, :]
    allowed = different.clone()
    allowed[idx, positive] = True
    valid = different.any(dim=1)
    if not valid.any():
        return z.sum() * 0, 0
    logits = (z @ z.T) / temperature
    logits = logits.masked_fill(~allowed, -torch.inf)
    losses = -logits[idx, positive] + torch.logsumexp(logits, dim=1)
    return losses[valid].mean(), int(valid.sum())
