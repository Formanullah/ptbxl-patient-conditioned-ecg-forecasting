"""LSTM/CNN-LSTM encoders, early/late static fusion, and patient-masked NT-Xent."""

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
        fusion="late",
        input_layout="channels_first",
        kernels=(7, 5, 3),
        pool=2,
        residual=False,
        head_width=256,
    ):
        super().__init__()
        if architecture not in ("lstm", "cnn_lstm"):
            raise ValueError("Unknown architecture")
        self.horizon = horizon
        if fusion not in ("late", "early") or input_layout not in ("time_first", "channels_first"):
            raise ValueError("Unknown fusion or input layout")
        if len(kernels) != 3 or any(k < 1 or k % 2 == 0 for k in kernels) or pool < 1:
            raise ValueError("Three odd positive kernels and positive pooling are required")
        self.fusion, self.input_layout, self.residual = fusion, input_layout, residual
        channels = 12 + (static_embedding if static_dim and fusion == "early" else 0)
        self.cnn = (
            nn.Identity()
            if architecture == "lstm"
            else nn.Sequential(
                nn.Conv1d(channels, 32, kernels[0], padding=kernels[0] // 2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernels[1], padding=kernels[1] // 2),
                nn.ReLU(),
                nn.AvgPool1d(pool),
                nn.Conv1d(64, 64, kernels[2], padding=kernels[2] // 2),
                nn.ReLU(),
            )
        )
        self.rnn = nn.LSTM(
            channels if architecture == "lstm" else 64,
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
            nn.Linear(
                hidden + (static_embedding if static_dim and fusion == "late" else 0), head_width
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 12 * horizon),
        )
        self.projector = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 64))

    def encode(self, x, static=None):
        if self.input_layout == "time_first":
            x = x.transpose(1, 2)
        if self.static_net is not None and self.fusion == "early":
            if static is None:
                # Contrastive projection deliberately excludes patient metadata.
                embedding = x.new_zeros((len(x), self.static_net[0].out_features))
            else:
                embedding = self.static_net(static)
            x = torch.cat([x, embedding[:, :, None].expand(-1, -1, x.shape[-1])], dim=1)
        _, (h, _) = self.rnn(self.cnn(x).transpose(1, 2))
        return h[-1]

    def forward(self, x, static):
        h = self.encode(x, static)
        if self.static_net is not None and self.fusion == "late":
            h = torch.cat([h, self.static_net(static)], dim=-1)
        result = self.head(h).reshape(-1, 12, self.horizon)
        if self.residual:
            last = x[:, -1, :, None] if self.input_layout == "time_first" else x[:, :, -1:]
            result = result + last
        return result.transpose(1, 2) if self.input_layout == "time_first" else result

    def project(self, x):
        return F.normalize(self.projector(self.encode(x)), dim=-1)


def build_model(cfg, architecture, static_dim):
    return Forecaster(
        architecture,
        static_dim,
        horizon=cfg["data"]["horizon"],
        input_layout=cfg["data"].get("layout", "channels_first"),
        **cfg["model"],
    )


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
