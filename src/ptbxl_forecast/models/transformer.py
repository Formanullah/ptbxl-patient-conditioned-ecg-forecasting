"""Channel-mixing encoder-decoder Transformer direct forecaster (Phase 9).

The history is cut into non-overlapping 100 ms tokens holding all channels (70 tokens).
A Transformer encoder reads them; a decoder with one learned query per 100 ms of the
horizon (29 queries) cross-attends to the encoder, and each query is mapped to its 10 x
leads output samples. All horizon steps are predicted at once (non-autoregressive).
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerForecaster(nn.Module):
    def __init__(
        self, in_channels: int = 24, out_leads: int = 12, horizon: int = 290, history: int = 700,
        token: int = 10, d_model: int = 128, heads: int = 8, enc_layers: int = 3,
        dec_layers: int = 2, ff: int = 512, dropout: float = 0.1,
    ):  # fmt: skip
        super().__init__()
        if history % token or horizon % token:
            raise ValueError("history and horizon must be multiples of the token length")
        self.horizon, self.out_leads, self.token = horizon, out_leads, token
        self.embed = nn.Linear(token * in_channels, d_model)
        self.pos = nn.Parameter(torch.empty(history // token, d_model))
        self.queries = nn.Parameter(torch.empty(horizon // token, d_model))
        self.transformer = nn.Transformer(
            d_model, heads, enc_layers, dec_layers, ff, dropout, batch_first=True, norm_first=True
        )
        self.out = nn.Linear(d_model, token * out_leads)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        src = self.embed(x.reshape(b, t // self.token, self.token * c)) + self.pos
        tgt = self.queries.expand(b, -1, -1)
        h = self.transformer(src, tgt)
        return self.out(h).view(b, self.horizon, self.out_leads)
