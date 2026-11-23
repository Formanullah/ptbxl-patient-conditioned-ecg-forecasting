"""PatchTST direct forecaster (Phase 9; Nie et al., ICLR 2023), adapted to multi-channel
leads.

Channel-independent: every lead is an independent sequence through a shared Transformer
encoder over patches of its history. With the decomposition input (baseline + residual
per lead, D5.11) a lead's patch holds both of its channels. RevIN (per-lead history
mean/SD, from the input only, so causal) normalizes the input and is inverted on the
output; the raw lead is the sum of its channels, so the mean is removed from the first
(baseline or raw) channel and every channel is divided by the SD.

Defaults are reduced for CPU (Phase 9): 100 ms non-overlapping patches, d_model 64,
2 layers. The published setting (patch 16, stride 8, d_model 128, 3 layers) costs
about 1,000 s/epoch on 4 CPU threads.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchTSTForecaster(nn.Module):
    def __init__(
        self, in_channels: int = 24, out_leads: int = 12, horizon: int = 290, history: int = 700,
        patch_len: int = 20, stride: int = 20, d_model: int = 64, heads: int = 4, layers: int = 2,
        ff: int = 128, dropout: float = 0.2, revin: bool = True,
    ):  # fmt: skip
        super().__init__()
        if in_channels % out_leads:
            raise ValueError("in_channels must be a multiple of out_leads (channels per lead)")
        self.horizon, self.out_leads, self.k = horizon, out_leads, in_channels // out_leads
        self.patch_len, self.stride, self.revin = patch_len, stride, revin
        n_patches = (history + stride - patch_len) // stride + 1  # end padded by `stride`
        self.embed = nn.Linear(patch_len * self.k, d_model)
        self.pos = nn.Parameter(torch.empty(n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Flatten(1), nn.Dropout(dropout), nn.Linear(n_patches * d_model, horizon)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z = x.view(b, t, self.k, self.out_leads).permute(0, 3, 1, 2)  # (b, leads, t, k)
        if self.revin:
            raw = z.sum(-1, keepdim=True)
            mean, sd = raw.mean(2, keepdim=True), raw.std(2, keepdim=True) + 1e-3
            z = torch.cat([z[..., :1] - mean, z[..., 1:]], -1) / sd
        z = torch.cat([z, z[:, :, -1:].expand(-1, -1, self.stride, -1)], 2)  # pad the end
        p = z.unfold(2, self.patch_len, self.stride)  # (b, leads, n, k, patch_len)
        p = p.reshape(b * self.out_leads, p.shape[2], -1)
        h = self.encoder(self.embed(p) + self.pos)
        y = self.head(h).view(b, self.out_leads, self.horizon)
        if self.revin:
            y = y * sd[..., 0] + mean[..., 0]
        return y.transpose(1, 2)
