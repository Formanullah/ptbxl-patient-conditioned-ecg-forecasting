import torch
from torch import nn


class PatchTST32(nn.Module):

    def __init__(
        self,
        static_dim=0,
        horizon=300,
        hidden=64,
        depth=1,
        dropout=0.2,
        head_width=128,
        patch_len=32,
        stride=8,
        **kwargs,
    ):
        super().__init__()

        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride

        self.patch_embed = nn.Linear(12 * patch_len, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(depth, 1),
        )

        n_patches = (700 - patch_len) // stride + 1

        self.head = nn.Sequential(
            nn.Linear(hidden * n_patches, head_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, horizon * 12),
        )

    def forward(self, history, static=None):

        if history.ndim != 3 or history.shape[-1] != 12:
            raise ValueError("Expected history [batch,time,12]")

        patches = history.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )

        patches = patches.contiguous()

        B, N, P, C = patches.shape

        patches = patches.reshape(B, N, P * C)

        tokens = self.patch_embed(patches)

        encoded = self.encoder(tokens)

        features = encoded.reshape(B, -1)

        forecast = self.head(features)

        return forecast.reshape(B, self.horizon, 12)
