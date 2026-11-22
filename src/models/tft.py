import torch
from torch import nn


class GatedResidualNetwork(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        residual = x

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        gated = x * self.gate(x)

        return self.norm(gated + residual)


class TFTForecaster(nn.Module):

    def __init__(
        self,
        static_dim=0,
        horizon=300,
        hidden=64,
        depth=1,
        dropout=0.2,
        head_width=128,
        **kwargs
    ):
        super().__init__()

        self.horizon = horizon

        self.input_projection = nn.Linear(12, hidden)

        self.grn_stack = nn.ModuleList(
            [
                GatedResidualNetwork(hidden, dropout)
                for _ in range(max(depth, 1))
            ]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dropout=dropout,
            batch_first=True,
        )

        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(depth, 1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(hidden, head_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, horizon * 12),
        )

    def forward(self, history, static=None):

        if history.ndim != 3 or history.shape[-1] != 12:
            raise ValueError("Expected history [batch,time,12]")

        x = self.input_projection(history)

        for block in self.grn_stack:
            x = block(x)

        x = self.attention(x)

        x = x.transpose(1, 2)

        x = self.pool(x).squeeze(-1)

        output = self.head(x)

        return output.reshape(-1, self.horizon, 12)
