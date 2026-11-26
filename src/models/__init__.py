"""Forecasting architectures for PTB-XL ECG forecasting."""

from .cnn_lstm import CNNLSTMForecaster
from .lstm import LSTMForecaster
from .tft import TFTForecaster
from .patchtst import PatchTSTForecaster
from .PatchTST_32 import PatchTST32



def build_model(architecture, static_dim=0, horizon=300, **settings):
    if architecture == "lstm":
        return LSTMForecaster(static_dim=static_dim, horizon=horizon, **settings)
    if architecture == "cnn_lstm":
        return CNNLSTMForecaster(static_dim=static_dim, horizon=horizon, **settings)
    if architecture == "tft":
        return TFTForecaster(static_dim=static_dim, horizon=horizon, **settings)
    if architecture == "patchtst":
        return PatchTSTForecaster(static_dim=static_dim, horizon=horizon, **settings)
    elif architecture == "patchtst_32":
        return PatchTST32(
            static_dim=static_dim,
            horizon=horizon,
            **settings,
        )

    raise ValueError(f"Unsupported architecture: {architecture}. Choose from lstm, cnn_lstm, tft, patchtst.")
