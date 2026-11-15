"""Two independent forecasting architectures; no contrastive projection head."""

from .cnn_lstm import CNNLSTMForecaster
from .lstm import LSTMForecaster


def build_model(architecture, static_dim=0, horizon=300, **settings):
    if architecture == "lstm":
        return LSTMForecaster(static_dim=static_dim, horizon=horizon, **settings)
    if architecture == "cnn_lstm":
        return CNNLSTMForecaster(static_dim=static_dim, horizon=horizon, **settings)
    raise ValueError("architecture must be lstm or cnn_lstm")
