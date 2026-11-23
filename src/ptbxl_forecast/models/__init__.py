"""Forecasting models. All map (batch, 700, in_channels) -> (batch, horizon, out_leads)."""

from .cnn import CNNForecaster
from .cnn_lstm import CNNLSTMForecaster
from .lstm import LSTMForecaster
from .patchtst import PatchTSTForecaster
from .transformer import TransformerForecaster

MODELS = {
    "cnn": CNNForecaster,
    "lstm": LSTMForecaster,
    "cnn_lstm": CNNLSTMForecaster,
    "patchtst": PatchTSTForecaster,
    "transformer": TransformerForecaster,
}
