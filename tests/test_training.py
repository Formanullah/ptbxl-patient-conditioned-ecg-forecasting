import numpy as np
import pytest
import torch

from ptbxl_forecast.augment import AUGMENTATIONS
from ptbxl_forecast.models import MODELS
from ptbxl_forecast.models.cnn import CNNForecaster
from ptbxl_forecast.preprocessing import Pipeline
from ptbxl_forecast.training import predict, train_forecaster


@pytest.mark.parametrize("name", list(AUGMENTATIONS))
def test_augmentations_keep_shape_and_input_only_ones_spare_target(name):
    g = torch.Generator().manual_seed(0)
    rec = torch.randn(8, 990, 12)
    out = AUGMENTATIONS[name](rec.clone(), 700, g)
    assert out.shape == rec.shape and torch.isfinite(out).all()
    if name in ("gaussian_noise", "masking", "lead_dropout"):
        assert torch.equal(out[:, 700:], rec[:, 700:])


def test_training_smoke_and_raw_output():
    rng = np.random.default_rng(0)
    rec = rng.normal(0, 0.1, (96, 990, 12)).astype(np.float32)
    pipe = Pipeline().fit(rec[:64, :700], ids=range(64))
    model = CNNForecaster(horizon=290, width=8)
    out = train_forecaster(model, pipe, rec[:64], rec[64:80], 700, 290, seed=0, epochs=2, threads=1,
                           augment=AUGMENTATIONS["magnitude"])  # fmt: skip
    assert out["epochs_run"] == 2 and 1 <= out["best_epoch"] <= 2
    assert predict(model, pipe, rec[80:, :700]).shape == (16, 290, 12)


SMALL = {
    "cnn": {"width": 8},
    "lstm": {"hidden": 8},
    "cnn_lstm": {"width": 8, "hidden": 8, "layers": 1},
    "patchtst": {"d_model": 16, "heads": 2, "layers": 1, "ff": 32},
    "transformer": {"d_model": 16, "heads": 2, "enc_layers": 1, "dec_layers": 1, "ff": 32},
}


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("leads8", [False, True])
def test_every_model_maps_decomposed_history_to_raw_forecast(name, leads8):
    rng = np.random.default_rng(0)
    rec = rng.normal(0, 0.1, (8, 990, 12))
    pipe = Pipeline("decompose_0.67", leads8=leads8).fit(rec[:, :700], ids=range(8))
    model = MODELS[name](
        in_channels=pipe.in_channels, out_leads=pipe.out_leads, horizon=290, **SMALL[name]
    )
    assert predict(model, pipe, rec[:, :700]).shape == (8, 290, 12)


@pytest.mark.parametrize("name", list(MODELS))
def test_initial_weights_depend_only_on_seed(name):
    rng = np.random.default_rng(0)
    rec = rng.normal(0, 0.1, (40, 990, 12)).astype(np.float32)
    pipe = Pipeline("decompose_0.67").fit(rec[:32, :700], ids=range(32))
    weights = []
    for torch_state in (1, 2):
        torch.manual_seed(torch_state)  # different RNG state when the model is built
        model = MODELS[name](in_channels=24, out_leads=12, horizon=290, **SMALL[name])
        train_forecaster(model, pipe, rec[:32], rec[32:], 700, 290, seed=0, epochs=1, threads=1)
        weights.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    assert torch.equal(weights[0], weights[1])


def test_default_loss_is_bitwise_l1():
    rng = np.random.default_rng(0)
    rec = rng.normal(0, 0.1, (40, 990, 12)).astype(np.float32)
    pipe = Pipeline("decompose_0.67").fit(rec[:32, :700], ids=range(32))
    weights = []
    for loss in (None, torch.nn.functional.l1_loss):
        model = MODELS["cnn_lstm"](in_channels=24, horizon=290, **SMALL["cnn_lstm"])
        train_forecaster(model, pipe, rec[:32], rec[32:], 700, 290, seed=0, epochs=2, threads=1,
                         loss=loss)  # fmt: skip
        weights.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    assert torch.equal(weights[0], weights[1])


@pytest.mark.parametrize("name", ["derivative", "spectral"])
def test_objectives_are_zero_at_truth_and_penalise_blur(name):
    from ptbxl_forecast.losses import aux_scale, make_objective

    t = torch.arange(290, dtype=torch.float32) / 100
    y = (torch.sin(2 * torch.pi * 1.2 * t) ** 31)[None, :, None].repeat(2, 1, 12)  # spiky beats
    blur = torch.nn.functional.avg_pool1d(y.transpose(1, 2), 9, 1, 4).transpose(1, 2)
    hist = np.zeros((2, 700, 12))
    scale = aux_scale(name, y.numpy(), hist)
    obj, l1 = make_objective(name, 1.0, scale), make_objective("l1")
    assert scale > 0 and obj(y, y).item() == pytest.approx(0, abs=1e-6)
    pred = blur.clone().requires_grad_(True)
    loss = obj(pred, y)
    loss.backward()
    assert loss.item() > l1(blur, y).item() and torch.isfinite(pred.grad).all()


def test_gated_baseline_model_uses_packed_causal_baseline():
    from ptbxl_forecast.models import GatedBaselineCNNLSTM
    from ptbxl_forecast.preprocessing import BaselineConditioned, pack_baseline

    rng = np.random.default_rng(0)
    rec = rng.normal(0, 0.1, (8, 990, 12))
    base = rng.normal(0, 0.1, (8, 290, 12))
    packed = pack_baseline(rec, base, 700)
    pipe = BaselineConditioned(Pipeline("decompose_0.67")).fit(packed[:, :700], ids=range(8))
    model = GatedBaselineCNNLSTM(pipe.in_channels, horizon=290, **SMALL["cnn_lstm"])
    assert pipe.targets(packed[:, 700:], packed[:, :700]).shape == (8, 290, 12)
    with torch.no_grad():  # open gate, no correction -> the baseline itself
        model.gate.weight.zero_(), model.gate.bias.fill_(50.0)
        model.body.head.weight.zero_(), model.body.head.bias.zero_()
    assert np.allclose(predict(model, pipe, packed[:, :700]), base, atol=1e-6)
    with pytest.raises(ValueError):
        BaselineConditioned(Pipeline("decompose_0.67", leads8=True))
