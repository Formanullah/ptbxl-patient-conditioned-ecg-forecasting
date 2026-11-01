"""Synthetic unit tests verify software, not ECG forecast performance."""

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from compare import bootstrap_pair
from config import load_config
from evaluate import metric_summary
from models import Forecaster, augment, nt_xent
from preprocessing import ECGDataset, metadata, preprocess, static_features, validate_cache
from train import atomic_checkpoint, restore_rng, rng_state, seed_everything


def test_actual_wfdb_units_and_order(tmp_path):
    import wfdb

    from config import LEADS
    from preprocessing import read_wave

    cfg = load_config(Path(__file__).parents[1] / "configs/base.yaml")
    cfg["data"]["root"] = str(tmp_path)
    physical = np.tile(np.sin(np.arange(5000)[:, None] / 50), (1, 12))
    wfdb.wrsamp(
        "example",
        fs=500,
        units=["mV"] * 12,
        sig_name=LEADS,
        p_signal=physical,
        fmt=["16"] * 12,
        write_dir=str(tmp_path),
    )
    actual = read_wave({"filename_hr": "example"}, cfg)
    np.testing.assert_allclose(actual, physical, atol=0.0001)
    header = tmp_path / "example.hea"
    original = header.read_text()
    header.write_text(original.replace("/mV", "/V"))
    with pytest.raises(ValueError, match="units"):
        read_wave({"filename_hr": "example"}, cfg)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).parents[1] / "configs/base.yaml")
    cfg["data"].update(
        root=str(tmp_path / "raw"), cache=str(tmp_path / "cache"), history=16, horizon=6
    )
    cfg["paths"].update(
        experiments=str(tmp_path / "experiments"),
        outputs=str(tmp_path / "outputs"),
        documents=str(tmp_path / "docs"),
    )
    cfg["training"].update(epochs=2, patience=5, batch_size=2, threads=1)
    cfg["model"].update(hidden=8, depth=1)
    root = Path(cfg["data"]["root"])
    root.mkdir()
    frame = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4, 5, 6],
            "patient_id": [10, 10, 10, 20, 30, 40],
            "strat_fold": [1, 1, 1, 8, 9, 10],
            "age": [20, 40, 60, 80, 100, 300],
            "sex": [0, 0, 0, 1, 1, np.nan],
            "recording_date": [
                "2000-01-01",
                "2000-01-01",
                "2000-02-01",
                "2001-01-01",
                "2001-01-01",
                "2001-01-01",
            ],
            "scp_codes": [
                "{'MI': 100}",
                "{'STTC': 0}",
                "{'NORM': 100}",
                "{'NORM': 100}",
                "{'MI': 100}",
                "{'NORM': 100}",
            ],
            "filename_hr": [f"record{i}" for i in range(6)],
        }
    )
    frame.to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {
            "code": ["MI", "STTC", "NORM"],
            "diagnostic": [1, 1, 1],
            "diagnostic_class": ["MI", "STTC", "NORM"],
        }
    ).set_index("code").to_csv(root / "scp_statements.csv")

    def wave(row, config):
        return np.arange(5000 * 12, dtype=np.float32).reshape(5000, 12) / 1000 + row.ecg_id

    monkeypatch.setattr("preprocessing.read_wave", wave)
    return cfg, frame


def test_history_and_train_only_age(prepared):
    cfg, _ = prepared
    df = metadata(cfg["data"]["root"])
    static, labels, stats = static_features(df, cfg["data"]["root"])
    assert stats["age_median"] == 50
    assert np.all(static[:2, -1] == 1)  # equal timestamps excluded
    assert np.array_equal(static[2, 4:9], [0, 1, 1, 0, 0])
    assert static[2, -1] == 0
    assert labels.loc[2, "NORM"] == 1 and static[2, 4] == 0
    assert static[-1, 1] == 1 and static[-1, 3] == 1
    df.loc[df.split != "train", "age"] = 5
    _, _, changed_stats = static_features(df, cfg["data"]["root"])
    assert stats == changed_stats


def test_patient_leakage_rejected(prepared):
    cfg, frame = prepared
    frame.loc[4, "patient_id"] = 10
    frame.to_csv(Path(cfg["data"]["root"]) / "ptbxl_database.csv", index=False)
    with pytest.raises(ValueError, match="Patient leakage"):
        metadata(cfg["data"]["root"])


def test_future_isolation_cache_and_inverse(prepared, monkeypatch):
    cfg, _ = prepared
    original = preprocess(cfg)
    cache = Path(cfg["data"]["cache"])
    before_x, before_static = np.load(cache / "X_train.npy"), np.load(cache / "static_train.npy")
    mean, std = np.array(original["mean_mV"]), np.array(original["std_mV"])
    restored = before_x[0] * std[:, None] + mean[:, None]
    expected = np.arange(16 * 12).reshape(16, 12).T / 1000 + 1
    np.testing.assert_allclose(restored, expected, atol=1e-6)

    def corrupt_future(row, config):
        a = np.arange(5000 * 12, dtype=np.float32).reshape(5000, 12) / 1000 + row.ecg_id
        a[16:] += 999
        return a

    monkeypatch.setattr("preprocessing.read_wave", corrupt_future)
    changed = preprocess(cfg)
    np.testing.assert_array_equal(before_x, np.load(cache / "X_train.npy"))
    np.testing.assert_array_equal(before_static, np.load(cache / "static_train.npy"))
    assert changed["mean_mV"] == original["mean_mV"]
    assert changed["std_mV"] == original["std_mV"]
    validate_cache(cfg)
    for v, n in [("ecg_only", 0), ("demographics", 4), ("patient_history", 10)]:
        ds = ECGDataset(cfg, "train", v)
        assert ds[0]["static"].shape == (n,)
    other = copy.deepcopy(cfg)
    other["data"]["horizon"] = 7
    with pytest.raises(ValueError, match="mismatch"):
        validate_cache(other)


@pytest.mark.parametrize("architecture", ["lstm", "cnn_lstm"])
@pytest.mark.parametrize("static_dim", [0, 4, 10])
def test_model_shape_and_gradients(architecture, static_dim):
    torch.set_num_threads(1)
    model = Forecaster(architecture, static_dim, horizon=6, hidden=8, depth=2)
    out = model(torch.randn(2, 12, 16), torch.randn(2, static_dim))
    assert out.shape == (2, 12, 6)
    out.square().mean().backward()
    assert model.rnn.weight_ih_l0.grad.isfinite().all()


def test_contrastive_masks_and_augmentation():
    a = torch.eye(2, requires_grad=True)
    loss, anchors = nt_xent(a, a, torch.tensor([1, 2]), temperature=1)
    assert anchors == 4
    assert loss.item() == pytest.approx(math.log(1 + 2 / math.e))
    loss.backward()
    assert a.grad.isfinite().all()
    zero, anchors = nt_xent(a, a, torch.tensor([1, 1]))
    assert zero.item() == 0 and anchors == 0
    one, anchors = nt_xent(a[:1], a[:1], torch.tensor([1]))
    assert one.item() == 0 and anchors == 0
    x = torch.ones(2, 12, 16)
    original = x.clone()
    augmented = augment(x, {"gain_min": 0.95, "gain_max": 1.05, "jitter_std": 0.01})
    assert torch.equal(x, original) and not torch.equal(augmented, x)


def test_metrics_and_paired_bootstrap():
    true = np.zeros((2, 12, 3))
    pred = np.ones_like(true) * 2
    m = metric_summary(pred, true, np.array([1, 2]))
    assert m["mae"] == 2 and m["mse"] == 4 and m["rmse"] == 2
    assert m["r2"] is None and m["undefined_pearson_leads"] == 12
    a = pd.DataFrame({"ecg_id": [1, 2, 3], "patient_id": [1, 1, 2], "mae_mV": [2, 2, 4]})
    b = a.copy()
    b["mae_mV"] -= 1
    result = bootstrap_pair(a, b, samples=100)
    assert result["delta_patient_macro_mae_mV"] == -1
    assert result["ci95_low"] == result["ci95_high"] == -1


def test_checkpoint_restores_rng(tmp_path):
    import random

    seed_everything(42)
    generator = torch.Generator().manual_seed(42)
    state = rng_state(generator)
    atomic_checkpoint(tmp_path / "last.pt", {"rng": state})
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    restore_rng(torch.load(tmp_path / "last.pt", weights_only=False)["rng"], generator)
    actual = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])


def test_cache_tampering_rejected(prepared):
    cfg, _ = prepared
    preprocess(cfg)
    cache = Path(cfg["data"]["cache"])
    x = np.load(cache / "X_val.npy")
    x[0, 0, 0] += 1
    np.save(cache / "X_val.npy", x)
    with pytest.raises(ValueError, match="Cached file changed"):
        validate_cache(cfg)


def test_evaluate_never_starts_training(prepared):
    from evaluate import evaluate_saved

    cfg, _ = prepared
    preprocess(cfg)
    with pytest.raises(ValueError, match="No selected checkpoint"):
        evaluate_saved(cfg, "lstm", "patient_history")


def test_preprocessing_script_rejects_duplicate_writer(tmp_path, monkeypatch):
    import runpy
    import sys

    from filelock import FileLock, Timeout

    project = Path(__file__).parents[1]
    config_path = project / "configs/base.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["preprocessing.py", "--config", str(config_path)])
    with FileLock(".pipeline.lock"):
        with pytest.raises(Timeout):
            runpy.run_path(str(project / "src/preprocessing.py"), run_name="__main__")


@pytest.mark.parametrize(
    "script", ["preprocessing.py", "train.py", "evaluate.py", "compare.py", "report.py"]
)
def test_script_help(script):
    import subprocess
    import sys

    project = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(project / "src" / script), "--help"],
        cwd=project,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


@pytest.mark.parametrize("flag", ["--all", "--smoke"])
def test_script_rejects_automated_training_modes(flag, monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["train.py", flag])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(__file__).parents[1] / "src/train.py"), run_name="__main__")
    assert exc.value.code == 2


def test_train_script_uses_selected_model(prepared, monkeypatch):
    import runpy
    import sys

    import yaml

    import evaluate

    cfg, _ = prepared
    preprocess(cfg)
    config_path = Path(cfg["data"]["cache"]).parent / "cli.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(evaluate, "learning_curves", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "-model",
            "cnn_lstm",
            "--variant",
            "demographics",
        ],
    )
    runpy.run_path(str(Path(__file__).parents[1] / "src/train.py"), run_name="__main__")
    run = Path(cfg["paths"]["experiments"]) / "cnn_lstm_demographics"
    assert (run / "best.pt").exists() and (run / "training_metrics.json").exists()
    assert not (run / "test_metrics.json").exists()


def test_manual_training_requires_separate_evaluation_and_explicit_resume(prepared, monkeypatch):
    import evaluate as evaluation
    import train as training

    cfg, _ = prepared
    preprocess(cfg)
    original_predict = evaluation.predict

    def no_test_evaluation(*args, **kwargs):
        pytest.fail("Training must not predict or evaluate the test split")

    monkeypatch.setattr(evaluation, "predict", no_test_evaluation)
    monkeypatch.setattr(evaluation, "evaluate_arrays", no_test_evaluation)
    monkeypatch.setattr(training, "learning_curves", lambda *args: None)
    with pytest.raises(ValueError, match="No checkpoint to resume"):
        training.train(cfg, "lstm", "patient_history", resume=True)
    summary = training.train(cfg, "lstm", "patient_history")
    run = Path(cfg["paths"]["experiments"]) / "lstm_patient_history"
    assert (run / "training_metrics.json").exists() and (run / "best.pt").exists()
    assert "best_val_mse" in summary and "mae" not in summary
    assert not (run / "test_metrics.json").exists() and not (run / "predictions.npz").exists()
    with pytest.raises(ValueError, match="Use --resume"):
        training.train(cfg, "lstm", "patient_history")
    resumed = training.train(cfg, "lstm", "patient_history", resume=True)
    assert resumed["epochs_completed"] == summary["epochs_completed"]

    evaluations = 0

    def measure(pred, true, x, metadata, *args):
        nonlocal evaluations
        evaluations += 1
        return metric_summary(pred, true, metadata.patient_id.to_numpy())

    monkeypatch.setattr(evaluation, "predict", original_predict)
    monkeypatch.setattr(evaluation, "evaluate_arrays", measure)
    evaluation.evaluate_saved(cfg, "lstm", "patient_history")
    evaluation.evaluate_saved(cfg, "lstm", "patient_history")
    assert evaluations == 2 and (run / "test_metrics.json").exists()


def test_patient_subset_is_repeatable_and_refits_scalers(prepared):
    cfg, _ = prepared
    preprocess(cfg)
    small = copy.deepcopy(cfg)
    small["subset"] = {
        "source_cache": cfg["data"]["cache"],
        "seed": 42,
        "patients": {"train": 1, "val": 1, "test": 1},
    }
    small["data"]["cache"] += "_small"
    first = preprocess(small)
    second = preprocess(small)
    assert first["fingerprint"] == second["fingerprint"]
    validate_cache(small)
    selected = pd.read_csv(Path(small["data"]["cache"]) / "metadata_train.csv")
    full = pd.read_csv(Path(cfg["data"]["cache"]) / "metadata_train.csv")
    assert selected.patient_id.nunique() == 1
    assert set(selected.ecg_id) == set(
        full.loc[full.patient_id.isin(selected.patient_id), "ecg_id"]
    )
    actual = np.load(Path(small["data"]["cache"]) / "X_train.npy")
    np.testing.assert_allclose(actual.mean(axis=(0, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(actual.std(axis=(0, 2)), 1, atol=1e-5)
    assert first["static_statistics"]["age_median"] == selected.age.median()
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_cache({**small, "subset": {**small["subset"], "seed": 123}})
    expanded = copy.deepcopy(small)
    expanded["data"]["cache"] += "_expanded"
    expanded["subset"]["patients"]["train"] = 2
    larger = preprocess(expanded)
    assert set(first["selection"]["patients"]["train"]) <= set(
        larger["selection"]["patients"]["train"]
    )


def test_interrupted_training_matches_uninterrupted(prepared, monkeypatch):
    import train as training

    cfg, _ = prepared
    preprocess(cfg)

    monkeypatch.setattr(training, "learning_curves", lambda *args: None)
    training.train(cfg, "lstm", "patient_history_contrastive")
    reference = torch.load(
        Path(cfg["paths"]["experiments"]) / "lstm_patient_history_contrastive/last.pt",
        weights_only=False,
    )
    interrupted_cfg = copy.deepcopy(cfg)
    interrupted_cfg["paths"]["experiments"] += "_interrupted"
    real_epoch = training.epoch
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(training, "epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        training.train(interrupted_cfg, "lstm", "patient_history_contrastive")
    monkeypatch.setattr(training, "epoch", real_epoch)
    training.train(interrupted_cfg, "lstm", "patient_history_contrastive", resume=True)
    resumed = torch.load(
        Path(interrupted_cfg["paths"]["experiments"]) / "lstm_patient_history_contrastive/last.pt",
        weights_only=False,
    )
    assert len(resumed["history"]) == 2
    for name in reference["model"]:
        assert torch.equal(reference["model"][name], resumed["model"][name]), name
