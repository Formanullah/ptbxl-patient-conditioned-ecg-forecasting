"""Synthetic unit tests verify software, not ECG forecast performance."""

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config import load_config
from metrics import bootstrap_pair, metric_summary
from models import Forecaster, augment, nt_xent
from preprocessing import ECGDataset, metadata, preprocess, static_features, validate_cache
from train import atomic_checkpoint, restore_rng, rng_state, seed_everything


def test_actual_wfdb_units_and_order(tmp_path):
    import wfdb

    from config import LEADS
    from preprocessing import read_wave

    cfg = load_config(Path(__file__).parents[1] / "configs/training.yaml")
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
    cfg["data"]["sampling_rate"] = 500
    actual = read_wave({"filename_hr": "example"}, cfg)
    np.testing.assert_allclose(actual, physical, atol=0.0001)
    header = tmp_path / "example.hea"
    original = header.read_text()
    header.write_text(original.replace("/mV", "/V"))
    with pytest.raises(ValueError, match="units"):
        read_wave({"filename_hr": "example"}, cfg)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).parents[1] / "configs/training.yaml")
    cfg["data"].update(
        root=str(tmp_path / "raw"), cache=str(tmp_path / "cache"), history=16, horizon=6,
        sampling_rate=500, history_seconds=0.032, horizon_seconds=0.012,
        window_stride_seconds=0.002, stride=1, layout="channels_first", training_windows="single"
    )
    cfg["subset"] = {"mode": "full", "seed": 42}
    cfg["preprocessing"]["scaling"] = "zscore"
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
    cfg["data"]["cache"] += "_future"
    cache = Path(cfg["data"]["cache"])
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
    from evaluate import evaluate_run

    cfg, _ = prepared
    preprocess(cfg)
    with pytest.raises(ValueError, match="No selected checkpoint"):
        evaluate_run(cfg, "lstm", "patient_history")


def test_preprocessing_script_rejects_duplicate_writer(tmp_path, monkeypatch):
    import runpy
    import sys

    from filelock import FileLock, Timeout

    project = Path(__file__).parents[1]
    config_path = project / "configs/training.yaml"
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
    # The CLI's writer lock belongs to this synthetic workspace, not a real study.
    monkeypatch.chdir(config_path.parent)
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
    monkeypatch.setattr(evaluation, "arrays_evaluation", no_test_evaluation)
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

    def measure(cfg, run, split, pred, true, x, metadata, manifest):
        nonlocal evaluations
        evaluations += 1
        return metric_summary(pred, true, metadata.patient_id.to_numpy())

    monkeypatch.setattr(evaluation, "predict", original_predict)
    monkeypatch.setattr(evaluation, "arrays_evaluation", measure)
    evaluation.evaluate_run(cfg, "lstm", "patient_history", "val")
    evaluation.evaluate_run(cfg, "lstm", "patient_history", "val")
    assert evaluations == 2 and (run / "val_metrics.json").exists()


def test_patient_subset_is_repeatable_and_refits_scalers(prepared):
    cfg, _ = prepared
    preprocess(cfg)
    small = copy.deepcopy(cfg)
    small["subset"] = {
        "mode": "custom",
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
    assert set(first["selected_patients"]["train"]) <= set(
        larger["selected_patients"]["train"]
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


@pytest.fixture
def research_prepared(prepared, monkeypatch):
    from preprocessing import preprocess

    old, _ = prepared
    cfg = load_config(Path(__file__).parents[1] / "configs/training.yaml")
    cfg["data"].update(root=old["data"]["root"], cache=old["data"]["cache"] + "_research")
    cfg["paths"] = old["paths"]
    cfg["subset"] = {"mode": "custom", "seed": 42, "patients": {"train": 2, "val": 1, "test": 1}}
    cfg["training"].update(epochs=2, batch_size=4, threads=1)
    cfg["preprocessing"]["scaling"] = "zscore"
    cfg["model"].update(hidden=8, depth=1, static_embedding=4)
    monkeypatch.setattr("preprocessing.observed_features", lambda x, fs: ([np.nan] * 3, False))
    preprocess(cfg)
    return cfg


@pytest.mark.parametrize(
    "history,horizon,windows", [(3, 1, 7), (5, 1, 5), (5, 3, 3), (7, 3, 1), (8, 2, 1)]
)
def test_seconds_protocol_geometry(history, horizon, windows, tmp_path):
    import yaml

    from preprocessing import window_starts

    cfg = load_config(Path(__file__).parents[1] / "configs/training.yaml")
    cfg["data"].update(history_seconds=history, horizon_seconds=horizon)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    actual = load_config(path)
    assert actual["data"]["history"] == history * 100
    assert actual["data"]["horizon"] == horizon * 100
    assert len(window_starts(actual, "train")) == windows
    assert window_starts(actual, "val") == window_starts(actual, "test") == [0]
    cfg["data"]["horizon_seconds"] = 11
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="exceeds"):
        load_config(path)


def test_research_patient_record_window_isolation_and_layout(research_prepared):
    cfg = research_prepared
    manifest = validate_cache(cfg)
    assert manifest["protocol"]["schema"] == 2
    assert manifest["splits"]["train"] == {
        **manifest["splits"]["train"],
        "windows": 12,
        "records": 4,
        "patients": 2,
    }
    ds = ECGDataset(cfg, "train", "expanded")
    assert ds[0]["x"].shape == (500, 12)
    assert ds[0]["y"].shape == (300, 12)
    assert ds[0]["static"].shape == (22,)
    assert ds.metadata.window_id.is_unique
    assert len(ECGDataset(cfg, "val")) == 1
    assert len(ECGDataset(cfg, "test")) == 1


@pytest.mark.parametrize("kind", ["none", "highpass", "bandpass"])
def test_causal_filter_future_mutation(kind):
    from preprocessing import input_filter, observed_features

    cfg = load_config(Path(__file__).parents[1] / "configs/training.yaml")
    cfg["preprocessing"]["filter"] = kind
    rng = np.random.default_rng(42)
    wave = rng.normal(size=(12, 1000)).astype(np.float32)
    changed = wave.copy()
    changed[:, 500:] += 10000
    a, b = [input_filter(v[:, :500], cfg) for v in (wave, changed)]
    np.testing.assert_array_equal(a, b)
    assert observed_features(wave[:, :500], 100) == observed_features(changed[:, :500], 100)
    torch.manual_seed(42)
    view_a = augment(torch.tensor(a[None]), cfg["contrastive"])
    torch.manual_seed(42)
    view_b = augment(torch.tensor(b[None]), cfg["contrastive"])
    assert torch.equal(view_a, view_b)


def test_research_train_only_scaling_and_raw_targets(research_prepared, monkeypatch):
    from preprocessing import preprocess
    from metrics import physical_arrays

    cfg = research_prepared
    old = validate_cache(cfg)
    cache = Path(cfg["data"]["cache"])
    x_train = np.load(cache / "X_train.npy")
    df = metadata(cfg["data"]["root"])
    original_read = __import__("preprocessing").read_wave

    def changed(row, settings):
        wave = original_read(row, settings).copy()
        if row.split != "train":
            wave[500:] += 1000
        return wave

    monkeypatch.setattr("preprocessing.read_wave", changed)
    cfg = copy.deepcopy(cfg)
    cfg["data"]["cache"] += "_changed"
    new = preprocess(cfg)
    assert old["mean_mV"] == new["mean_mV"] and old["std_mV"] == new["std_mV"]
    np.testing.assert_array_equal(x_train, np.load(Path(cfg["data"]["cache"]) / "X_train.npy"))
    np.testing.assert_array_equal(
        np.load(cache / "X_val.npy"), np.load(Path(cfg["data"]["cache"]) / "X_val.npy")
    )
    first_val = df.loc[df.split.eq("val")].iloc[0]
    raw = changed(first_val, cfg)[500:800].T
    recovered = physical_arrays(np.load(Path(cfg["data"]["cache"]) / "y_val.npy"), new)[0]
    np.testing.assert_allclose(recovered, raw, atol=1e-4)


@pytest.mark.parametrize("method", ["zscore", "robust"])
def test_affine_scaling_round_trip(method):
    from preprocessing import fit_scaling

    x = np.random.default_rng(1).normal(size=(5, 12, 500)).astype(np.float32)
    center, scale = fit_scaling(x, method)
    normalized = (x - center[None, :, None]) / scale[None, :, None]
    np.testing.assert_allclose(
        normalized * scale[None, :, None] + center[None, :, None], x, atol=1e-6
    )


@pytest.mark.parametrize("architecture", ["lstm", "cnn_lstm"])
@pytest.mark.parametrize("fusion", ["early", "late"])
def test_time_first_fusion_gradients_and_static_conditioning(architecture, fusion):
    torch.set_num_threads(1)
    model = Forecaster(
        architecture,
        10,
        horizon=30,
        hidden=8,
        depth=2,
        fusion=fusion,
        input_layout="time_first",
        pool=1,
    ).eval()
    x = torch.randn(2, 50, 12)
    static = torch.randn(2, 10)
    out = model(x, static)
    assert out.shape == (2, 30, 12)
    assert not torch.equal(out, model(x, static + 5))
    out.square().mean().backward()
    assert model.static_net[0].weight.grad.isfinite().all()
    assert model.static_net[0].weight.grad.abs().sum() > 0
    assert model.rnn.weight_ih_l0.grad.isfinite().all()


def test_peak_matching_no_peaks_and_one_to_one():
    from morphology import peak_counts

    counts, pairs = peak_counts([100, 110], [105], 10)
    assert counts["tp"] == 1 and counts["fn"] == 1 and counts["fp"] == 0
    assert len(pairs) == 1
    empty, _ = peak_counts([], [], 10)
    assert empty["f1"] is None and empty["precision"] is None and empty["recall"] is None
    false, _ = peak_counts([], [10], 10)
    assert false["fp"] == 1 and false["f1"] == 0 and false["precision"] == 0
    exact, _ = peak_counts([10, 100], [12, 97], 5)
    assert exact["f1"] == exact["precision"] == exact["recall"] == 1


def test_mrmr_relevance_and_missingness_groups():
    from feature_selection import feature_groups, mrmr_ranking, mutual_information

    rng = np.random.default_rng(3)
    relevant = rng.integers(0, 2, 1000)
    irrelevant = rng.integers(0, 2, 1000)
    x = np.column_stack([relevant, np.zeros(1000), irrelevant, np.zeros(1000)])
    names = ["age_scaled", "age_missing", "sex_encoded", "sex_missing"]
    ranking, groups = mrmr_ranking(x, relevant[:, None], names)
    assert ranking.iloc[0].group == "age"
    assert groups == feature_groups(names)
    assert mutual_information(relevant, relevant) > mutual_information(relevant, irrelevant)
    assert groups["age"] == ["age_scaled", "age_missing"]


def test_fold10_guard_and_selector_provenance(research_prepared):
    from config import write_json
    from evaluate import authorize_split

    cfg = research_prepared
    with pytest.raises(ValueError, match="reserved"):
        authorize_split(cfg, "test")
    artifact = Path(cfg["data"]["cache"]) / "mrmr.json"
    write_json(
        artifact, {"method": "mrmr", "fingerprint": "wrong", "selected_features": ["age_scaled"]}
    )
    cfg["features"] = {"mrmr_path": str(artifact)}
    with pytest.raises(ValueError, match="provenance"):
        ECGDataset(cfg, "train", "mrmr")


def test_selection_uses_patient_macro_mae_and_full_epoch_budget(research_prepared, monkeypatch):
    import train as training

    cfg = research_prepared
    cfg["training"].update(epochs=2, patience=1, early_stopping=False)
    calls = 0

    def scores(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Validation MSE improves but primary physical-unit MAE worsens in epoch two.
        second = calls > 2
        return {
            "mse": 0.1 if second else 0.2,
            "contrastive_loss": 0.0,
            "valid_anchors": 0,
            "skipped_anchors": 0,
            "mae": 0.4 if second else 0.3,
            "patient_macro_mae": 0.4 if second else 0.3,
            "mse_mV2": 0.2,
            "rmse": 0.5,
        }

    monkeypatch.setattr(training, "epoch", scores)
    monkeypatch.setattr(training, "learning_curves", lambda *args: None)
    training.train(cfg, "lstm", "patient_history")
    run = Path(cfg["paths"]["experiments"]) / "lstm_patient_history"
    best = torch.load(run / "best.pt", weights_only=True)
    last = torch.load(run / "last.pt", weights_only=False)
    assert best["epoch"] == 1 and best["selection_metric"] == "patient_macro_mae_mV"
    assert len(last["history"]) == 2


def test_streamed_cache_matches_in_memory(research_prepared):
    from preprocessing import preprocess

    cfg = research_prepared
    old = Path(cfg["data"]["cache"])
    streamed = copy.deepcopy(cfg)
    streamed["data"].update(cache=str(old) + "_streamed", memmap_threshold=1)
    preprocess(streamed)
    new = Path(streamed["data"]["cache"])
    validate_cache(streamed)
    for source in old.glob("*.npy"):
        np.testing.assert_array_equal(np.load(source), np.load(new / source.name))
    assert not list(new.glob("_work_*.npy"))


def test_named_recipe_overrides_inherited_cache(tmp_path):
    import yaml

    base = load_config(Path(__file__).parents[1] / "configs/training.yaml")
    base["data"]["cache"] = "data/reference"
    base["experiments"] = {"short": {"data": {"cache": "data/short", "history_seconds": 3.0}}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    child = {"extends": "base.yaml", "data": {"cache": "data/shared"}}
    (tmp_path / "child.yaml").write_text(yaml.safe_dump(child))
    cfg = load_config(tmp_path / "child.yaml", "short")
    assert cfg["data"]["cache"] == "data/short" and cfg["data"]["history"] == 300


def test_context_controls_have_identical_future_intervals():
    from preprocessing import window_starts

    cfg = load_config(Path(__file__).parents[1] / "configs/training.yaml")
    for short, long, horizon, origin in ((3, 5, 1, 5), (5, 7, 3, 7)):
        controls = []
        for history in (short, long):
            d = {
                **cfg["data"], "history": history * 100, "horizon": horizon * 100,
                "start": (origin - history) * 100, "training_windows": "single",
            }
            assert window_starts({**cfg, "data": d}, "val") == [d["start"]]
            controls.append((d["start"] + d["history"], d["horizon"]))
        assert controls[0] == controls[1]



def test_qrs_timing_amplitude_and_empty_windows(monkeypatch):
    import morphology

    observed = np.zeros((1, 12, 100), dtype=np.float32)
    true, pred = np.zeros((1, 12, 200)), np.zeros((1, 12, 200))
    true[0, 1, [50, 150]] = [1.0, 2.0]
    pred[0, 1, 52] = 0.8
    responses = iter([(np.array([150, 250]), False), (np.array([152]), False)])
    monkeypatch.setattr(morphology, "detect_peaks", lambda *args: next(responses))
    summary, _, _, _ = morphology.qrs_metrics(pred, true, observed, 100)
    assert summary["qrs_precision"] == 1 and summary["qrs_recall"] == 0.5
    assert summary["qrs_f1"] == pytest.approx(2 / 3)
    assert summary["peak_timing_mae_ms"] == 20
    assert summary["peak_amplitude_mae_mV"] == pytest.approx(0.2)
    monkeypatch.setattr(morphology, "detect_peaks", lambda *args: (np.array([], dtype=int), False))
    empty, _, _, _ = morphology.qrs_metrics(pred * 0, true * 0, observed, 100)
    assert empty["qrs_f1"] is None and empty["peak_timing_mae_ms"] is None


def test_mrmr_reads_only_training_arrays(research_prepared, monkeypatch):
    from feature_selection import mrmr

    original = np.load

    def training_only(path, *args, **kwargs):
        assert "_val.npy" not in str(path) and "_test.npy" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", training_only)
    selected = mrmr(research_prepared, 2)
    assert selected["fitted_split"] == "train" and selected["used_test"] is False
    assert selected["used_validation"] is False
