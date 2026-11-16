"""Acceptance checks for the primary protocol, isolation, resume, and final lock."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config import LEADS, load_config, read_json, validate
from data import ECGDataset, metadata, prepare, read_wave, select_patients, validate_cache
from evaluate import evaluate_run, final_evaluation, freeze_selection
from feature_selection import mrmr, mrmr_rank
from fixtures import fixture_config
from leakage_audit import audit
from metrics import bootstrap_pair, metric_summary
from models import build_model
from preprocessing import (
    feature_arrays,
    filter_history,
    inverse_scale,
    prepare_transforms,
    prior_diagnoses,
    waveform_scaler,
)
from report import SUMMARY_ORDER, generate
from train import train_run


@pytest.fixture
def prepared(tmp_path):
    cfg = fixture_config(tmp_path)
    cache, manifest = prepare(cfg)
    return cfg, cache, manifest


def test_official_patient_selection_retains_all_records(prepared):
    cfg, _, _ = prepared
    frame = metadata(cfg["data"]["root"])
    cfg["cohorts"]["pilot"]["train"] = 2
    selected = select_patients(frame, cfg)
    assert selected == select_patients(frame, cfg)
    assert len(frame.loc[frame.patient_id.isin(selected["train"])]) == 4
    assert not set(selected["train"]) & set(selected["val"])


def test_patient_fold_leakage_is_rejected(prepared):
    cfg, _, _ = prepared
    path = Path(cfg["data"]["root"]) / "ptbxl_database.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "strat_fold"] = 9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different official folds"):
        metadata(cfg["data"]["root"])


def test_physical_wfdb_shape_units_and_order(prepared):
    cfg, _, _ = prepared
    wave = read_wave(metadata(cfg["data"]["root"]).iloc[0], cfg["data"]["root"])
    assert wave.shape == (1000, 12)
    assert np.isfinite(wave).all()


def test_primary_window_and_matched_scaler_targets(prepared):
    cfg, cache, manifest = prepared
    a = prepare_transforms(cache, cfg, "P0")
    b = prepare_transforms(cache, cfg, "P1")
    assert a == b
    static = np.zeros((manifest["splits"]["train"]["records"], 0), np.float32)
    raw = np.load(cache / "waves_train.npy")
    p0, p1 = [ECGDataset(cache, cfg, "train", p, static) for p in ("P0", "P1")]
    assert p0[0]["x"].shape == (700, 12)
    assert p0[0]["y"].shape == (300, 12)
    np.testing.assert_array_equal(p0[0]["y"], p1[0]["y"])
    np.testing.assert_allclose(inverse_scale(p0[0]["y"], a), raw[0, 700:], atol=1e-7)
    assert manifest["splits"]["train"]["windows"] == len(raw)
    assert not (cache / "waves_test.npy").exists()


def test_raw_training_scaler_ignores_every_future(prepared):
    cfg, cache, _ = prepared
    raw = np.load(cache / "waves_train.npy")
    expected = waveform_scaler(raw, 700)
    changed = raw.copy()
    changed[:, 700:] += 10000
    assert expected == waveform_scaler(changed, 700)
    np.testing.assert_allclose(expected["mean"], raw[:, :700].mean((0, 1), dtype=np.float64))


def test_causal_filter_prefix_and_future_invariance(prepared):
    cfg, cache, _ = prepared
    wave = np.load(cache / "waves_val.npy")[0]
    changed = wave.copy()
    changed[700:] += 10000
    np.testing.assert_array_equal(
        filter_history(wave[:700], cfg), filter_history(changed[:700], cfg)
    )
    changed[350:700] -= 1000
    np.testing.assert_array_equal(
        filter_history(wave[:700], cfg)[:350], filter_history(changed[:700], cfg)[:350]
    )


def test_static_fitting_missingness_and_no_test_features(prepared):
    cfg, cache, _ = prepared
    arrays, names, info = feature_arrays(cfg, cache, "F1")
    assert names == ["age", "age_missing", "sex", "sex_missing"]
    assert set(arrays) == {"train", "val"}
    train = pd.read_csv(cache / "metadata_train.csv")
    valid = train.age.where(train.age.between(0, 120)).dropna()
    assert info["statistics"]["age"]["median"] == valid.median()
    assert arrays["train"][:, 1].sum() == 2


def test_equal_time_and_current_labels_are_not_history(prepared):
    cfg, cache, _ = prepared
    frame = pd.read_csv(cache / "metadata_train.csv").iloc[:2].copy().reset_index(drop=True)
    frame.recording_date = "1990-01-01 12:00:00"
    prior, result = prior_diagnoses(frame, cfg["data"]["root"])
    assert result["strict_prior_edges"] == 0
    np.testing.assert_array_equal(prior[:, -1], 1)
    frame.loc[1, "recording_date"] = "1990-01-02 12:00:00"
    prior, result = prior_diagnoses(frame, cfg["data"]["root"])
    assert prior[0, 0] == 0 and prior[1, 0] == 1
    frame.loc[1, "scp_codes"] = "{}"
    changed, _ = prior_diagnoses(frame, cfg["data"]["root"])
    np.testing.assert_array_equal(prior, changed)


@pytest.mark.parametrize("architecture", ["lstm", "cnn_lstm"])
@pytest.mark.parametrize("static_dim", [0, 4])
def test_model_shapes_early_conditioning_and_active_gradients(architecture, static_dim):
    torch.set_num_threads(1)
    model = build_model(architecture, static_dim, hidden=8, head_width=16, static_embedding=4)
    output = model(torch.randn(2, 700, 12), torch.randn(2, static_dim))
    assert output.shape == (2, 300, 12)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any("projector" in name for name, _ in model.named_parameters())
    expected = 12 + (4 if static_dim else 0)
    assert (
        model.lstm.input_size if architecture == "lstm" else model.convolutions[0].in_channels
    ) == expected


def test_alternative_geometry_is_separate(prepared):
    cfg, cache, manifest = prepared
    cfg = copy.deepcopy(cfg)
    cfg["study_id"] = "five_to_five_ablation"
    cfg["data"].update(history=500, horizon=500)
    other, second = prepare(cfg)
    assert other != cache and second["cohort"] != manifest["cohort"]
    cfg["data"]["horizon"] = 300
    with pytest.raises(ValueError):
        validate(cfg)


def test_cache_tampering_is_rejected(prepared):
    cfg, cache, _ = prepared
    path = cache / "waves_train.npy"
    values = np.load(path, mmap_mode="r+")
    values[0, 0, 0] += 1
    values.flush()
    del values
    with pytest.raises(ValueError, match="Cache file changed"):
        validate_cache(cache, cfg)


def test_scaler_tampering_is_rejected(prepared):
    cfg, cache, _ = prepared
    path = cache / "scaler.json"
    info = read_json(path)
    info["mean"][0] += 1
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="scaler changed"):
        prepare_transforms(cache, cfg, "P0")


def test_executable_leakage_audit(prepared):
    cfg, cache, _ = prepared
    assert audit(cfg, cache)["passed"]


def test_hand_calculated_errors_patient_weighting_and_constant_pearson():
    target = np.zeros((3, 2, 12))
    prediction = np.ones_like(target)
    prediction[2] = 3
    result, leads, _ = metric_summary(prediction, target, [1, 1, 2])
    assert result["mae_mV"] == pytest.approx(5 / 3)
    assert result["mse_mV2"] == pytest.approx(11 / 3)
    assert result["rmse_mV"] == pytest.approx(np.sqrt(11 / 3))
    assert result["patient_macro_mae_mV"] == 2
    assert result["pearson_mean"] is None and result["r2"] is None
    assert leads.lead.tolist() == LEADS


def test_patient_bootstrap_requires_identical_records():
    first = pd.DataFrame({"ecg_id": [1, 2, 3], "patient_id": [1, 1, 2], "mae_mV": [2.0, 2.0, 4.0]})
    second = first.copy()
    second.mae_mV -= 1
    result = bootstrap_pair(first, second, samples=50)
    assert result["difference_B_minus_A_mV"] == -1
    assert result["conclusion"] == "helped"
    with pytest.raises(ValueError):
        bootstrap_pair(first, second.iloc[:2])


def test_mrmr_never_opens_heldout_arrays(prepared, monkeypatch):
    cfg, cache, _ = prepared
    original = np.load

    def guarded(path, *args, **kwargs):
        if "waves_val" in str(path) or "waves_test" in str(path):
            raise AssertionError("mRMR opened held-out outcomes")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    artifact = mrmr(cfg, cache)
    assert artifact["used_validation_outcomes"] is False and artifact["used_test"] is False
    assert len(artifact["selected_groups"]) == 3
    for group in artifact["selected_groups"]:
        assert set(artifact["groups"][group]) <= set(artifact["selected_features"])


def test_mrmr_ranks_training_relevance():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=100)
    features = np.column_stack([signal, np.zeros(100), rng.normal(size=100), np.zeros(100)])
    names = ["age", "age_missing", "sex", "sex_missing"]
    groups = {"age": names[:2], "sex": names[2:]}
    ranking = mrmr_rank(features, signal[:, None], names, groups)
    assert ranking.iloc[0].group == "age"


def test_interrupted_resume_matches_uninterrupted(prepared):
    cfg, cache, _ = prepared
    first = train_run(cfg, cache, "lstm", name="reference")
    second = train_run(cfg, cache, "lstm", name="interrupted", stop_after=1)
    train_run(cfg, cache, "lstm", name="interrupted", resume=True)
    a, b = [torch.load(p / "last.pt", weights_only=False) for p in (first, second)]
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])
    assert len(b["history"]) == 2
    with pytest.raises(ValueError, match="Existing run"):
        train_run(cfg, cache, "lstm", name="reference")
    with pytest.raises(ValueError, match="configuration/source changed"):
        train_run(cfg, cache, "lstm", name="reference", lr=0.001, resume=True)


def test_fold10_denied_before_loading(prepared):
    cfg, cache, _ = prepared
    with pytest.raises(ValueError, match="Fold 10"):
        ECGDataset(cache, cfg, "test", "P0", np.zeros((4, 0)))
    with pytest.raises(ValueError, match="Fold 10"):
        final_evaluation(cfg, cache)
    with pytest.raises(ValueError, match="Fold 10"):
        prepare(
            cfg, include_test=True, lock={"cohort": read_json(cache / "manifest.json")["cohort"]}
        )


def test_four_condition_final_lock_and_idempotent_ledger(prepared, monkeypatch):
    cfg, _, _ = prepared
    cfg["run_mode"] = "full"
    cfg["training"]["full_epochs"] = 1
    cache, _ = prepare(cfg)
    runs = []
    for model in ("lstm", "cnn_lstm"):
        for filtering in ("P0", "P1"):
            run = train_run(cfg, cache, model, filtering)
            evaluate_run(cfg, cache, run)
            runs.append(run)
    lock = freeze_selection(cfg, cache, runs)
    assert len(lock["runs"]) == 4 and lock["used_test_for_selection"] is False
    cfg["evaluation"]["final_test"] = True
    ledger = final_evaluation(cfg, cache)
    assert len(ledger["conditions"]) == 6
    assert all(r["state"] == "completed" for r in ledger["conditions"].values())
    import evaluate

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed final evaluation ran twice")

    monkeypatch.setattr(evaluate, "evaluate_run", forbidden)
    monkeypatch.setattr(evaluate, "baseline", forbidden)
    assert final_evaluation(cfg, cache) == ledger
    with (runs[0] / "best.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="frozen checkpoint"):
        final_evaluation(cfg, cache)


def test_reports_do_not_promote_smoke_metrics(prepared, monkeypatch):
    cfg, cache, _ = prepared
    run = train_run(cfg, cache, "lstm", epochs=1)
    evaluate_run(cfg, cache, run)
    # Keep this test's generated documentation in its isolated result folder.
    monkeypatch.chdir(cache)
    path = generate(cfg)
    summary = read_json(path.parent / "summary.json")
    assert summary["order"] == SUMMARY_ORDER
    assert "SYNTHETIC SOFTWARE FIXTURE" in summary["warning"]
    table = pd.read_csv(path.parent / "experiment_results.csv")
    assert table.mae_mV.isna().all()


def test_legacy_config_rejected():
    with pytest.raises(ValueError, match="Legacy configuration"):
        load_config("archive/legacy_2ad3e68/configs/pilot.yaml")


def test_notebook_schema_plain_python_and_shared_code():
    import ast

    import nbformat

    path = Path("notebooks/PTBXL_Full_Colab_GPU.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert "from models.lstm import LSTMForecaster" in source
    assert "class LSTM" not in source and "class CNN" not in source
    assert "RUN_MODE = 'pilot'" in source and "FINAL_TEST = False" in source
    assert "records500" not in source
