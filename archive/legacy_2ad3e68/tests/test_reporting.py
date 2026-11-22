"""Reporting must follow the chosen cohort and never reuse unrelated measurements."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compare import collect_runs, comparison_pairs
from config import digest, load_config, protocol, write_json
from metrics import metric_summary
from report import measured_findings, report


def configuration(tmp_path, patients=3):
    source = Path(__file__).parents[1] / "configs/training.yaml"
    settings = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    settings["subset"]["patients"] = {"train": patients, "val": 2, "test": 1}
    settings["storage"] = {
        name: str(tmp_path / name) for name in ("data", "experiments", "outputs", "documents")
    }
    path = tmp_path / f"patients_{patients}.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return load_config(path)


def save_manifest(cfg):
    manifest = {
        "protocol": protocol(cfg),
        "metadata_sha256": "synthetic-source",
        "scp_sha256": "synthetic-statements",
        "preprocessing_seconds": 1.0,
        "splits": {
            split: {
                "patients": count,
                "records": count,
                "windows": count,
                "ids_sha256": digest([split, count]),
                "window_ids_sha256": digest([split, count, "windows"]),
            }
            for split, count in cfg["subset"]["patients"].items()
        },
    }
    manifest["fingerprint"] = digest(manifest)
    write_json(Path(cfg["data"]["cache"]) / "manifest.json", manifest)
    return manifest


def save_evaluation(cfg, manifest):
    run = Path(cfg["paths"]["experiments"]) / "lstm_patient_history"
    write_json(
        run / "config.json",
        {"config": cfg, "architecture": "lstm", "variant": "patient_history", "parameters": 8},
    )
    true = np.ones((2, 12, 4))
    metrics = metric_summary(true + 0.25, true, [1, 2])
    metrics.update(fingerprint=manifest["fingerprint"], qrs_f1=0.0, best_epoch=1)
    write_json(run / "val_metrics.json", metrics)
    pd.DataFrame(
        {"epoch": [1], "seconds": [0.1], "val_patient_macro_mae": [0.25]}
    ).to_csv(run / "history.csv", index=False)
    return run


def test_cohort_changes_paths_but_model_changes_reuse_data(tmp_path):
    small = configuration(tmp_path, 3)
    large = configuration(tmp_path, 6)
    assert small["data"]["cache"] != large["data"]["cache"]
    assert small["paths"]["experiments"] != large["paths"]["experiments"]
    assert small["paths"]["outputs"] != large["paths"]["outputs"]
    copied = copy.deepcopy(small)
    copied["model"]["hidden"] = 128
    assert protocol(copied) == protocol(small)


def test_report_before_preparation_has_no_fabricated_results(tmp_path):
    cfg = configuration(tmp_path)
    destination = report(cfg)
    text = destination.read_text(encoding="utf-8")
    assert "Preprocessing pending" in text
    assert "Completed comparable **val evaluations: 0**" in text
    assert "Training/evaluation pending" in text


def test_report_uses_measured_values_and_current_requested_cohort(tmp_path):
    cfg = configuration(tmp_path)
    save_evaluation(cfg, save_manifest(cfg))
    text = report(cfg).read_text(encoding="utf-8")
    assert "train: 3 patients" in text
    assert "0.250000" in text
    assert "lstm_patient_history" in text
    assert "QRS F1 of zero" in text


def test_report_excludes_old_scores_even_if_output_root_is_reused(tmp_path):
    old = configuration(tmp_path, 3)
    save_evaluation(old, save_manifest(old))
    new = configuration(tmp_path, 6)
    save_manifest(new)
    new["paths"]["experiments"] = old["paths"]["experiments"]
    table, states = collect_runs(new)
    assert table.empty
    assert states.state.str.contains("cohort/window/provenance mismatch").all()
    text = report(new).read_text(encoding="utf-8")
    assert "train: 6 patients" in text and "0.250000" not in text


def test_report_distinguishes_training_from_evaluation(tmp_path):
    cfg = configuration(tmp_path)
    save_manifest(cfg)
    run = Path(cfg["paths"]["experiments"]) / "lstm_patient_history"
    write_json(run / "config.json", {"config": cfg, "architecture": "lstm"})
    write_json(run / "training_metrics.json", {"epochs_completed": 1})
    table, states = collect_runs(cfg)
    assert table.empty and states.iloc[0].state == "evaluation pending"


def test_report_conclusions_follow_scores_instead_of_old_study_names():
    table = pd.DataFrame(
        [
            {"run": "new_model", "model": "lstm", "parameters": 8,
             "patient_macro_mae": 0.1, "qrs_f1": 0.8, "epoch_budget": 2},
            {"run": "new_baseline", "model": "baseline", "parameters": 0,
             "patient_macro_mae": 0.2, "qrs_f1": 0.0, "epoch_budget": 0},
        ]
    )
    findings = " ".join(measured_findings(table, "val"))
    assert "new_model" in findings and "lower than new_baseline" in findings
    assert "QRS F1 of zero" not in findings


def test_matched_comparisons_require_same_hyperparameters_and_stable_direction():
    common = {"variant": "patient_history", "fusion": "late", "control_key": "same"}
    first = {**common, "run": "a", "model": "lstm"}
    second = {**common, "run": "b", "model": "cnn_lstm"}
    different = {**common, "run": "c", "model": "cnn_lstm", "control_key": "different"}
    table = pd.DataFrame([second, different, first])
    assert comparison_pairs(table) == [("a", "b")]


def test_snapshot_keeps_explicit_storage_paths(tmp_path):
    cfg = configuration(tmp_path)
    cfg["paths"]["experiments"] = "recorded-location"
    path = tmp_path / "snapshot.json"
    write_json(path, {"config": cfg, "architecture": "lstm"})
    restored = load_config(path)
    assert restored["paths"]["experiments"] == "recorded-location"
    assert restored["data"]["cache"] == cfg["data"]["cache"]
