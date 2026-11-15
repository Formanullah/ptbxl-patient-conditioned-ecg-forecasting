"""Explicit inference, raw-history baselines, and a frozen final-test ledger."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    SCHEMA,
    device_for,
    digest,
    file_hash,
    protocol,
    read_json,
    source_hashes,
    study_dir,
    write_json,
)
from data import ECGDataset, loader, prepare
from metrics import metric_summary
from models import build_model
from preprocessing import inverse_scale, prepare_transforms
from train import static_inputs


def outputs(cfg, name):
    path = study_dir(cfg) / cfg["run_mode"] / "metrics" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def predict(model, batches, device):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in batches:
            predictions.append(
                model(batch["x"].to(device), batch["static"].to(device)).cpu().numpy()
            )
            targets.append(batch["y"].numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def verify_lock(cfg, cache):
    path = study_dir(cfg) / "full" / "selection.json"
    if cfg["run_mode"] != "full" or not cfg["evaluation"]["final_test"] or not path.exists():
        raise ValueError(
            "Fold 10 is locked: full mode, explicit final_test and a frozen selection are required"
        )
    lock = read_json(path)
    manifest = read_json(Path(cache) / "manifest.json")
    if (
        lock["schema"] != SCHEMA
        or lock["cohort"] != manifest["cohort"]
        or lock["protocol"] != protocol(cfg)
        or lock["manifest_sha256"] != file_hash(Path(cache) / "manifest.json")
    ):
        raise ValueError("Frozen protocol/cohort mismatch")
    if lock["scaler_sha256"] != file_hash(Path(cache) / "scaler.json"):
        raise ValueError("Frozen scaler changed")
    if lock["filter"] != cfg["preprocessing"]:
        raise ValueError("Frozen filter settings changed")
    for name, item in lock["runs"].items():
        run = study_dir(cfg) / "full" / "checkpoints" / name
        if item["checkpoint_sha256"] != file_hash(run / "best.pt") or item[
            "run_sha256"
        ] != file_hash(run / "run.json"):
            raise ValueError("A frozen checkpoint or configuration changed")
        if item["validation_sha256"] != file_hash(outputs(cfg, name) / "val.json"):
            raise ValueError("Frozen validation evidence changed")
        if item["validation_receipt_sha256"] != file_hash(outputs(cfg, name) / "val_receipt.json"):
            raise ValueError("Frozen validation receipt changed")
        if read_json(run / "run.json")["scientific"]["source"] != source_hashes():
            raise ValueError("Source differs from the frozen scientific implementation")
        verify_evaluation(outputs(cfg, name), "val")
    return lock


def save_evaluation(cfg, cache, name, split, prediction, target, details):
    out = outputs(cfg, name)
    metadata = pd.read_csv(Path(cache) / f"metadata_{split}.csv")
    # Choose the displayed example before scoring; the other example uses median error.
    specified = cfg["evaluation"].get("example_ecg_id")
    if specified is None:
        specified = int(metadata.ecg_id.min())
    if specified not in set(metadata.ecg_id):
        raise ValueError(f"Pre-specified ECG {specified} is not in the evaluated cohort")
    summary, leads, records = metric_summary(prediction, target, metadata.patient_id.to_numpy())
    records.insert(0, "ecg_id", metadata.ecg_id.to_numpy())
    manifest = read_json(Path(cache) / "manifest.json")
    summary.update(
        name=name,
        split=split,
        cohort=manifest["cohort"],
        protocol=protocol(cfg),
        synthetic=manifest["synthetic"],
        smoke=cfg["run_mode"] == "pilot" or manifest["synthetic"],
        **details,
    )
    summary["target_sha256"] = hashlib.sha256(
        np.ascontiguousarray(target, dtype=np.float32).tobytes()
    ).hexdigest()
    records.to_csv(out / f"{split}_records.csv", index=False)
    leads.to_csv(out / f"{split}_leads.csv", index=False)
    np.savez_compressed(
        out / f"{split}_predictions.npz",
        prediction=prediction,
        target=target,
        ecg_id=metadata.ecg_id.to_numpy(),
        patient_id=metadata.patient_id.to_numpy(),
    )
    summary["pre_specified_ecg_id"] = int(specified)
    order = records.sort_values(["mae_mV", "ecg_id"], kind="stable")
    summary["median_error_ecg_id"] = int(order.iloc[len(order) // 2].ecg_id)
    write_json(out / f"{split}.json", summary)
    files = {
        p.name: file_hash(p)
        for p in (
            out / f"{split}.json",
            out / f"{split}_records.csv",
            out / f"{split}_leads.csv",
            out / f"{split}_predictions.npz",
        )
    }
    write_json(out / f"{split}_receipt.json", {"files": files, "cohort": manifest["cohort"]})
    return summary


def verify_evaluation(out, split):
    out = Path(out)
    receipt = read_json(out / f"{split}_receipt.json")
    for name, expected in receipt["files"].items():
        if not (out / name).exists() or file_hash(out / name) != expected:
            raise ValueError(f"Saved evaluation artifact changed: {name}")
    return receipt


def evaluate_run(cfg, cache, run, split="val", checkpoint="best.pt", label=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    run = Path(run)
    record = read_json(run / "run.json")
    if record.get("schema") != SCHEMA:
        raise ValueError("Legacy checkpoint: use the archived workflow")
    if record["scientific"]["source"] != source_hashes():
        raise ValueError("Use the recorded source snapshot to evaluate this checkpoint")
    if split == "test":
        lock = verify_lock(cfg, cache)
        if run.name not in lock["runs"] or checkpoint != "best.pt":
            raise ValueError("Only frozen selected checkpoints may evaluate fold 10")
    manifest = read_json(Path(cache) / "manifest.json")
    if record["scientific"]["cohort"] != manifest["cohort"]:
        raise ValueError("Evaluation cohort differs from training")
    if record["scientific"]["scaler_sha256"] != file_hash(Path(cache) / "scaler.json"):
        raise ValueError("Evaluation scaler differs from training")
    if record["preprocessing"] == "P1" and record["scientific"]["filter"] != cfg["preprocessing"]:
        raise ValueError("Evaluation filter differs from training")
    scaler = prepare_transforms(cache, cfg, record["preprocessing"], include_test=split == "test")
    arrays, names, _ = static_inputs(
        record["config"]
        | {
            "data": cfg["data"],
            "evaluation": cfg["evaluation"],
            "paths": cfg["paths"],
            "study_id": cfg["study_id"],
        },
        cache,
        record["feature_set"],
        record["selection"],
        include_test=split == "test",
    )
    if names != record["selected_features"]:
        raise ValueError("Evaluation feature order differs from training")
    dataset = ECGDataset(cache, cfg, split, record["preprocessing"], arrays[split])
    device = device_for(cfg)
    model = build_model(
        record["architecture"], len(names), cfg["data"]["horizon"], **record["config"]["model"]
    ).to(device)
    state = torch.load(run / checkpoint, map_location=device, weights_only=True)
    if state["schema"] != SCHEMA or state["signature"] != record["signature"]:
        raise ValueError("Checkpoint signature mismatch")
    model.load_state_dict(state["model"])
    prediction, target = predict(model, loader(dataset, cfg), device)
    prediction = inverse_scale(prediction, scaler)
    # Use the stored physical truth directly, avoiding normalization round-off in ground truth.
    target = np.asarray(dataset.waves[:, cfg["data"]["history"] :])
    result = save_evaluation(
        cfg,
        cache,
        label or run.name,
        split,
        prediction,
        target,
        {
            "architecture": record["architecture"],
            "preprocessing": record["preprocessing"],
            "feature_set": record["feature_set"],
            "parameters": record["parameters"],
            "best_epoch": state["epoch"],
            "checkpoint_sha256": file_hash(run / checkpoint),
            "run_signature": record["signature"],
            "runtime_seconds": read_json(run / "status.json")["runtime_seconds"],
        },
    )
    print(
        f"{run.name} {split}: patient-macro MAE={result['patient_macro_mae_mV']:.6f} mV", flush=True
    )
    return result


def baseline(cfg, cache, name, split="val"):
    if split not in ("val", "test"):
        raise ValueError("Baselines evaluate validation or locked final test only")
    if split == "test":
        verify_lock(cfg, cache)
    if name not in ("persistence", "repeat_window"):
        raise ValueError("Supported baselines: persistence, repeat_window; ARIMA is deferred")
    raw = np.load(Path(cache) / f"waves_{split}.npy", mmap_mode="r")
    h, f = cfg["data"]["history"], cfg["data"]["horizon"]
    target = np.asarray(raw[:, h : h + f])
    prediction = (
        np.repeat(raw[:, h - 1 : h], f, axis=1)
        if name == "persistence"
        else np.asarray(raw[:, h - f : h]).copy()
    )
    return save_evaluation(
        cfg,
        cache,
        name,
        split,
        prediction,
        target,
        {
            "architecture": "baseline",
            "preprocessing": "raw",
            "feature_set": "F0",
            "parameters": 0,
            "definition": "repeat last observed voltage"
            if name == "persistence"
            else "repeat last horizon-length observed segment",
        },
    )


def ranking_key(row):
    return (
        row["patient_macro_mae_mV"],
        row["rmse_mV"],
        row.get("parameters", 0),
        row.get("runtime_seconds", 0),
        row["name"],
    )


def freeze_selection(cfg, cache, runs):
    if cfg["run_mode"] != "full":
        raise ValueError("Only full-population validation may freeze final selection")
    manifest = read_json(Path(cache) / "manifest.json")
    path = study_dir(cfg) / "full" / "selection.json"
    items, recipes, highlights = {}, {}, {}
    for run in map(Path, runs):
        record, status = read_json(run / "run.json"), read_json(run / "status.json")
        expected = 120 if not manifest["synthetic"] else cfg["training"]["full_epochs"]
        if (
            record["scientific"]["cohort"] != manifest["cohort"]
            or status["epochs_completed"] != expected
            or record["epoch_budget"] != expected
        ):
            raise ValueError(
                "Final conditions must complete the full 120-epoch budget on this cohort"
            )
        metrics_path = outputs(cfg, run.name) / "val.json"
        verify_evaluation(metrics_path.parent, "val")
        metrics = read_json(metrics_path)
        if metrics["checkpoint_sha256"] != file_hash(run / "best.pt"):
            raise ValueError("Validation evaluation is stale")
        if record["scientific"]["scaler_sha256"] != file_hash(Path(cache) / "scaler.json"):
            raise ValueError("Validation scaler is stale")
        if (
            record["preprocessing"] == "P1"
            and record["scientific"]["filter"] != cfg["preprocessing"]
        ):
            raise ValueError("Validation filter is stale")
        pair = (record["architecture"], record["preprocessing"])
        if pair in recipes or record["feature_set"] not in ("F0", "F1"):
            raise ValueError(
                "Final selection requires one F0/F1 run for each model/filter condition"
            )
        recipes[pair] = record
        items[run.name] = {
            "checkpoint_sha256": file_hash(run / "best.pt"),
            "run_sha256": file_hash(run / "run.json"),
            "validation_sha256": file_hash(metrics_path),
            "validation_receipt_sha256": file_hash(metrics_path.with_name("val_receipt.json")),
            "recipe": {
                "architecture": record["architecture"],
                "preprocessing": record["preprocessing"],
                "feature_set": record["feature_set"],
                "lr": record["config"]["training"]["lr"],
                "hidden": record["config"]["model"]["hidden"],
                "selected_epoch": metrics["best_epoch"],
            },
        }
    if set(recipes) != {(m, p) for m in ("lstm", "cnn_lstm") for p in ("P0", "P1")}:
        raise ValueError("Freeze exactly four matched model/filter conditions")
    for model in ("lstm", "cnn_lstm"):
        a, b = [recipes[(model, p)] for p in ("P0", "P1")]
        for key in (
            "model",
            "features",
            "feature_statistics",
            "training",
            "budget",
            "cohort",
            "source",
            "versions",
            "scaler_sha256",
        ):
            if a["scientific"][key] != b["scientific"][key]:
                raise ValueError(f"Unmatched P0/P1 control: {key}")
        choices = [read_json(outputs(cfg, r["name"]) / "val.json") for r in (a, b)]
        if choices[0]["target_sha256"] != choices[1]["target_sha256"]:
            raise ValueError("The P0/P1 validation targets differ")
        highlights[model] = min(choices, key=ranking_key)["name"]
    lock = {
        "schema": SCHEMA,
        "cohort": manifest["cohort"],
        "manifest_sha256": file_hash(Path(cache) / "manifest.json"),
        "protocol": protocol(cfg),
        "scaler_sha256": file_hash(Path(cache) / "scaler.json"),
        "filter": cfg["preprocessing"],
        "runs": items,
        "highlighted_by_validation": highlights,
        "baselines": ["persistence", "repeat_window"],
        "synthetic": manifest["synthetic"],
        "used_test_for_selection": False,
    }
    if path.exists():
        existing = read_json(path)
        if {k: v for k, v in existing.items() if k != "frozen_at"} != lock:
            raise ValueError("Selection is already frozen; use a new study ID")
        return existing
    lock["frozen_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, lock)
    return lock


def final_evaluation(cfg, cache):
    lock = verify_lock(cfg, cache)
    cache, _ = prepare(cfg, include_test=True, lock=lock)
    path = study_dir(cfg) / "full" / "test_ledger.json"
    ledger = read_json(path) if path.exists() else {"lock_hash": digest(lock), "conditions": {}}
    if ledger["lock_hash"] != digest(lock):
        raise ValueError("Final-test ledger belongs to another selection")
    for name in list(lock["runs"]) + lock["baselines"]:
        receipt = outputs(cfg, name) / "test.json"
        previous = ledger["conditions"].get(name, {})
        if previous.get("state") == "completed":
            if previous["metrics_sha256"] != file_hash(receipt):
                raise ValueError("Completed test result changed")
            verify_evaluation(receipt.parent, "test")
            continue
        if previous.get("state") == "running" and (receipt.parent / "test_receipt.json").exists():
            verify_evaluation(receipt.parent, "test")
            saved = read_json(receipt)
            if (
                saved["cohort"] != lock["cohort"]
                or saved.get("checkpoint_sha256") != previous["checkpoint_sha256"]
            ):
                raise ValueError("Interrupted test result does not match the selection")
            ledger["conditions"][name].update(state="completed", metrics_sha256=file_hash(receipt))
            write_json(path, ledger)
            continue
        ledger["conditions"][name] = {
            "state": "running",
            "checkpoint_sha256": lock["runs"].get(name, {}).get("checkpoint_sha256"),
        }
        write_json(path, ledger)
        if name in lock["baselines"]:
            baseline(cfg, cache, name, "test")
        else:
            evaluate_run(cfg, cache, study_dir(cfg) / "full" / "checkpoints" / name, "test")
        ledger["conditions"][name].update(state="completed", metrics_sha256=file_hash(receipt))
        write_json(path, ledger)
    return ledger
