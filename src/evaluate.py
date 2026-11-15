<<<<<<< HEAD
"""Explicit validation/test inference, peak scoring and waveform figures."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from config import (
    LEADS,
    VARIANTS,
    file_hash,
    load_config,
    read_json,
    run_directory,
    runtime,
    write_json,
)
from metrics import metric_summary, physical_arrays, subgroup_metrics
from models import build_model
from morphology import qrs_metrics
from preprocessing import ECGDataset, make_loaders, validate_cache


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_examples(x, pred, true, meta, out, cfg):
    fs = cfg["data"]["sampling_rate"]
    errors = np.abs(pred - true).mean(axis=(1, 2))
    order = np.argsort(errors, kind="stable")
    selected = {
        "lowest_id": int(np.argmin(meta.ecg_id)),
        "median_error": int(order[len(order) // 2]),
        "worst_error": int(order[-1]),
    }
    for selection, idx in selected.items():
        fig, axes = plt.subplots(6, 2, figsize=(14, 13), sharex=True)
        tx = (np.arange(x.shape[-1]) - x.shape[-1]) / fs
        ty = np.arange(true.shape[-1]) / fs
        for k, ax in enumerate(axes.flat):
            ax.plot(tx, x[idx, k], color="0.6", lw=0.9, label="Observed")
            ax.plot(ty, true[idx, k], lw=1.2, label="Actual")
            ax.plot(ty, pred[idx, k], lw=1.1, label="Forecast")
            ax.axvline(0, color="black", ls="--", lw=0.7)
            ax.set(ylabel=f"{LEADS[k]} (mV)")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Seconds relative to forecast origin")
        fig.suptitle(f"{selection}: ECG {int(meta.iloc[idx].ecg_id)}, MAE {errors[idx]:.4f} mV")
        save_figure(fig, out / f"forecast_{selection}.png")
    return {name: int(meta.iloc[i].ecg_id) for name, i in selected.items()}


def learning_curves(run, cfg):
    run = Path(run)
    history = pd.read_csv(run / "history.csv")
    out = Path(cfg["paths"]["outputs"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    contrastive = bool(history.contrastive_loss.abs().sum())
    fig, axes = plt.subplots(
        1, 2 if contrastive else 1, figsize=(12 if contrastive else 7, 4), squeeze=False
    )
    ax = axes[0, 0]
    key = "patient_macro_mae" if "val_patient_macro_mae" in history else "mse"
    ax.plot(history.epoch, history["train_" + key], label="Training")
    ax.plot(history.epoch, history["val_" + key], label="Validation")
    ax.set(
        ylabel="Patient-macro MAE (mV)" if key != "mse" else "MSE (standardized units)",
        xlabel="Epoch",
        title=run.name,
    )
    ax.legend()
    if contrastive:
        axes[0, 1].plot(history.epoch, history.contrastive_loss, label="Auxiliary NT-Xent")
        axes[0, 1].set(xlabel="Epoch", ylabel="Contrastive loss")
        axes[0, 1].legend()
    save_figure(fig, out / "learning_curves.png")


def predict(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
=======
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
>>>>>>> origin/work/manual-src-workflow
            predictions.append(
                model(batch["x"].to(device), batch["static"].to(device)).cpu().numpy()
            )
            targets.append(batch["y"].numpy())
    return np.concatenate(predictions), np.concatenate(targets)


<<<<<<< HEAD
def authorize_split(cfg, split, run=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if split == "test":
        settings = cfg.get("evaluation", {})
        if not settings.get("allow_test", False) or cfg.get("subset", {}).get("mode") != "full":
            raise ValueError("Fold 10 is reserved: require full cohort and explicit allow_test")
        selection = read_json(settings["selection_path"])
        if (
            selection["fingerprint"]
            != read_json(Path(cfg["data"]["cache"]) / "manifest.json")["fingerprint"]
        ):
            raise ValueError("Final selection cohort mismatch")
        if run is not None and run.name not in ("persistence", "repeat_window", "training_center"):
            if selection["checkpoints"].get(run.name) != file_hash(run / "best.pt"):
                raise ValueError("Checkpoint is not in the frozen validation selection")


def arrays_evaluation(cfg, run, split, pred, true, observed, meta, manifest):
    import matplotlib.pyplot as plt

    out = Path(cfg["paths"]["outputs"]) / run.name / split
    out.mkdir(parents=True, exist_ok=True)
    summary = metric_summary(pred, true, meta.patient_id.to_numpy())
    summary.update(
        split=split,
        fingerprint=manifest["fingerprint"],
        comparison_key={k: manifest["protocol"][k] for k in ("data", "subset")},
    )
    qrs, peaks, detections, timing = qrs_metrics(
        pred,
        true,
        observed,
        cfg["data"]["sampling_rate"],
        cfg["evaluation"].get("peak_tolerance_ms", 100),
    )
    summary.update(qrs)
    peaks.to_csv(out / "peak_metrics.csv", index=False)
    per_record = meta.copy()
    error = pred.astype(float) - true.astype(float)
    per_record["mae_mV"] = np.abs(error).mean(axis=(1, 2))
    per_record["mse_mV2"] = (error**2).mean(axis=(1, 2))
    per_record.to_csv(run / f"{split}_per_record_metrics.csv", index=False)
    subgroup_metrics(pred, true, meta).to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(
        {
            "lead": manifest["protocol"]["leads"],
            "mae_mV": summary["mae_by_lead"],
            "rmse_mV": summary["rmse_by_lead"],
            "pearson": summary["pearson_by_lead"],
        }
    ).to_csv(out / "per_lead_metrics.csv", index=False)
    pd.DataFrame(
        {
            "seconds": np.arange(pred.shape[-1]) / cfg["data"]["sampling_rate"],
            "mae_mV": np.abs(error).mean(axis=(0, 1)),
            "rmse_mV": np.sqrt((error**2).mean(axis=(0, 1))),
        }
    ).to_csv(out / "horizon_metrics.csv", index=False)
    metric_plots(out)
    summary["example_ecg_ids"] = plot_examples(observed, pred, true, meta, out, cfg)
    example = int(np.argsort(per_record.mae_mV.to_numpy(), kind="stable")[len(meta) // 2])
    fs = cfg["data"]["sampling_rate"]
    t = np.arange(pred.shape[-1]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax in axes:
        ax.plot(t, true[example, 1], label="Actual lead II")
        ax.plot(t, pred[example, 1], label="Predicted lead II")
        for values, color, label in zip(
            detections[example], ["black", "red"], ["Actual R", "Predicted R"], strict=True
        ):
            voltage = true if color == "black" else pred
            ax.scatter(t[values], voltage[example, 1, values], c=color, marker="x", label=label)
        ax.legend(fontsize=8)
        ax.set(xlabel="Forecast time (s)", ylabel="mV")
    reference = detections[example][0]
    if len(reference):
        center = reference[len(reference) // 2] / fs
        axes[1].set_xlim(max(0, center - 0.25), min(pred.shape[-1] / fs, center + 0.25))
    axes[0].set_title(
        f"Median-error example ECG {int(meta.iloc[example].ecg_id)}; detector-derived peaks"
    )
    save_figure(fig, out / "qrs_overlay_zoom.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(timing, bins=20)
    ax.set(
        xlabel="Matched peak timing error (ms)",
        ylabel="Peaks",
        title="No matched peaks" if not timing else "Matched R-peak timing errors",
    )
    save_figure(fig, out / "peak_timing_errors.png")
    np.savez_compressed(
        run / f"{split}_predictions.npz",
        pred=pred,
        true=true,
        ecg_id=meta.ecg_id.to_numpy(),
        patient_id=meta.patient_id.to_numpy(),
    )
    write_json(run / f"{split}_metrics.json", summary)
    return summary


def metric_plots(out):
    """Render already-saved metrics without repeating inference or selecting examples."""
    import matplotlib.pyplot as plt

    for filename, index, title in (
        ("per_lead_metrics", "lead", "Errors by ECG lead"),
        ("horizon_metrics", "seconds", "Errors through the forecast horizon"),
        ("subgroup_metrics", "group", "Errors by overlapping patient subgroups"),
    ):
        frame = pd.read_csv(out / f"{filename}.csv")
        fig, ax = plt.subplots(figsize=(11, 5))
        values = frame.set_index(index)[["mae_mV", "rmse_mV"]]
        if filename == "horizon_metrics":
            values.plot(ax=ax)
        else:
            values.plot.bar(ax=ax)
            ax.tick_params(axis="x", rotation=40)
        ax.set(ylabel="Error (mV)", title=title)
        save_figure(fig, out / f"{filename}.png")


def evaluate_run(cfg, architecture, variant, split="val"):
    manifest = validate_cache(cfg)
    run = run_directory(cfg, architecture, variant)
    authorize_split(cfg, split, run)
    if not (run / "best.pt").exists() or not (run / "config.json").exists():
        raise ValueError(f"No selected checkpoint at {run}; run train.py first")
    recorded = read_json(run / "config.json")
    if recorded["architecture"] != architecture or recorded["variant"] != variant:
        raise ValueError("Requested model/variant differs from the saved run")
    if recorded["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Evaluation cache differs from training")
    loader = make_loaders(cfg, variant, splits=(split,))[split]
    if loader.dataset.selected_features != recorded["selected_features"]:
        raise ValueError("Selected feature order differs from training")
    device = runtime(cfg)
    model = build_model(recorded["config"], architecture, loader.dataset.static_dim).to(device)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    pred, true = predict(model, loader, device)
    pred, true = [physical_arrays(a, manifest, cfg["data"].get("layout")) for a in (pred, true)]
    observed = np.load(Path(cfg["data"]["cache"]) / f"raw_X_{split}.npy")
    summary = arrays_evaluation(
        cfg, run, split, pred, true, observed, loader.dataset.metadata, manifest
    )
    summary.update(best_epoch=checkpoint["epoch"], checkpoint_sha256=file_hash(run / "best.pt"))
    write_json(run / f"{split}_metrics.json", summary)
    print(f"{run.name} {split}: patient-macro MAE={summary['patient_macro_mae']:.6f} mV", flush=True)
    return summary


def baseline_runs(cfg, split="val"):
    manifest = validate_cache(cfg)
    authorize_split(cfg, split)
    ds = ECGDataset(cfg, split, "ecg_only")
    observed = np.load(Path(cfg["data"]["cache"]) / f"raw_X_{split}.npy")
    true = physical_arrays(np.asarray(ds.y), manifest)
    horizon = cfg["data"]["horizon"]
    predictions = {
        "persistence": np.repeat(observed[:, :, -1:], horizon, axis=2),
        "repeat_window": np.tile(
            observed[:, :, -min(horizon, observed.shape[-1]) :],
            (1, 1, int(np.ceil(horizon / min(horizon, observed.shape[-1])))),
        )[:, :, :horizon],
    }
    # A train-only constant reference exposes apparent gains from near-flat outputs.
    training_raw = np.load(Path(cfg["data"]["cache"]) / "raw_X_train.npy", mmap_mode="r")
    center = np.array([np.median(training_raw[:, j, :]) for j in range(12)])
    predictions["training_center"] = np.broadcast_to(center[None, :, None], true.shape).copy()
    for name, pred in predictions.items():
        run = Path(cfg["paths"]["experiments"]) / name
        run.mkdir(parents=True, exist_ok=True)
        write_json(
            run / "config.json",
            {
                "config": cfg,
                "baseline": name,
                "definition": "constant per-lead training-input median"
                if name == "training_center"
                else name,
            },
        )
        arrays_evaluation(cfg, run, split, pred, true, observed, ds.metadata, manifest)


def freeze_selection(cfg, names):
    manifest = validate_cache(cfg)
    checkpoints = {}
    for name in names:
        run = Path(cfg["paths"]["experiments"]) / name
        metrics = read_json(run / "val_metrics.json")
        if metrics["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Selection cohort mismatch")
        checkpoints[name] = file_hash(run / "best.pt")
    path = Path(cfg["paths"]["experiments"]) / "selection.json"
    if path.exists():
        raise ValueError("Selection already frozen; use a new experiment directory")
    write_json(
        path,
        {
            "fingerprint": manifest["fingerprint"],
            "checkpoints": checkpoints,
            "objective": "validation patient-macro MAE (mV)",
            "used_test": False,
        },
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--model", "-model", choices=["lstm", "cnn_lstm"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--baselines", action="store_true", help="Evaluate all three simple forecasts")
    parser.add_argument(
        "--freeze", nargs="+", metavar="RUN",
        help="Freeze validation-selected checkpoints before test inference",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.experiment)
    if args.device:
        cfg["training"]["device"] = args.device
    with FileLock(".pipeline.lock", timeout=0):
        if args.freeze:
            print(freeze_selection(cfg, args.freeze))
        elif args.baselines:
            baseline_runs(cfg, args.split)
        else:
            evaluate_run(
                cfg,
                args.model or cfg.get("experiment", {}).get("architecture", "lstm"),
                args.variant or cfg.get("experiment", {}).get("variant", "patient_history"),
                args.split,
            )
=======
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
>>>>>>> origin/work/manual-src-workflow
