"""Matched Week 5 conditions, with validation-only comparisons and extensions."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    LEADS,
    digest,
    file_hash,
    load_config,
    read_json,
    source_hashes,
    study_dir,
    validate,
)
from data import prepare, read_wave
from evaluate import evaluation_model, freeze_selection, outputs, ranking_key, verify_evaluation
from metrics import bootstrap_pair
from study import condition

EXPERIMENTS = {
    "E1": ("lstm", "P0"),
    "E2": ("lstm", "P1"),
    "E3": ("cnn_lstm", "P0"),
    "E4": ("cnn_lstm", "P1"),
}
CONTRASTS = {
    "LSTM: filtering": ("E1", "E2"),
    "CNN-LSTM: filtering": ("E3", "E4"),
    "Architecture without filtering": ("E1", "E3"),
    "Architecture with filtering": ("E2", "E4"),
}


def configure(settings, base=None):
    """Map the editable notebook cell to the shared training configuration."""
    cfg = (
        copy.deepcopy(base)
        if base
        else load_config(Path(__file__).resolve().parents[1] / "configs/config.yaml")
    )
    mode = settings["mode"]
    cfg.update(study_id=settings["study_id"], run_mode=mode)
    cfg["paths"] = {"cache": str(settings["cache_dir"]), "results": str(settings["results_dir"])}
    hz = settings["sampling_rate"]
    history = settings["history_seconds"] * hz
    forecast = settings["forecast_seconds"] * hz
    if hz != 100 or (history, forecast) != (700, 300):
        raise ValueError("The four primary conditions require native 100 Hz and a 7s → 3s window.")
    cfg["data"].update(
        root=str(settings["data_root"]),
        sampling_rate=hz,
        history=int(history),
        horizon=int(forecast),
    )
    for subset in ("pilot", "medium"):
        cfg["cohorts"][subset] = {
            "train": settings["train_patients" if subset == "pilot" else "medium_train_patients"],
            "val": settings["val_patients" if subset == "pilot" else "medium_val_patients"],
        }
    long_budget, short_budget = settings["epochs"], settings["smoke_epochs"]
    if not 1 <= short_budget <= long_budget <= 120:
        raise ValueError("Epoch budgets must satisfy 1 ≤ smoke_epochs ≤ epochs ≤ 120")
    if mode == "full" and settings["smoke_test"]:
        raise ValueError("Choose pilot for a smoke test; full mode uses every development patient.")
    budget = short_budget if settings["smoke_test"] else long_budget
    cfg["training"].update(
        batch_size=settings["batch_size"],
        lr=settings["learning_rate"],
        seed=settings["seed"],
        device=settings["device"],
        allow_full_cpu=settings["allow_full_cpu"],
        pilot_epochs=budget,
        epochs=budget,
        full_epochs=budget,
    )
    cfg["model"].update(
        hidden=settings["hidden_size"], depth=settings["num_layers"], dropout=settings["dropout"]
    )
    cfg["preprocessing"].update(
        low_hz=settings["filter_low_hz"], high_hz=settings["filter_high_hz"]
    )
    cfg["evaluation"]["final_test"] = bool(settings["final_test"])
    validate(cfg)
    return cfg


def epoch_budget(cfg):
    key = {"pilot": "pilot_epochs", "medium": "epochs", "full": "full_epochs"}[cfg["run_mode"]]
    return cfg["training"][key]


def run_path(cfg, experiment, features="F0"):
    return (
        study_dir(cfg)
        / cfg["run_mode"]
        / "checkpoints"
        / (f"{experiment}_{features}_e{epoch_budget(cfg)}")
    )


def run_experiment(cfg, cache, experiment, *, features="F0", resume=False):
    architecture, filtering = EXPERIMENTS[experiment]
    return condition(
        cfg,
        cache,
        architecture,
        filtering,
        features,
        resume=resume,
        name=run_path(cfg, experiment, features).name,
    )


def validation_result(cfg, cache, experiment, features="F0"):
    """Read evidence only if it still belongs to this exact condition; never train."""
    run = run_path(cfg, experiment, features)
    folder = outputs(cfg, run.name)
    if not (folder / "val_receipt.json").exists():
        return None
    record = read_json(run / "run.json")
    science = record["scientific"]
    manifest = read_json(Path(cache) / "manifest.json")
    architecture, filtering = EXPERIMENTS[experiment]
    expected_training = {key: cfg["training"][key] for key in science["training"]}
    if (
        record["signature"] != digest(science)
        or science["cohort"] != manifest["cohort"]
        or science["model"] != cfg["model"]
        or science["source"] != source_hashes()
        or science["training"] != expected_training
        or science["scaler_sha256"] != file_hash(Path(cache) / "scaler.json")
        or science["filter"] != (cfg["preprocessing"] if filtering == "P1" else None)
        or (record["architecture"], record["preprocessing"], record["feature_set"])
        != (architecture, filtering, features)
        or read_json(run / "status.json")["epochs_completed"] != epoch_budget(cfg)
    ):
        raise ValueError(f"{run.name}: saved evidence is incompatible; use a new study_id.")
    verify_evaluation(folder, "val")
    scores = read_json(folder / "val.json")
    if scores["run_signature"] != record["signature"] or scores["checkpoint_sha256"] != file_hash(
        run / "best.pt"
    ):
        raise ValueError(f"{run.name}: validation evidence does not match the checkpoint.")
    return scores


def percentage_reduction(reference, candidate):
    return 100 * (reference - candidate) / reference if reference > 0 else None


def compare_validation(reference_cfg, reference, candidate_cfg, candidate):
    """Compare the same validation ECGs in original mV, resampling patients."""
    if any(x["split"] != "val" for x in (reference, candidate)):
        raise ValueError("Week 5 comparisons use validation only.")
    if reference["protocol"] != candidate["protocol"] or (
        reference["target_sha256"] != candidate["target_sha256"]
    ):
        raise ValueError("Comparisons require identical validation targets and protocol.")
    first = pd.read_csv(outputs(reference_cfg, reference["name"]) / "val_records.csv")
    second = pd.read_csv(outputs(candidate_cfg, candidate["name"]) / "val_records.csv")
    interval = bootstrap_pair(first, second, seed=reference_cfg["training"]["seed"])
    return {
        "reference": reference["name"],
        "candidate": candidate["name"],
        "reference_patient_MAE_mV": reference["patient_macro_mae_mV"],
        "candidate_patient_MAE_mV": candidate["patient_macro_mae_mV"],
        "MAE_mV": candidate["mae_mV"],
        "RMSE_mV": candidate["rmse_mV"],
        "improvement_pct": percentage_reduction(
            reference["patient_macro_mae_mV"], candidate["patient_macro_mae_mV"]
        ),
        "difference_mV": interval["difference_B_minus_A_mV"],
        "CI_low_mV": interval["ci95_low"],
        "CI_high_mV": interval["ci95_high"],
        "interpretation": interval["conclusion"],
    }


def primary_comparison(cfg, cache):
    measured = {eid: validation_result(cfg, cache, eid) for eid in EXPERIMENTS}
    table = []
    for eid, (architecture, filtering) in EXPERIMENTS.items():
        score = measured[eid]
        fields = (
            "mae_mV",
            "rmse_mV",
            "patient_macro_mae_mV",
            "mse_mV2",
            "parameters",
            "best_epoch",
            "runtime_seconds",
        )
        table.append(
            {
                "experiment": eid,
                "architecture": architecture,
                "filter": filtering,
                "status": "measured" if score else "pending",
                **{key: score[key] if score else None for key in fields},
            }
        )
    comparisons = []
    for question, (a, b) in CONTRASTS.items():
        row = {"question": question, "comparison": f"{a} → {b}"}
        if measured[a] and measured[b]:
            row.update(compare_validation(cfg, measured[a], cfg, measured[b]))
        else:
            row["interpretation"] = "pending"
        comparisons.append(row)
    available = {eid: score for eid, score in measured.items() if score}
    ranking = sorted(available, key=lambda eid: ranking_key(available[eid]))
    ranks = {eid: rank for rank, eid in enumerate(ranking, 1)}
    for row in table:
        row["rank"] = ranks.get(row["experiment"])
    return pd.DataFrame(table), pd.DataFrame(comparisons), available


def training_ablation(cfg, cache, winner, smaller_patients, shorter_epochs, *, resume=False):
    """Change patient count or epoch budget separately; keep validation IDs fixed."""
    reference = validation_result(cfg, cache, winner)
    manifest = read_json(Path(cache) / "manifest.json")
    n_train = len(manifest["selected_patients"]["train"])
    if not 0 < smaller_patients < n_train or not 0 < shorter_epochs < epoch_budget(cfg):
        raise ValueError("Ablation patient and epoch budgets must be smaller than the main run.")
    small = copy.deepcopy(cfg)
    small.update(study_id=f"{cfg['study_id']}_size_{smaller_patients}", run_mode="pilot")
    val = manifest["selected_patients"]["val"]
    val_limit = None if cfg["run_mode"] == "full" else cfg["cohorts"][cfg["run_mode"]]["val"]
    small["cohorts"]["pilot"] = {"train": smaller_patients, "val": val_limit}
    small["training"]["pilot_epochs"] = epoch_budget(cfg)
    smaller_cache, smaller_manifest = prepare(small)
    if smaller_manifest["selected_patients"]["val"] != val or not set(
        smaller_manifest["selected_patients"]["train"]
    ).issubset(manifest["selected_patients"]["train"]):
        raise ValueError("Ablation requires identical validation and nested training patients.")
    _, small_score = run_experiment(small, smaller_cache, winner, resume=resume)
    short = copy.deepcopy(cfg)
    short["training"][
        {"pilot": "pilot_epochs", "medium": "epochs", "full": "full_epochs"}[cfg["run_mode"]]
    ] = shorter_epochs
    _, short_score = run_experiment(short, cache, winner, resume=resume)
    return pd.DataFrame(
        [
            {
                "change": f"{smaller_patients} → {n_train} train patients",
                **compare_validation(small, small_score, cfg, reference),
            },
            {
                "change": f"{shorter_epochs} → {epoch_budget(cfg)} epochs",
                **compare_validation(short, short_score, cfg, reference),
            },
        ]
    )


def quality_sample(frame, cfg, limit):
    """Bound the exploratory scan; preparation separately validates every selected ECG."""
    valid, problems = [], []
    for row in frame.head(limit).itertuples(index=False):
        try:
            valid.append(read_wave(row, cfg["data"]["root"]))
        except (ValueError, OSError) as exc:
            problems.append({"ecg_id": row.ecg_id, "problem": str(exc)})
    amplitude = pd.DataFrame()
    if valid:
        voltage = np.concatenate(valid, axis=0)
        percentiles = np.quantile(voltage, [0.01, 0.5, 0.99], axis=0)
        amplitude = pd.DataFrame(
            percentiles.T, index=LEADS, columns=["p01_mV", "median_mV", "p99_mV"]
        )
    return len(valid), pd.DataFrame(problems, columns=["ecg_id", "problem"]), amplitude


def validate_final(cfg, cache):
    """Freeze all four exact full-data ECG-only conditions before final evaluation."""
    if cfg["run_mode"] != "full" or not cfg["evaluation"]["final_test"]:
        raise ValueError("Choose full mode and explicitly enable final_test before freezing.")
    if not all(validation_result(cfg, cache, eid) for eid in EXPERIMENTS):
        raise ValueError("All four ECG-only full-data conditions must finish before final testing.")
    return freeze_selection(cfg, cache, [run_path(cfg, eid) for eid in EXPERIMENTS])


def temporal_occlusion(cfg, cache, run, *, patients=12, window_seconds=1.0):
    """Perturb one lead/time block in the actual normalized validation model input."""
    model, dataset, device, scaler, _, _ = evaluation_model(cfg, cache, run)
    rows = dataset.metadata
    chosen = rows.drop_duplicates("patient_id").head(patients)
    width = int(round(window_seconds * cfg["data"]["sampling_rate"]))
    if not len(chosen) or width < 1:
        raise ValueError("Choose at least one validation patient and a positive interval.")
    items = [dataset[int(i)] for i in chosen.index]
    x = torch.stack([item["x"] for item in items]).to(device)
    static = torch.stack([item["static"] for item in items]).to(device)
    y = torch.as_tensor(np.array(dataset.waves[chosen.index, dataset.history :]), device=device)
    std = torch.as_tensor(scaler["std"], dtype=x.dtype, device=device)
    mean = torch.as_tensor(scaler["mean"], dtype=x.dtype, device=device)
    model.eval()
    intervals, matrix = [], []
    with torch.inference_mode():
        original = (model(x, static) * std + mean - y).abs().mean(dim=(1, 2))
        for start in range(0, x.shape[1], width):
            end = min(start + width, x.shape[1])
            increments = []
            for lead in range(x.shape[2]):
                masked = x.clone()
                masked[:, start:end, lead] = 0.0  # train mean in normalized input space
                error = (model(masked, static) * std + mean - y).abs().mean(dim=(1, 2))
                increments.append(float((error - original).mean().cpu()))
            intervals.append(f"{start / 100:g}–{end / 100:g}s")
            matrix.append(increments)
    return pd.DataFrame(np.asarray(matrix).T, index=LEADS, columns=intervals), {
        "patients": len(chosen),
        "ecg_ids": chosen.ecg_id.astype(int).tolist(),
        "baseline_MAE_mV": float(original.mean().cpu()),
        "checkpoint_sha256": file_hash(Path(run) / "best.pt"),
        "window_seconds": window_seconds,
        "baseline": "zero in normalized model input (raw training mean)",
    }
