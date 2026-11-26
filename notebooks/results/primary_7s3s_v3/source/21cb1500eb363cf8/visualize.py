"""Figures with physical axes, reproducible examples, and explicit captions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from config import LEADS, read_json, study_dir, write_json
from data import metadata
from preprocessing import filter_history


def save(fig, path, shows, conclusion):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    path.with_suffix(".md").write_text(f"{shows}\n\n{conclusion}\n", encoding="utf-8")


def diagnostic_figures(cfg, cache):
    out = study_dir(cfg) / cfg["run_mode"] / "figures" / "data"
    frame = metadata(cfg["data"]["root"])
    counts = frame.groupby("strat_fold").agg(
        records=("ecg_id", "size"), patients=("patient_id", "nunique")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot.bar(ax=ax)
    ax.set(xlabel="Official fold", ylabel="Count", title="Official record and patient assignments")
    save(
        fig,
        out / "fold_counts.png",
        "The chart shows metadata counts in the official folds.",
        "No patient occurs in more than one fold; fold 10 is reserved and no test scores are used here.",
    )
    raw = np.load(Path(cache) / "waves_val.npy", mmap_mode="r")
    rows = pd.read_csv(Path(cache) / "metadata_val.csv")
    index = int(np.argmin(rows.ecg_id.to_numpy()))
    wave, h = raw[index], cfg["data"]["history"]
    observed, filtered = wave[:h], filter_history(wave[:h], cfg)
    for kind in ("original", "filtered_history", "morphology_zoom", "psd"):
        fig, axes = plt.subplots(4, 3, figsize=(14, 11))
        for lead, ax in enumerate(axes.flat):
            if kind == "original":
                ax.plot(np.arange(1000) / 100, wave[:, lead], lw=0.7)
                ax.axvline(h / 100, color="black", ls="--")
                ax.set(xlabel="Time (s)", ylabel="mV")
            elif kind == "psd":
                for values, label in (
                    (observed, "Raw history"),
                    (filtered, "Causal filtered history"),
                ):
                    frequency, density = signal.welch(values[:, lead], fs=100, nperseg=min(256, h))
                    ax.semilogy(frequency, np.maximum(density, 1e-15), label=label)
                ax.set(xlabel="Hz", ylabel="mV²/Hz")
            else:
                time = np.arange(h) / 100
                ax.plot(time, observed[:, lead], label="Raw history", lw=0.8)
                ax.plot(time, filtered[:, lead], label="Causal filtered history", lw=0.8)
                ax.set(xlabel="Observed time (s)", ylabel="mV")
                if kind == "morphology_zoom":
                    center = np.argmax(np.abs(observed[:, 1] - np.median(observed[:, 1]))) / 100
                    ax.set_xlim(max(0, center - 0.25), min(h / 100, center + 0.25))
            ax.set_title(LEADS[lead])
        if kind != "original":
            axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"ECG {int(rows.iloc[index].ecg_id)}: {kind.replace('_', ' ')}", y=1.01)
        save(
            fig,
            out / f"{kind}.png",
            f"This figure shows all twelve leads for the lowest-ID validation recording ({kind}).",
            "Filtering changes observed voltages and phase; these plots alone do not establish better forecasts or clinical morphology preservation.",
        )
    write_json(
        out / "filter_distortion.json",
        {
            "ecg_id": int(rows.iloc[index].ecg_id),
            "history_only_rmse_mV_by_lead": np.sqrt(
                ((filtered - observed) ** 2).mean(axis=0)
            ).tolist(),
            "future_used": False,
            "filter_prototype_order": cfg["preprocessing"]["order"],
            "bandpass_total_order": 2 * cfg["preprocessing"]["order"],
        },
    )


def run_figures(cfg, cache, run, split="val"):
    run = Path(run)
    metrics_dir = study_dir(cfg) / cfg["run_mode"] / "metrics" / run.name
    out = study_dir(cfg) / cfg["run_mode"] / "figures" / run.name
    if (run / "history.csv").exists():
        history = pd.read_csv(run / "history.csv")
        selected = read_json(metrics_dir / f"{split}.json")["best_epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for prefix, label in (("train", "Training"), ("val", "Validation")):
            axes[0].plot(history.epoch, history[prefix + "_mse_normalized"], label=label)
            axes[1].plot(history.epoch, history[prefix + "_patient_macro_mae_mV"], label=label)
        for ax in axes:
            ax.axvline(selected, color="black", ls="--", label=f"Selected epoch {selected}")
            ax.set_xlabel("Actual completed epoch")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Normalized MSE")
        axes[1].set_ylabel("Patient-macro MAE (mV)")
        save(
            fig,
            out / "learning_curves.png",
            "The curves show completed training epochs and the validation-selected checkpoint.",
            "A lower training loss alone does not demonstrate better held-out forecasts; selection uses validation patient-macro MAE.",
        )
    result = read_json(metrics_dir / f"{split}.json")
    with np.load(metrics_dir / f"{split}_predictions.npz") as archive:
        ids, target, prediction = (archive[key] for key in ("ecg_id", "target", "prediction"))
    wave = np.load(Path(cache) / f"waves_{split}.npy", mmap_mode="r")
    h = cfg["data"]["history"]
    for kind, ecg_id in (
        ("pre_specified", result["pre_specified_ecg_id"]),
        ("median_error", result["median_error_ecg_id"]),
    ):
        index = int(np.flatnonzero(ids == ecg_id)[0])
        fig, axes = plt.subplots(4, 3, figsize=(14, 11))
        for lead, ax in enumerate(axes.flat):
            ax.plot(np.arange(h) / 100, wave[index, :h, lead], label="Observed", lw=0.65)
            future = np.arange(h, 1000) / 100
            ax.plot(future, target[index, :, lead], label="Actual future", lw=0.8)
            ax.plot(future, prediction[index, :, lead], label="Predicted future", lw=0.8)
            ax.axvline(h / 100, color="black", ls="--", lw=0.7)
            ax.set(title=LEADS[lead], xlabel="Time (s)", ylabel="mV")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{split}: ECG {ecg_id}; {kind.replace('_', ' ')} example", y=1.01)
        save(
            fig,
            out / f"{split}_{kind}_12_leads.png",
            f"Observed history, raw future truth, and forecast are shown separately for each lead; this is the {kind.replace('_', ' ')} case.",
            "This example illustrates waveform error, not clinical validity or performance on every patient.",
        )
    lead_metrics = pd.read_csv(metrics_dir / f"{split}_leads.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    lead_metrics.set_index("lead")[["mae_mV", "rmse_mV"]].plot.bar(ax=ax)
    ax.set(ylabel="Error (mV)", xlabel="ECG lead")
    save(
        fig,
        out / f"{split}_per_lead_errors.png",
        "MAE and RMSE are shown in mV for each of the twelve leads.",
        "Differences show absolute voltage errors; low error does not prove accurate peak timing.",
    )
