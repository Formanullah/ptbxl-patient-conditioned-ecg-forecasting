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


def recording_figures(wave, sampling_rate, history):
    """Show the complete record, the forecast boundary and a sample-level zoom."""
    time = np.arange(len(wave)) / sampling_rate
    boundary = history / sampling_rate
    overview, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(time, wave[:, i], lw=0.7)
        ax.axvline(boundary, color="black", ls="--", lw=0.8)
        ax.set(title=LEADS[i], ylabel="mV")
        if i >= 9:
            ax.set_xlabel("Time (s)")
    overview.suptitle("All 12 leads; dashed line marks the end of the observed history")
    detail, (long_ax, short_ax) = plt.subplots(1, 2, figsize=(12, 3), layout="constrained")
    long_ax.plot(time, wave[:, 1], lw=0.8)
    long_ax.axvline(boundary, color="black", ls="--")
    short_ax.plot(time[:sampling_rate], wave[:sampling_rate, 1], ".-", lw=0.65, ms=3)
    long_ax.set(title="Lead II, complete recording", xlabel="Time (s)", ylabel="mV")
    short_ax.set(title="Lead II, first second (one dot per sample)", xlabel="Time (s)", ylabel="mV")
    return overview, detail


def qrs_index(wave, sampling_rate, start=0, stop=None):
    """Return the first detected Lead II QRS whose plotting window fits."""
    from wfdb.processing import xqrs_detect

    stop = len(wave) if stop is None else stop
    margin = round(0.2 * sampling_rate)
    try:
        peaks = xqrs_detect(wave[:, 1], fs=sampling_rate, verbose=False)
    except (ValueError, IndexError, FloatingPointError):
        return None
    return next((int(i) for i in peaks if start + margin <= i < stop - margin), None)


def preprocessing_figures(raw, filtered, sampling_rate):
    """Compare the two observed histories without extending filtering into the target."""
    time = np.arange(len(raw)) / sampling_rate
    overview, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(time, raw[:, i], label="P0 raw", lw=0.7)
        ax.plot(time, filtered[:, i], label="P1 causal", lw=0.7)
        ax.set(title=LEADS[i], ylabel="mV")
        if i >= 9:
            ax.set_xlabel("Observed time (s)")
    axes.flat[0].legend(fontsize=7)
    detail, (qrs, psd) = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    index = qrs_index(raw, sampling_rate)
    half = round(0.2 * sampling_rate)
    zoom = slice(index - half, index + half + 1) if index is not None else slice(0, sampling_rate)
    for values, label in ((raw, "P0 raw"), (filtered, "P1 causal")):
        qrs.plot(time[zoom], values[zoom, 1], label=label)
        freq, power = signal.welch(values[:, 1], fs=sampling_rate, nperseg=min(256, len(raw)))
        psd.semilogy(freq, np.maximum(power, 1e-15), label=label)
    qrs.set(
        title="Lead II detected QRS" if index is not None else "Lead II: no QRS located",
        xlabel="Observed time (s)",
        ylabel="mV",
    )
    psd.set(title="Welch PSD, observed Lead II", xlabel="Frequency (Hz)", ylabel="mV²/Hz")
    qrs.legend()
    psd.legend()
    return overview, detail


def matched_forecasts(cfg, cache, labels):
    """Plot the lowest common validation ECG ID for both selected checkpoints."""
    from evaluate import outputs, verify_evaluation

    series = {}
    for label, name in labels.items():
        folder = outputs(cfg, name)
        verify_evaluation(folder, "val")
        with np.load(folder / "val_predictions.npz") as archive:
            series[label] = {key: archive[key] for key in ("ecg_id", "target", "prediction")}
    ecg_id = min(set.intersection(*(set(run["ecg_id"]) for run in series.values())))
    metadata = pd.read_csv(Path(cache) / "metadata_val.csv")
    index = int(np.flatnonzero(metadata.ecg_id == ecg_id)[0])
    truth = np.load(Path(cache) / "waves_val.npy", mmap_mode="r")[index]
    h, hz = cfg["data"]["history"], cfg["data"]["sampling_rate"]
    predicted = {}
    for label, run in series.items():
        row = int(np.flatnonzero(run["ecg_id"] == ecg_id)[0])
        if not np.array_equal(truth[h:], run["target"][row]):
            raise ValueError("The forecast plots must share one identical raw target.")
        predicted[label] = run["prediction"][row]
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex=True, layout="constrained")
    time = np.arange(len(truth)) / hz
    for i, ax in enumerate(axes.flat):
        ax.plot(time[:h], truth[:h, i], lw=0.65, color="0.45", label="Observed")
        ax.plot(time[h:], truth[h:, i], lw=0.95, color="black", label="True future")
        for label, prediction in predicted.items():
            ax.plot(time[h:], prediction[:, i], lw=0.8, label=label)
        ax.axvline(h / hz, ls="--", color="black", lw=0.7)
        ax.set(title=LEADS[i], ylabel="mV")
        if i >= 9:
            ax.set_xlabel("Time (s)")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Validation ECG {ecg_id} – lowest common ECG ID")
    peak = qrs_index(truth, hz, start=h)
    zoom_fig = None
    if peak is not None:
        zoom_fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        local = slice(peak - round(0.2 * hz), peak + round(0.2 * hz) + 1)
        ax.plot(time[local], truth[local, 1], color="black", label="True future")
        for label, prediction in predicted.items():
            pred_slice = slice(local.start - h, local.stop - h)
            ax.plot(time[local], prediction[pred_slice, 1], label=label)
        ax.set(
            title=f"ECG {ecg_id}: first detected future QRS in Lead II",
            xlabel="Time (s)",
            ylabel="mV",
        )
        ax.legend()
    return int(ecg_id), fig, zoom_fig


def occlusion_heatmap(importance):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    extent = max(float(np.abs(importance.to_numpy()).max()), 1e-9)
    heat = ax.imshow(importance, cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
    ax.set(
        xticks=range(len(importance.columns)),
        xticklabels=importance.columns,
        yticks=range(len(importance.index)),
        yticklabels=importance.index,
        xlabel="Masked observed interval",
        ylabel="Masked ECG lead",
    )
    fig.colorbar(heat, ax=ax, label="Occluded MAE − original MAE (mV)")
    return fig
