"""Plot raw/causal-filtered observed ECG, removed components and signal spectra."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from config import LEADS, load_config, read_json, write_json
from evaluate import save_figure
from morphology import detect_peaks, peak_counts
from preprocessing import input_filter, metadata, read_wave


def diagnostics(cfg, record=None):
    cache = Path(cfg["data"]["cache"])
    manifest = read_json(cache / "manifest.json")
    meta = pd.read_csv(cache / "metadata_train.csv")
    record = int(meta.ecg_id.min()) if record is None else record
    if record not in set(meta.ecg_id):
        raise ValueError("Diagnostics example must be from selected training records")
    row = metadata(cfg["data"]["root"]).set_index("ecg_id").loc[record]
    raw = read_wave(row, cfg)[
        cfg["data"]["start"] : cfg["data"]["start"] + cfg["data"]["history"]
    ].T
    processed = input_filter(raw, cfg)
    out = (
        Path(cfg["paths"]["outputs"])
        / "preprocessing"
        / (cfg["preprocessing"]["filter"] + "_" + cfg["preprocessing"]["scaling"])
    )
    out.mkdir(parents=True, exist_ok=True)
    fs = cfg["data"]["sampling_rate"]
    t = np.arange(raw.shape[-1]) / fs
    for removed in (False, True):
        fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(
                t,
                raw[i] - processed[i] if removed else raw[i],
                label="Removed component" if removed else "Raw",
            )
            if not removed:
                ax.plot(t, processed[i], label="Processed", alpha=0.8)
            ax.set(ylabel=f"{LEADS[i]} (mV)")
        axes[0, 0].legend()
        fig.suptitle(f"Training ECG {record}; observed window only")
        save_figure(fig, out / ("removed_components.png" if removed else "raw_filtered_12lead.png"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, values in (("Raw", raw[1]), ("Processed", processed[1])):
        axes[0].plot(
            np.fft.rfftfreq(len(values), 1 / fs),
            np.abs(np.fft.rfft(values)) / len(values),
            label=name,
        )
        f, power = signal.welch(values, fs=fs, nperseg=min(256, len(values)))
        axes[1].semilogy(f, np.maximum(power, 1e-16), label=name)
    axes[0].set(xlabel="Hz", ylabel="FFT amplitude (mV)")
    axes[1].set(xlabel="Hz", ylabel="Welch PSD (mV squared/Hz)")
    for ax in axes:
        ax.legend()
    save_figure(fig, out / "fft_welch.png")
    spectra = [signal.stft(a, fs=fs, nperseg=min(100, len(a))) for a in (raw[1], processed[1])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, name, (f, times, z) in zip(
        axes, ("Raw lead II", "Processed lead II"), spectra, strict=True
    ):
        ax.pcolormesh(
            times, f, 20 * np.log10(np.maximum(np.abs(z), 1e-8)), shading="auto", vmin=-80, vmax=0
        )
        ax.set(title=name, xlabel="Seconds", ylabel="Hz")
    save_figure(fig, out / "stft.png")
    a, fail_a = detect_peaks(raw[1], fs)
    b, fail_b = detect_peaks(processed[1], fs)
    counts, pairs = peak_counts(a, b, round(0.1 * fs))
    center = a[len(a) // 2] / fs if len(a) else len(t) / fs / 2
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, raw[1], label="Raw")
    ax.plot(t, processed[1], label="Processed")
    ax.set(
        xlim=(max(0, center - 0.3), min(t[-1], center + 0.3)),
        xlabel="Seconds",
        ylabel="mV",
        title="Observed QRS zoom; causal filters may shift/distort peaks",
    )
    ax.legend()
    save_figure(fig, out / "qrs_before_after.png")
    summary = {
        "ecg_id": record,
        "fingerprint": manifest["fingerprint"],
        "split": "train",
        "filter": cfg["preprocessing"],
        "qrs_detector_comparison": counts,
        "detector_failures": int(fail_a) + int(fail_b),
        "matched_timing_mae_ms": float(np.mean([abs(x - y) * 1000 / fs for x, y in pairs]))
        if pairs
        else None,
        "removed_rms_mV": float(np.sqrt(np.mean((raw - processed) ** 2))),
        "note": "Removed components are not verified noise; causal phase delay is retained.",
    }
    write_json(out / "diagnostics.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--record", type=int)
    args = parser.parse_args()
    print(diagnostics(load_config(args.config, args.experiment), args.record))
