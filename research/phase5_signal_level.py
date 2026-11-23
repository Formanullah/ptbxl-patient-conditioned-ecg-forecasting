"""Phase 5, signal level: IEC 60601-2-25 impulse test (S1) and synthetic-noise distortion /
morphology / runtime (S2) for every filter candidate. No learner involved.

    python research/phase5_signal_level.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sps  # noqa: E402

from ptbxl_forecast import metrics as M  # noqa: E402
from ptbxl_forecast.config import load_config  # noqa: E402
from ptbxl_forecast.data import load_dataset  # noqa: E402
from ptbxl_forecast.preprocessing import FILTERS  # noqa: E402

OUT = Path("reports/phase5_preprocessing")
H, FS, SEED = 700, 100, 20261122
SEQUENCE_FILTERS = [f for f in FILTERS if f not in ("none", "decompose_0.67")]


GROUP_DELAY = {"causal_fir_lp_40": 15}  # linear-phase FIR: (numtaps - 1) / 2 samples


def impulse_test(name: str) -> dict:
    """3 mV x 100 ms pulse on a flat 7 s history. Causal filters: pulse at 1 s, measured
    after it (excluding a linear-phase FIR's delayed pulse and finite support). History-only zero-phase /
    nonlinear filters: pulse at 3.5 s, measured on both sides."""
    causal = name.startswith("causal")
    start = 100 if causal else 350
    x = np.zeros((1, H, 12))
    x[0, start : start + 10] = 3.0
    y = FILTERS[name](x)[0, :, 0]
    end = start + 10 + 2 * GROUP_DELAY.get(name, 0)  # FIR: delayed pulse plus its finite support
    outside = np.ones(H, bool)
    outside[start:end] = False
    if causal:
        outside[:start] = False
    disp = float(np.abs(y[outside]).max())
    region = np.flatnonzero(outside)
    region = region[(region < start - 10) | (region > end + 10)]  # 100 ms clear of the pulse
    slope = (
        np.abs(y[region[10:]] - y[region[:-10]]) * FS / 10 if len(region) > 10 else np.array([0.0])
    )
    return {"displacement_mV": disp, "max_slope_mV_per_s": float(slope.max()),
            "pass_S1": bool(disp <= 0.1 and slope.max() <= 0.3), "trace": y}  # fmt: skip


def synthetic_noise(x: np.ndarray, rng) -> np.ndarray:
    n, t_len, leads = x.shape
    t = np.arange(t_len) / FS
    bw = np.zeros_like(x)
    for _ in range(2):
        f = rng.uniform(0.1, 0.4, (n, 1, 1))
        phase = rng.uniform(0, 2 * np.pi, (n, 1, leads))
        bw += 0.25 * np.sin(2 * np.pi * f * t[None, :, None] + phase)
    emg = sps.sosfiltfilt(
        sps.butter(4, (20, 45), "bandpass", fs=FS, output="sos"), rng.normal(size=x.shape), axis=1
    )
    emg *= 0.05 / emg.std()
    return x + bw + emg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "train", cache_dir=".cache/data")
    q = pd.read_csv(".cache/data/phase3_per_record.csv", index_col=0)
    clean_ids = q.index[(q.kSQI_med >= 15) & (q.bw_pp_med <= 0.1)].intersection(meta.index)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(clean_ids, 500, replace=False))
    rows = meta.index.get_indexer(pick)
    clean = sig[rows, :H].astype(np.float64)
    noisy = synthetic_noise(clean, rng)
    peaks = [M.detect_beats(np.zeros((0, 12)), c) for c in clean]  # R peaks of the clean history

    results, traces = {}, {}
    for name in ["none", *SEQUENCE_FILTERS]:
        t0 = time.perf_counter()
        out = FILTERS[name](noisy)
        ms = 1000 * (time.perf_counter() - t0) / len(noisy)
        diff = (out - out.mean(1, keepdims=True)) - (clean - clean.mean(1, keepdims=True))
        rmse = np.sqrt((diff**2).mean(axis=(1, 2)))
        corr, st = [], []
        for i in range(len(clean)):
            tc, to = M.template(clean[i], peaks[i]), M.template(out[i], peaks[i])
            if tc is None or to is None:
                continue
            a, b = tc - tc.mean(0), to - to.mean(0)
            corr.append(np.mean((a * b).sum(0) / np.sqrt((a**2).sum(0) * (b**2).sum(0) + 1e-12)))
            st.append(np.abs(M.amplitudes(to)[1] - M.amplitudes(tc)[1]).mean())
        imp = (
            impulse_test(name)
            if name != "none"
            else {"displacement_mV": 0.0, "max_slope_mV_per_s": 0.0, "pass_S1": True, "trace": None}
        )
        traces[name] = imp.pop("trace")
        results[name] = {
            **imp,
            "rmse_vs_clean_mV": float(np.median(rmse)),
            "template_corr_median": float(np.median(corr)),
            "st_error_median_mV": float(np.median(st)),
            "ms_per_record": ms,
        }
        results[name]["pass_S2"] = bool(
            results[name]["template_corr_median"] >= 0.99
            and results[name]["st_error_median_mV"] <= 0.05
        )
        print(
            name,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[name].items()},
            flush=True,
        )
    table = pd.DataFrame(results).T
    table.to_csv(OUT / "table_signal_level.csv", float_format="%.5f")
    (OUT / "signal_level.json").write_text(
        json.dumps({"records": len(pick), "results": results}, indent=2) + "\n"
    )

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for name, tr in traces.items():
        if tr is None:
            continue
        t = np.arange(H) / FS - (1.0 if name.startswith("causal") else 3.5)
        ax[0].plot(t, tr, lw=1, label=name)
    ax[0].axhline(0.1, color="k", ls="--", lw=0.8)
    ax[0].axhline(-0.1, color="k", ls="--", lw=0.8)
    ax[0].set(xlim=(-1, 3), ylim=(-0.5, 0.5), xlabel="time from pulse onset (s)", ylabel="mV",
              title="IEC 60601-2-25 impulse test (3 mV x 100 ms); limit +-0.1 mV")  # fmt: skip
    ax[0].legend(fontsize=7)
    i = 0
    tt = np.arange(H) / FS
    ax[1].plot(tt, clean[i, :, 1], color="k", lw=1.2, label="clean")
    ax[1].plot(tt, noisy[i, :, 1], color="#bbbbbb", lw=0.8, label="noisy input")
    for name in (
        "causal_bp_0.5_40",
        "zero_phase_bp_0.5_40_history",
        "median_baseline",
        "wavelet_db4",
    ):
        ax[1].plot(tt, FILTERS[name](noisy[i : i + 1])[0, :, 1], lw=0.9, label=name)
    ax[1].set(
        xlim=(4, 7),
        xlabel="time (s)",
        ylabel="mV",
        title="Synthetic noise test (lead II, last 3 s of history)",
    )
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_signal_level.png", dpi=140)


if __name__ == "__main__":
    main()
