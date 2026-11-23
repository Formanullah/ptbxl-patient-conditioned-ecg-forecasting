"""Phase 3 (post hoc, decision D0.9): is the end-of-record artifact in the official 100 Hz
signals present in the 500 Hz originals, or introduced by the 500 -> 100 Hz conversion?

Uses training-fold records (folds 1-8) whose 500 Hz files were downloaded with
``scripts/download_ptbxl.py --include records500/00000/ --include records500/01000/``.

    python research/phase3_edge_artifact.py --data PTB-XL --out reports/phase3_quality
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wfdb  # noqa: E402
from scipy import signal as sps  # noqa: E402

from ptbxl_forecast.data import load_metadata  # noqa: E402

ROOT = Path("PTB-XL")


def load_pair(row: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    lr, hr = row
    return wfdb.rdsamp(str(ROOT / lr))[0], wfdb.rdsamp(str(ROOT / hr))[0]


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PTB-XL"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase3_quality"))
    args = ap.parse_args()
    ROOT = args.data
    meta = load_metadata(args.data, range(1, 9))  # training folds only
    have = meta.filename_hr.apply(lambda f: (args.data / f"{f}.dat").exists())
    meta = meta[have]
    with Pool(32) as pool:
        pairs = pool.map(load_pair, list(zip(meta.filename_lr, meta.filename_hr, strict=True)))
    x100 = np.stack([p[0] for p in pairs])  # (n, 1000, 12)
    x500 = np.stack([p[1] for p in pairs])  # (n, 5000, 12)

    # Alignment: which 500 Hz phase does the official 100 Hz signal follow?
    interior = slice(100, 900)
    phase_err = [
        float(np.abs(x100[:, interior] - x500[:, k::5][:, interior]).mean()) for k in range(5)
    ]
    refs = {
        "subsample x500[::5]": x500[:, ::5],
        "resample_poly (line padding)": sps.resample_poly(x500, 1, 5, axis=1, padtype="line"),
        "resample_poly (zero padding)": sps.resample_poly(x500, 1, 5, axis=1),
    }
    prof = {name: np.abs(x100 - r).mean(axis=(0, 2)) for name, r in refs.items()}
    dev = lambda x: np.abs(x - np.median(x, axis=1, keepdims=True)).mean(axis=(0, 2))  # noqa: E731
    amp = {"official 100 Hz": dev(x100), "500 Hz subsampled": dev(x500[:, ::5]),
           "500 Hz (all samples)": dev(x500)[::5]}  # fmt: skip

    base = prof["resample_poly (line padding)"]
    level = float(np.median(base[100:900]))
    S = {
        "records": len(meta),
        "phase_error_mV_by_offset_k": phase_err,
        "official_vs_reference_interior_mae_mV": {
            k: float(v[100:900].mean()) for k, v in prof.items()
        },
        "official_vs_line_padded_last_samples_mV": base[-12:].round(4).tolist(),
        "official_vs_line_padded_first_samples_mV": base[:12].round(4).tolist(),
        "end_samples_over_3x_interior": int(np.sum(base[-50:] > 3 * level)),
        "start_samples_over_3x_interior": int(np.sum(base[:50] > 3 * level)),
        "amplitude_last_50ms_over_interior": {
            k: float(v[-5:].mean() / np.median(v[100:900])) for k, v in amp.items()
        },
        "amplitude_first_50ms_over_interior": {
            k: float(v[:5].mean() / np.median(v[100:900])) for k, v in amp.items()
        },
    }
    (args.out / "edge_artifact.json").write_text(json.dumps(S, indent=2) + "\n")
    print(json.dumps(S, indent=2))

    t = np.arange(1000) / 100
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for k, v in prof.items():
        ax[0].plot(t, v, lw=1, label=k)
    ax[0].set(yscale="log", xlabel="time (s)", ylabel="mean |official 100 Hz - reference| (mV)",
              title=f"Official 100 Hz vs references from 500 Hz (n={len(meta)})")  # fmt: skip
    ax[0].legend(fontsize=8)
    for k, v in amp.items():
        ax[1].plot(t, v, lw=1, label=k)
    ax[1].set(
        xlabel="time (s)",
        ylabel="mean |x - record median| (mV)",
        title="Amplitude profile along the record",
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "fig6_edge_artifact.png", dpi=140)


if __name__ == "__main__":
    main()
