"""Explainability for the presentation (descriptive; fold 9, out-of-sample models).

Uses the 6 fold-9 models of the gated CNN-LSTM saved in Phase 12 (hidden 256; trained
on folds 1-8 only). Outputs:
- input-gradient saliency of the first 500 ms of the lead-II forecast over the history
  (baseline vs residual channels), and its concentration around history R peaks;
- decomposition of each forecast into the gated `sn4` term and the learned correction;
- one example record with the decomposition.
Fold 10 is not used.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "research")
import phase0_feasibility as p0  # noqa: E402

from ptbxl_forecast.config import load_config  # noqa: E402
from ptbxl_forecast.data import load_dataset  # noqa: E402
from ptbxl_forecast.models import GatedBaselineCNNLSTM  # noqa: E402
from ptbxl_forecast.preprocessing import pack_baseline  # noqa: E402

H, F, FS = 700, 290, 100
SEEDS = (42, 123, 999, 7, 2024, 31337)
MODELS = Path(".cache/data/phase12/fold9_models")
OUT = Path("reports/presentation/figures")
N = 400


def load(seed):
    pipe = pickle.load((MODELS / f"final_k9_s{seed}.pipe.pkl").open("rb"))
    m = GatedBaselineCNNLSTM(pipe.in_channels, 12, F, 64, 256, 2)
    m.load_state_dict(torch.load(MODELS / f"final_k9_s{seed}.pt"))
    return m.eval(), pipe


def parts(m, x):
    """gate, sn4 channels, correction for model input x (n, 700, 36)."""
    b = m.body
    z = b.stem(x[..., : m.in_channels].transpose(1, 2)).transpose(1, 2)
    h = b.lstm(z)[0][:, -1]
    g = torch.sigmoid(m.gate(h)).view(-1, F, 12)
    return g, x[:, -F:, m.in_channels :], b.head(h).view(-1, F, 12)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    rows9 = np.flatnonzero(meta.strat_fold == 9)
    pick = np.sort(np.random.default_rng(0).choice(rows9, N, replace=False))
    sn4 = np.load(".cache/data/phase10/sn4_dev.npy")
    rec = sig[pick].astype(np.float64)
    packed = pack_baseline(rec, sn4[pick].astype(np.float64), H)[:, :H]
    sal, gate_sn4, corr = [], [], []
    for s in SEEDS:
        m, pipe = load(s)
        x = torch.from_numpy(pipe.inputs(packed).astype(np.float32)).requires_grad_(True)
        g, b, c = parts(m, x)
        (g * b + c)[:, :50, 1].sum().backward()
        sal.append(x.grad.abs().numpy())
        gate_sn4.append((g * b).detach().numpy())
        corr.append(c.detach().numpy())
    sal = np.mean(sal, 0)  # (N, 700, 36): 12 baseline, 12 residual, 12 sn4
    gs, co = np.mean(gate_sn4, 0), np.mean(corr, 0)
    base_t, resid_t = sal[:, :, :12].sum(2).mean(0), sal[:, :, 12:24].sum(2).mean(0)

    # concentration of residual-channel saliency around history R peaks (lead II)
    near, far = [], []
    for i in range(N):
        pk = p0.rpeaks(rec[i, :H, 1])
        mask = np.zeros(H, bool)
        for p in pk:
            mask[max(0, p - 5) : p + 6] = True  # +-50 ms
        r = sal[i, :H, 12:24].sum(1)
        if mask.any() and (~mask).any():
            near.append(r[mask].mean())
            far.append(r[~mask].mean())
    last1 = resid_t[-FS:].sum() / resid_t.sum()
    res = {
        "records": N,
        "models": "6 fold-9 models (hidden 256), trained on folds 1-8",
        "share_resid_saliency_last_1s": float(last1),
        "share_saliency_baseline_vs_residual": float(base_t.sum() / (base_t.sum() + resid_t.sum())),
        "qrs_vs_rest_saliency_ratio_median": float(np.median(np.array(near) / np.array(far))),
        "mean_abs_gated_sn4_mV": float(np.abs(gs).mean()),
        "mean_abs_correction_mV": float(np.abs(co).mean()),
    }
    json.dump(res, (OUT / "explain.json").open("w"), indent=2)
    print(res)

    t = (np.arange(H) - H) / FS
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(t, resid_t / resid_t.max(), color="#c0392b", lw=1, label="residual (ECG) channels")
    ax[0].plot(t, base_t / resid_t.max(), color="#2c3e50", lw=1, label="baseline channels")
    ax[0].set(xlabel="time before forecast start (s)", ylabel="mean |gradient| (normalised)",
              title="Saliency of the first 500 ms of the lead-II forecast")  # fmt: skip
    ax[0].legend(frameon=False)
    i = int(np.argmax([np.ptp(rec[k, H - 300 : H, 1]) for k in range(N)]))
    ax[1].plot(t[-300:], rec[i, H - 300 : H, 1], color="k", lw=1, label="history, lead II")
    a2 = ax[1].twinx()
    a2.fill_between(t[-300:], sal[i, H - 300 :, 12:24].sum(1), color="#c0392b", alpha=0.35, lw=0)
    a2.set_yticks([])
    ax[1].set(xlabel="time before forecast start (s)", ylabel="mV", title="One record: saliency (shaded) over the last 3 s")
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "explain_saliency.png", dpi=150)

    # forecast decomposition for one periodic record
    acf = np.load(".cache/data/phase12/acf_peak.npy")[pick]
    j = int(np.argsort(acf)[-25])
    tf = np.arange(F) / FS
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(np.arange(-200, 0) / FS, rec[j, H - 200 : H, 1], color="0.5", lw=1, label="history")
    ax.plot(tf, rec[j, H : H + F, 1], color="k", lw=1.2, label="truth")
    ax.plot(tf, sn4[pick[j], :, 1], color="#27ae60", lw=0.9, ls=":", label="sn4")
    ax.plot(tf, gs[j, :, 1], color="#e67e22", lw=1, label="gate × sn4")
    ax.plot(tf, co[j, :, 1], color="#2980b9", lw=1, label="correction")
    ax.plot(tf, gs[j, :, 1] + co[j, :, 1], color="#c0392b", lw=1.6, label="forecast = gate×sn4 + correction")
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    ax.set(xlabel="time (s, 0 = forecast start)", ylabel="mV (lead II)",
           title=f"How the gated model builds a forecast (fold-9 record, history periodicity {acf[j]:.2f})")  # fmt: skip
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "explain_decomposition.png", dpi=150)


if __name__ == "__main__":
    main()
