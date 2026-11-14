"""Train and evaluate two SOTA DL forecasting models (PatchTST and TFT) and generate
empirical Interpretability/Explainability (XAI) figures and metrics for the LaTeX report.

Reads the existing prepared 1,500/300/100 cohort cache (data/custom_100hz_5s_3s_ee477b14/6ba08ad4)
in read-only mode and writes outputs exclusively to outputs/sota_and_xai/.
"""

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURES, LEADS, read_json, write_json
from metrics import bootstrap_pair, metric_summary, physical_arrays, subgroup_metrics
from models import build_model
from morphology import qrs_metrics

CACHE_DIR = Path("data/custom_100hz_5s_3s_ee477b14/6ba08ad4")
EXP_DIR = Path("experiments/custom_100hz_5s_3s_ee477b14")
OUT_DIR = Path("outputs/sota_and_xai")


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from Temporal Fusion Transformer (Lim et al., 2021)."""

    def __init__(self, in_dim, hidden_dim, out_dim=None, context_dim=None, dropout=0.2):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, out_dim * 2)  # GLU gate + value
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        val, gate = h.chunk(2, dim=-1)
        return self.norm(residual + val * torch.sigmoid(gate))


class StaticVariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) computing per-feature Softmax importance weights."""

    def __init__(self, num_features, d_model, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.feature_embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.weight_grn = GatedResidualNetwork(
            num_features * d_model, d_model, num_features, dropout=dropout
        )
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout) for _ in range(num_features)]
        )

    def forward(self, static_x):
        # static_x: [B, F]
        embedded = [
            proj(static_x[:, i : i + 1]) for i, proj in enumerate(self.feature_embeds)
        ]  # list of [B, d_model]
        flat = torch.cat(embedded, dim=-1)  # [B, F * d_model]
        weights = F.softmax(self.weight_grn(flat), dim=-1)  # [B, F]
        transformed = torch.stack(
            [grn(e) for grn, e in zip(self.feature_grns, embedded, strict=True)], dim=1
        )  # [B, F, d_model]
        combined = (weights.unsqueeze(-1) * transformed).sum(dim=1)  # [B, d_model]
        return combined, weights


class PatchTSTForecaster(nn.Module):
    """Patient-Conditioned PatchTST (Nie et al., ICLR 2023) for 12-lead ECG forecasting.

    Segments the 500-sample observed sequence into overlapping subseries patches (P=20, S=10),
    embeds each patch into d_model, prepends a static patient-context token, applies
    Multi-Head Transformer Self-Attention with pre-norm residual connections, and maps
    the contextualized patch representations to the 300-sample 12-lead horizon.
    """

    def __init__(
        self,
        static_dim=10,
        history=500,
        horizon=300,
        patch_len=20,
        stride=10,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        head_width=128,
    ):
        super().__init__()
        self.history = history
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (history - patch_len) // stride + 1  # 49 patches for 500, P=20, S=10
        self.patch_proj = nn.Linear(12 * patch_len, d_model)
        self.static_token = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model, nhead, dropout=dropout, batch_first=True
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 2, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 9, head_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 12 * horizon),
        )

    def forward(self, x, static, return_attention=False):
        # x: [B, 12, T]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, 12, N_p, P]
        b, c, n_p, p = patches.shape
        patches = patches.permute(0, 2, 1, 3).reshape(b, n_p, c * p)  # [B, N_p, 12*P]
        tokens = self.patch_proj(patches)  # [B, N_p, d_model]
        s_tok = self.static_token(static).unsqueeze(1)  # [B, 1, d_model]
        seq = self.dropout(torch.cat([s_tok, tokens], dim=1) + self.pos_embed)
        last_attn = None
        for layer in self.layers:
            normed = layer["norm1"](seq)
            attn_out, attn_weights = layer["attn"](
                normed, normed, normed, need_weights=return_attention, average_attn_weights=True
            )
            seq = seq + attn_out
            seq = seq + layer["ffn"](layer["norm2"](seq))
            last_attn = attn_weights
        seq = self.final_norm(seq)
        # Preserve cardiac phase at forecast boundary using static token + terminal 8 patch tokens
        pooled = torch.cat([seq[:, 0], seq[:, -8:].reshape(b, -1)], dim=-1)  # [B, 9*d_model]
        out = self.head(pooled).reshape(b, 12, self.horizon)
        if return_attention:
            return out, last_attn
        return out


class TFTForecaster(nn.Module):
    """Patient-Conditioned Temporal Fusion Transformer (Lim et al., 2021; Butt et al., 2025).

    Combines a Static Variable Selection Network (VSN), Gated Residual Networks (GRN) for
    static context conditioning (initializing LSTM hidden/cell states and enriching temporal
    features), local causal strided convolution + LSTM sequence encoding, and Interpretable
    Multi-Head Temporal Self-Attention.
    """

    def __init__(
        self,
        static_dim=10,
        horizon=300,
        d_model=96,
        nhead=4,
        dropout=0.2,
        downsample=5,
        head_width=128,
    ):
        super().__init__()
        self.horizon = horizon
        self.downsample = downsample
        self.vsn = StaticVariableSelectionNetwork(static_dim, d_model, dropout=dropout)
        self.ctx_init_h = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.ctx_init_c = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.ctx_enrich = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(12, d_model, kernel_size=7, padding=3),
            nn.ELU(),
            nn.AvgPool1d(downsample),  # 500 -> 100 steps (50 ms resolution)
        )
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.lstm_gate = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.enrichment_grn = GatedResidualNetwork(
            d_model, d_model, d_model, context_dim=d_model, dropout=dropout
        )
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.pos_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model * 8, head_width),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 12 * horizon),
        )

    def forward(self, x, static, return_explanations=False):
        # x: [B, 12, 500], static: [B, 10]
        static_vec, var_weights = self.vsn(static)  # [B, d_model], [B, 10]
        h0 = self.ctx_init_h(static_vec).unsqueeze(0)  # [1, B, d_model]
        c0 = self.ctx_init_c(static_vec).unsqueeze(0)  # [1, B, d_model]
        c_e = self.ctx_enrich(static_vec).unsqueeze(1)  # [B, 1, d_model]

        local_feat = self.temporal_conv(x).transpose(1, 2)  # [B, 100, d_model]
        lstm_out, _ = self.lstm(local_feat, (h0, c0))
        temporal = self.lstm_gate(local_feat + lstm_out)
        enriched = self.enrichment_grn(temporal, c_e.expand(-1, temporal.size(1), -1))
        attn_out, attn_weights = self.attn(
            enriched,
            enriched,
            enriched,
            need_weights=return_explanations,
            average_attn_weights=True,
        )
        attended = self.attn_norm(enriched + attn_out)
        refined = self.pos_grn(attended)
        summary = refined[:, -8:].reshape(x.size(0), -1)  # [B, 8*d_model]
        out = self.head(summary).reshape(x.size(0), 12, self.horizon)
        if return_explanations:
            return out, var_weights, attn_weights
        return out


def load_cohort_arrays():
    manifest = read_json(CACHE_DIR / "manifest.json")
    feature_names = manifest["static_statistics"]["features"]
    selected = FEATURES[:10]
    idx = [feature_names.index(name) for name in selected]

    data = {}
    for split in ("train", "val"):
        x = np.load(CACHE_DIR / f"X_{split}.npy", mmap_mode="r")
        y = np.load(CACHE_DIR / f"y_{split}.npy", mmap_mode="r")
        s = np.load(CACHE_DIR / f"static_{split}.npy", mmap_mode="r")[:, idx]
        meta = pd.read_csv(CACHE_DIR / f"metadata_{split}.csv")
        data[split] = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(y, dtype=np.float32)),
            "s": torch.from_numpy(np.asarray(s, dtype=np.float32)),
            "meta": meta,
            "patients": meta.patient_id.to_numpy(dtype=np.int64),
        }
    data["raw_X_val"] = np.load(CACHE_DIR / "raw_X_val.npy")
    return manifest, selected, data


def train_and_evaluate_sota(name, model, data, manifest, device, epochs=15, lr=0.0005, batch_size=128):
    seed_all(42)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    train_ds = TensorDataset(
        data["train"]["x"],
        data["train"]["y"],
        data["train"]["s"],
        torch.from_numpy(data["train"]["patients"]),
    )
    val_ds = TensorDataset(
        data["val"]["x"],
        data["val"]["y"],
        data["val"]["s"],
        torch.from_numpy(data["val"]["patients"]),
    )
    gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scale = torch.tensor(manifest["std_mV"], device=device, dtype=torch.float32).reshape(1, 12, 1)

    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_mse_sum, tr_n = 0.0, 0
        tr_records = []
        for bx, by, bs, bp in train_loader:
            bx, by, bs = bx.to(device), by.to(device), bs.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx, bs)
            loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_mse_sum += loss.item() * len(bx)
            tr_n += len(bx)
            err = (pred.detach() - by) * scale
            mae = err.abs().mean(dim=(1, 2)).cpu().numpy()
            tr_records.extend(zip(bp.tolist(), mae.tolist(), strict=True))

        model.eval()
        va_mse_sum, va_n = 0.0, 0
        va_records = []
        with torch.inference_mode():
            for bx, by, bs, bp in val_loader:
                bx, by, bs = bx.to(device), by.to(device), bs.to(device)
                pred = model(bx, bs)
                loss = F.mse_loss(pred, by)
                va_mse_sum += loss.item() * len(bx)
                va_n += len(bx)
                err = (pred - by) * scale
                mae = err.abs().mean(dim=(1, 2)).cpu().numpy()
                va_records.extend(zip(bp.tolist(), mae.tolist(), strict=True))

        tr_df = pd.DataFrame(tr_records, columns=["patient", "mae"])
        va_df = pd.DataFrame(va_records, columns=["patient", "mae"])
        tr_pmae = float(tr_df.groupby("patient").mae.mean().mean())
        va_pmae = float(va_df.groupby("patient").mae.mean().mean())
        scheduler.step(va_pmae)

        if va_pmae < best_mae:
            best_mae = va_pmae
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": ep,
                "train_mse": tr_mse_sum / tr_n,
                "val_mse": va_mse_sum / va_n,
                "train_patient_macro_mae": tr_pmae,
                "val_patient_macro_mae": va_pmae,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - ep_t0,
            }
        )
        print(
            f"[{name}] Epoch {ep:02d}/{epochs:02d} | train_pMAE={tr_pmae:.6f} | val_pMAE={va_pmae:.6f} | best={best_mae:.6f} (ep {best_epoch})",
            flush=True,
        )

    runtime_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.inference_mode():
        for bx, _, bs, _ in val_loader:
            preds.append(model(bx.to(device), bs.to(device)).cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)
    true_norm = data["val"]["y"].numpy()
    pred_mv = physical_arrays(pred_norm, manifest, layout="channels_first")
    true_mv = physical_arrays(true_norm, manifest, layout="channels_first")

    summary = metric_summary(pred_mv, true_mv, data["val"]["patients"])
    qrs, _, _, _ = qrs_metrics(pred_mv, true_mv, data["raw_X_val"], 100, 100)
    summary.update(qrs)
    summary.update(
        run=name,
        parameters=params,
        best_epoch=best_epoch,
        epochs=epochs,
        runtime_seconds=runtime_s,
        final_val_patient_macro_mae=history[-1]["val_patient_macro_mae"],
    )

    per_rec = data["val"]["meta"].copy()
    err = pred_mv.astype(float) - true_mv.astype(float)
    per_rec["mae_mV"] = np.abs(err).mean(axis=(1, 2))
    per_rec["mse_mV2"] = (err**2).mean(axis=(1, 2))

    return model, summary, pd.DataFrame(history), pred_mv, true_mv, per_rec


def compute_integrated_gradients(model, x, y, static, steps=20):
    """Compute Integrated Gradients (Sundararajan et al., 2017) attribution for forecast MAE."""
    model.eval()
    baseline = torch.zeros_like(x)
    scaled_inputs = [baseline + (float(k) / steps) * (x - baseline) for k in range(1, steps + 1)]
    grads = []
    for xt in scaled_inputs:
        xt = xt.clone().detach().requires_grad_(True)
        out = model(xt, static)
        score = F.l1_loss(out, y, reduction="sum")
        score.backward()
        grads.append(xt.grad.detach())
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    ig = ((x - baseline) * avg_grad).abs()
    return ig.detach().cpu().numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running SOTA & XAI study on device={device}", flush=True)

    manifest, selected_features, data = load_cohort_arrays()

    # 1. Train & evaluate the two SOTA models
    patchtst = PatchTSTForecaster(
        static_dim=len(selected_features),
        history=500,
        horizon=300,
        patch_len=20,
        stride=10,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        head_width=128,
    )
    patchtst, ptst_summary, ptst_hist, ptst_pred, true_mv, ptst_rec = train_and_evaluate_sota(
        "patchtst_patient_history", patchtst, data, manifest, device, epochs=15, lr=0.0005
    )

    tft = TFTForecaster(
        static_dim=len(selected_features),
        horizon=300,
        d_model=96,
        nhead=4,
        dropout=0.2,
        downsample=5,
        head_width=128,
    )
    tft, tft_summary, tft_hist, tft_pred, _, tft_rec = train_and_evaluate_sota(
        "tft_patient_history", tft, data, manifest, device, epochs=15, lr=0.0005
    )

    # Load existing LSTM and CNN-LSTM checkpoints & predictions from the same 1,500/300/100 cohort
    cnn_npz = np.load(EXP_DIR / "cnn_lstm_patient_history" / "val_predictions.npz")
    lstm_npz = np.load(EXP_DIR / "lstm_patient_history" / "val_predictions.npz")
    cnn_metrics = read_json(EXP_DIR / "cnn_lstm_patient_history" / "val_metrics.json")
    lstm_metrics = read_json(EXP_DIR / "lstm_patient_history" / "val_metrics.json")
    cnn_rec = pd.read_csv(EXP_DIR / "cnn_lstm_patient_history" / "val_per_record_metrics.csv")
    lstm_rec = pd.read_csv(EXP_DIR / "lstm_patient_history" / "val_per_record_metrics.csv")

    # Paired bootstrap comparisons between SOTA models and existing models
    boot_rows = []
    for name_a, rec_a, name_b, rec_b in [
        ("lstm_patient_history", lstm_rec, "patchtst_patient_history", ptst_rec),
        ("cnn_lstm_patient_history", cnn_rec, "patchtst_patient_history", ptst_rec),
        ("lstm_patient_history", lstm_rec, "tft_patient_history", tft_rec),
        ("cnn_lstm_patient_history", cnn_rec, "tft_patient_history", tft_rec),
        ("patchtst_patient_history", ptst_rec, "tft_patient_history", tft_rec),
    ]:
        res = bootstrap_pair(rec_a, rec_b, samples=1000, seed=42)
        boot_rows.append({"A": name_a, "B": name_b, **res})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "sota_paired_bootstrap.csv", index=False)

    # Save SOTA summaries
    write_json(
        OUT_DIR / "sota_metrics.json",
        {
            "patchtst_patient_history": ptst_summary,
            "tft_patient_history": tft_summary,
            "paired_bootstrap": boot_rows,
        },
    )
    ptst_hist.to_csv(OUT_DIR / "patchtst_history.csv", index=False)
    tft_hist.to_csv(OUT_DIR / "tft_history.csv", index=False)

    # --- Plot 1: SOTA Learning Curves & Model Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(
        ptst_hist.epoch, ptst_hist.train_patient_macro_mae, "--", color="#1f77b4", label="PatchTST Train"
    )
    axes[0].plot(
        ptst_hist.epoch, ptst_hist.val_patient_macro_mae, "-", lw=2, color="#1f77b4", label="PatchTST Val"
    )
    axes[0].plot(
        tft_hist.epoch, tft_hist.train_patient_macro_mae, "--", color="#d62728", label="TFT Train"
    )
    axes[0].plot(
        tft_hist.epoch, tft_hist.val_patient_macro_mae, "-", lw=2, color="#d62728", label="TFT Val"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Patient-Macro MAE (mV)")
    axes[0].set_title("(a) SOTA Deep Learning Training & Validation Curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8.5)

    models_cmp = [
        ("LSTM\n(History)", lstm_metrics["patient_macro_mae"], lstm_metrics["rmse"], lstm_metrics["pearson_mean"]),
        ("CNN-LSTM\n(History)", cnn_metrics["patient_macro_mae"], cnn_metrics["rmse"], cnn_metrics["pearson_mean"]),
        ("PatchTST\n(History)", ptst_summary["patient_macro_mae"], ptst_summary["rmse"], ptst_summary["pearson_mean"]),
        ("TFT\n(History)", tft_summary["patient_macro_mae"], tft_summary["rmse"], tft_summary["pearson_mean"]),
    ]
    x_pos = np.arange(len(models_cmp))
    width = 0.26
    axes[1].bar(
        x_pos - width,
        [m[1] for m in models_cmp],
        width,
        label="Patient-Macro MAE (mV)",
        color="#2c7bb6",
    )
    axes[1].bar(
        x_pos,
        [m[2] for m in models_cmp],
        width,
        label="RMSE (mV)",
        color="#fdae61",
    )
    axes[1].bar(
        x_pos + width,
        [m[3] for m in models_cmp],
        width,
        label="Mean Lead Pearson r",
        color="#2ca02c",
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([m[0] for m in models_cmp])
    axes[1].set_ylabel("Metric Value")
    axes[1].set_title("(b) Validation Performance Across 4 DL Architectures (N=300 Patients)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sota_comparison_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Multi-Model Forecast Waveform Overlay (Lead I, II, V2, V5) ---
    errors = np.abs(cnn_npz["pred"] - true_mv).mean(axis=(1, 2))
    med_idx = int(np.argsort(errors)[len(errors) // 2])
    obs_mv = data["raw_X_val"][med_idx]
    tx = (np.arange(500) - 500) / 100.0
    ty = np.arange(300) / 100.0
    leads_to_show = [(0, "Lead I"), (1, "Lead II"), (7, "Lead V2"), (10, "Lead V5")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True)
    for ax, (l_idx, l_name) in zip(axes.flat, leads_to_show, strict=True):
        ax.plot(tx[-200:], obs_mv[l_idx, -200:], color="0.55", lw=1.0, label="Observed (-2s to 0s)")
        ax.plot(ty, true_mv[med_idx, l_idx], color="black", lw=1.3, alpha=0.85, label="Actual Future")
        ax.plot(ty, lstm_npz["pred"][med_idx, l_idx], color="#1f77b4", lw=1.1, ls="--", label="LSTM")
        ax.plot(ty, cnn_npz["pred"][med_idx, l_idx], color="#ff7f0e", lw=1.2, label="CNN-LSTM")
        ax.plot(ty, ptst_pred[med_idx, l_idx], color="#2ca02c", lw=1.2, label="PatchTST")
        ax.plot(ty, tft_pred[med_idx, l_idx], color="#d62728", lw=1.2, label="TFT")
        ax.axvline(0.0, color="black", ls=":", lw=0.8)
        ax.set_ylabel(f"{l_name} (mV)")
        ax.set_title(f"{l_name} Forecast Trajectory")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=3, loc="upper right")
    axes[1, 0].set_xlabel("Time Relative to Forecast Origin (s)")
    axes[1, 1].set_xlabel("Time Relative to Forecast Origin (s)")
    fig.suptitle(
        f"Multi-Architecture 12-Lead Forecast Overlay (Validation ECG {int(data['val']['meta'].iloc[med_idx].ecg_id)})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sota_forecast_overlay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Interpretability & Explainability (XAI) Analysis ---
    # Load trained CNN-LSTM checkpoint on CPU (for backward through RNN/CNN)
    cpu_dev = torch.device("cpu")
    cnn_cfg = read_json(EXP_DIR / "cnn_lstm_patient_history" / "config.json")["config"]
    cnn_cfg["data"]["layout"] = "channels_first"
    cnn_model = build_model(cnn_cfg, "cnn_lstm", 10).to(cpu_dev)
    cnn_ckpt = torch.load(
        EXP_DIR / "cnn_lstm_patient_history" / "best.pt", map_location=cpu_dev, weights_only=True
    )
    cnn_model.load_state_dict(cnn_ckpt["model"])

    sample_x = data["val"]["x"][:64].to(cpu_dev)
    sample_y = data["val"]["y"][:64].to(cpu_dev)
    sample_s = data["val"]["s"][:64].to(cpu_dev)
    ig_map = compute_integrated_gradients(cnn_model, sample_x, sample_y, sample_s, steps=16)  # [64, 12, 500]
    mean_ig = ig_map.mean(axis=0)  # [12, 500]
    mean_ig_norm = mean_ig / (mean_ig.max() + 1e-12)

    # Extract TFT intrinsic Variable Selection Weights & Temporal Self-Attention
    tft_cpu = tft.to(cpu_dev).eval()
    with torch.inference_mode():
        _, var_weights, tft_attn = tft_cpu(
            data["val"]["x"].to(cpu_dev), data["val"]["s"].to(cpu_dev), return_explanations=True
        )
    mean_var_weights = var_weights.mean(dim=0).numpy()  # [10]
    std_var_weights = var_weights.std(dim=0).numpy()  # [10]
    mean_tft_attn = tft_attn.mean(dim=0).numpy()  # [100, 100]

    # Extract PatchTST Attention
    ptst_cpu = patchtst.to(cpu_dev).eval()
    with torch.inference_mode():
        _, ptst_attn = ptst_cpu(
            data["val"]["x"].to(cpu_dev), data["val"]["s"].to(cpu_dev), return_attention=True
        )
    mean_ptst_attn = ptst_attn.mean(dim=0).numpy()  # [50, 50]

    # Temporal Occlusion Sensitivity across all 4 models (10 bins of 0.5s each across 5.0s)
    val_x_full = data["val"]["x"].to(device)
    val_s_full = data["val"]["s"].to(device)
    cnn_gpu = cnn_model.to(device).eval()
    ptst_gpu = patchtst.to(device).eval()
    tft_gpu = tft.to(device).eval()

    lstm_cfg = read_json(EXP_DIR / "lstm_patient_history" / "config.json")["config"]
    lstm_cfg["data"]["layout"] = "channels_first"
    lstm_model = build_model(lstm_cfg, "lstm", 10).to(device)
    lstm_ckpt = torch.load(
        EXP_DIR / "lstm_patient_history" / "best.pt", map_location=device, weights_only=True
    )
    lstm_model.load_state_dict(lstm_ckpt["model"])
    lstm_model.eval()

    occlusion_results = {m: [] for m in ("LSTM", "CNN-LSTM", "PatchTST", "TFT")}
    bin_centers = []
    for b_idx in range(10):
        start_idx = b_idx * 50
        end_idx = (b_idx + 1) * 50
        bin_centers.append(-5.0 + (b_idx + 0.5) * 0.5)
        x_occ = val_x_full.clone()
        x_occ[:, :, start_idx:end_idx] = 0.0
        with torch.inference_mode():
            for m_name, m_obj, base_mae in [
                ("LSTM", lstm_model, lstm_metrics["patient_macro_mae"]),
                ("CNN-LSTM", cnn_gpu, cnn_metrics["patient_macro_mae"]),
                ("PatchTST", ptst_gpu, ptst_summary["patient_macro_mae"]),
                ("TFT", tft_gpu, tft_summary["patient_macro_mae"]),
            ]:
                preds = []
                for offset in range(0, len(x_occ), 128):
                    preds.append(
                        m_obj(x_occ[offset : offset + 128], val_s_full[offset : offset + 128])
                        .cpu()
                        .numpy()
                    )
                p_mv = physical_arrays(np.concatenate(preds, axis=0), manifest, "channels_first")
                occ_mae = metric_summary(p_mv, true_mv, data["val"]["patients"])["patient_macro_mae"]
                occlusion_results[m_name].append((occ_mae - base_mae) * 1000.0)  # in microvolts (uV)

    # Save XAI numerical summary
    xai_summary = {
        "lead_ig_importance": {
            LEADS[i]: float(mean_ig_norm[i].mean()) for i in range(12)
        },
        "tft_variable_selection_weights": {
            selected_features[i]: {
                "mean": float(mean_var_weights[i]),
                "std": float(std_var_weights[i]),
            }
            for i in range(len(selected_features))
        },
        "temporal_occlusion_delta_uV": {
            "bin_centers_s": bin_centers,
            **{k: [float(v) for v in vals] for k, vals in occlusion_results.items()},
        },
    }
    write_json(OUT_DIR / "xai_summary.json", xai_summary)

    # --- Plot 3: Comprehensive 4-Panel XAI Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Spatio-Temporal Integrated Gradients Heatmap
    im0 = axes[0, 0].imshow(
        mean_ig_norm,
        aspect="auto",
        cmap="magma",
        extent=[-5.0, 0.0, 11.5, -0.5],
    )
    axes[0, 0].set_yticks(range(12))
    axes[0, 0].set_yticklabels(LEADS)
    axes[0, 0].set_xlabel("Observed Time Relative to Forecast Origin (s)")
    axes[0, 0].set_ylabel("ECG Lead")
    axes[0, 0].set_title("(a) Integrated Gradients Attribution Heatmap (CNN-LSTM)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Normalized Attribution")

    # Panel (b): TFT Intrinsic Static Variable Selection Weights
    order = np.argsort(mean_var_weights)
    axes[0, 1].barh(
        [selected_features[i] for i in order],
        mean_var_weights[order],
        xerr=std_var_weights[order],
        color="#2c7bb6",
        alpha=0.85,
        ecolor="black",
        capsize=3,
    )
    axes[0, 1].axvline(1.0 / len(selected_features), color="#d62728", ls="--", lw=1.0, label="Uniform (0.10)")
    axes[0, 1].set_xlabel("Softmax Variable Selection Weight")
    axes[0, 1].set_title("(b) TFT Static Variable Selection Network (VSN) Weights")
    axes[0, 1].grid(axis="x", alpha=0.25)
    axes[0, 1].legend(fontsize=8.5)

    # Panel (c): Temporal Occlusion Sensitivity across all 4 DL Architectures
    for m_name, color, marker in [
        ("LSTM", "#1f77b4", "o"),
        ("CNN-LSTM", "#ff7f0e", "s"),
        ("PatchTST", "#2ca02c", "^"),
        ("TFT", "#d62728", "d"),
    ]:
        axes[1, 0].plot(
            bin_centers,
            occlusion_results[m_name],
            marker=marker,
            lw=1.8,
            color=color,
            label=m_name,
        )
    axes[1, 0].axhline(0.0, color="black", ls=":", lw=0.8)
    axes[1, 0].set_xlabel("Occluded 0.5s Window Center (s relative to forecast origin)")
    axes[1, 0].set_ylabel(r"$\Delta$ Validation Patient-Macro MAE ($\mu$V)")
    axes[1, 0].set_title("(c) Temporal Occlusion Sensitivity Profile Across Models")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8.5)

    # Panel (d): Temporal Self-Attention Profiles (TFT vs PatchTST)
    tft_temporal_imp = mean_tft_attn[-8:, :].mean(axis=0)  # attention from terminal 8 steps over 100 steps
    tft_t = np.linspace(-5.0, 0.0, len(tft_temporal_imp))
    ptst_patch_imp = mean_ptst_attn[-8:, 1:].mean(axis=0)  # attention from terminal 8 patches over 49 patches
    ptst_t = np.linspace(-4.9, -0.1, len(ptst_patch_imp))
    ax_d = axes[1, 1]
    ax_d.plot(
        tft_t,
        tft_temporal_imp / tft_temporal_imp.sum(),
        color="#d62728",
        lw=2.0,
        label="TFT Temporal Self-Attention (100 steps)",
    )
    ax_d.plot(
        ptst_t,
        ptst_patch_imp / ptst_patch_imp.sum(),
        color="#2ca02c",
        lw=2.0,
        ls="--",
        marker=".",
        label="PatchTST Cross-Patch Attention (49 patches)",
    )
    ax_d.set_xlabel("Observed Time Relative to Forecast Origin (s)")
    ax_d.set_ylabel("Normalized Attention Mass")
    ax_d.set_title("(d) Intrinsic Attention Distribution Over Observed History")
    ax_d.grid(alpha=0.25)
    ax_d.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "xai_interpretability_suite.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved all SOTA & XAI figures and metrics to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
