"""Phase 2: dataset understanding for PTB-XL folds 1-9 (fold 10 sealed).

See reports/phase2_dataset/PREREGISTRATION.md. Writes tables, figures and summary.json.

    python research/phase2_dataset.py --data PTB-XL --out reports/phase2_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ptbxl_forecast.data import (  # noqa: E402
    LEADS,
    QUALITY_FLAGS,
    SUPERCLASSES,
    load_metadata,
    load_signals,
)

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402

DEV_FOLDS = list(range(1, 10))
TRAIN_FOLDS, VAL_FOLDS = list(range(1, 9)), [9]
FS = 100


def heart_rate(sig: np.ndarray) -> float:
    rr = p0.plausible_rr(p0.rpeaks(sig[:, 1].astype(np.float64)))
    return float(60 * FS / np.median(rr)) if len(rr) else np.nan


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def counts(series: pd.Series, n: int) -> pd.DataFrame:
    c = series.value_counts()
    return pd.DataFrame({"records": c, "percent": (100 * c / n).round(2)})


def explode_counts(sets: pd.Series, n: int) -> pd.DataFrame:
    return counts(pd.Series([x for s in sets for x in s]), n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PTB-XL"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase2_dataset"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/data"))
    ap.add_argument("--workers", type=int, default=38)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    S: dict = {}

    meta = load_metadata(args.data, DEV_FOLDS)
    n = len(meta)
    S["records"], S["patients"] = n, int(meta.patient_id.nunique())

    # ------------------------------------------------------------ Q1 patients
    per_patient = meta.groupby("patient_id").size()
    multi = per_patient[per_patient > 1].index
    S["records_per_patient"] = {
        str(k): int(v) for k, v in per_patient.value_counts().sort_index().items()
    }
    S["R3_share_records_from_multi_record_patients"] = float(meta.patient_id.isin(multi).mean())
    S["R3_pass"] = S["R3_share_records_from_multi_record_patients"] >= 0.10
    m = meta[meta.patient_id.isin(multi)].sort_values(["patient_id", "recording_dt"])
    gaps = m.groupby("patient_id").recording_dt.diff().dropna().dt.total_seconds() / 86400
    S["repeat_gap_days"] = {
        "n_gaps": int(len(gaps)),
        "same_day_share": float((gaps < 1).mean()),
        "within_1h_share": float((gaps < 1 / 24).mean()),
        "median": float(gaps.median()),
        "p90": float(gaps.quantile(0.9)),
        "max": float(gaps.max()),
    }

    # ------------------------------------------------------------ Q2 demographics
    S["age"] = {
        "missing": int(meta.age.isna().sum()),
        "over_89": int(meta.age_over_89.sum()),
        "over_89_share": float(meta.age_over_89.mean()),
        **{k: float(v) for k, v in meta.age_years.describe().items()},
        "under_18": int((meta.age_years < 18).sum()),
    }
    S["sex"] = {str(k): int(v) for k, v in meta.sex_label.value_counts(dropna=False).items()}
    S["height_missing_share"] = float(meta.height.isna().mean())
    S["weight_missing_share"] = float(meta.weight.isna().mean())
    S["height_and_weight_present"] = int((meta.height.notna() & meta.weight.notna()).sum())
    S["height_cm"] = {k: float(v) for k, v in meta.height.describe().items()}
    S["weight_kg"] = {k: float(v) for k, v in meta.weight.describe().items()}

    # ------------------------------------------------------------ Q3 labels
    sc = pd.DataFrame({c: meta[f"sc_{c}"] for c in SUPERCLASSES})
    S["superclass_prevalence"] = {c: float(sc[c].mean()) for c in SUPERCLASSES}
    S["n_superclasses_per_record"] = {
        str(k): int(v) for k, v in sc.sum(1).value_counts().sort_index().items()
    }
    combos = meta.diag_superclasses.apply(lambda s: "+".join(sorted(s)) or "(none)")
    counts(combos, n).head(20).to_csv(out / "table_superclass_combinations.csv")
    explode_counts(meta.diag_subclasses, n).to_csv(out / "table_subclasses.csv")
    explode_counts(meta.rhythm_codes, n).to_csv(out / "table_rhythm_codes.csv")
    explode_counts(meta.form_codes, n).to_csv(out / "table_form_codes.csv")
    S["records_without_rhythm_code"] = int((meta.rhythm_codes.apply(len) == 0).sum())
    S["records_without_diagnostic_code"] = int((meta.diag_superclasses.apply(len) == 0).sum())
    S["codes_per_record"] = {k: float(v) for k, v in meta.scp.apply(len).describe().items()}
    likelihoods = Counter(v for c in meta.scp for v in c.values())
    S["scp_likelihood_values"] = {str(k): v for k, v in sorted(likelihoods.items())}
    S["validated_by_human_share"] = float(meta.validated_by_human.mean())
    S["second_opinion_share"] = float(meta.second_opinion.mean())
    S["initial_autogenerated_report_share"] = float(meta.initial_autogenerated_report.mean())
    S["heart_axis"] = {
        str(k): int(v) for k, v in meta.heart_axis.value_counts(dropna=False).items()
    }
    S["pacemaker_flag"] = int(meta.pacemaker.notna().sum())
    S["quality_flags_share"] = {f: float(meta[f"q_{f}"].mean()) for f in (*QUALITY_FLAGS, "any")}

    # ------------------------------------------------------------ Q4 acquisition
    S["years"] = {
        str(k): int(v) for k, v in meta.recording_dt.dt.year.value_counts().sort_index().items()
    }
    S["sites"] = {"n_unique": int(meta.site.nunique()), "missing": int(meta.site.isna().sum()),
                  "top5_share": float(meta.site.value_counts().head(5).sum() / n)}  # fmt: skip
    S["devices"] = {str(k): int(v) for k, v in meta.device.value_counts(dropna=False).items()}
    S["nurses_unique"] = int(meta.nurse.nunique())
    S["records_per_fold"] = {
        int(k): int(v) for k, v in meta.strat_fold.value_counts().sort_index().items()
    }

    # ------------------------------------------------------------ Q5 signals
    sig = np.asarray(load_signals(args.data, meta, cache_dir=args.cache, workers=args.workers))
    offset = sig.mean(axis=1)  # (n, 12) per-record per-lead mean
    sd = sig.std(axis=1)
    p01, p99 = np.percentile(sig, [1, 99], axis=1)
    lead_stats = pd.DataFrame(
        {
            "sd_median": np.median(sd, 0),
            "sd_p05": np.percentile(sd, 5, 0),
            "sd_p95": np.percentile(sd, 95, 0),
            "offset_median": np.median(offset, 0),
            "abs_offset_p95": np.percentile(np.abs(offset), 95, 0),
            "range_p01_p99_median": np.median(p99 - p01, 0),
            "abs_max_p99": np.percentile(np.abs(sig).max(axis=1), 99, 0),
        },
        index=LEADS,
    )
    lead_stats.to_csv(out / "table_lead_statistics.csv", float_format="%.4f")
    S["sd_ratio_max_over_min_lead"] = {
        k: float(v)
        for k, v in pd.Series(sd.max(1) / np.maximum(sd.min(1), 1e-6)).describe().items()
    }
    S["global_value_percentiles_mV"] = {
        p: float(v) for p, v in zip(("p0.1", "p1", "p50", "p99", "p99.9"),
                                    np.percentile(sig[::7], [0.1, 1, 50, 99, 99.9]), strict=True)
    }  # fmt: skip
    centred = sig - offset[:, None, :]
    corr = np.corrcoef(centred.reshape(-1, 12)[::50].T)
    pd.DataFrame(corr, index=LEADS, columns=LEADS).to_csv(
        out / "table_lead_correlation.csv", float_format="%.3f"
    )

    # R2 lead relationships
    I, II = sig[..., 0], sig[..., 1]  # noqa: E741
    derived = {"III": II - I, "aVR": -(I + II) / 2, "aVL": I - II / 2, "aVF": II - I / 2}
    resid = {}
    for name, pred in derived.items():
        r = np.sqrt(((sig[..., LEADS.index(name)] - pred) ** 2).mean(axis=1))
        resid[name] = r
    resid_df = pd.DataFrame(resid, index=meta.index)
    S["R2_residual_rms_mV"] = {k: {"median": float(v.median()), "p99": float(v.quantile(0.99)),
                                  "max": float(v.max()), "share_le_0.01": float((v <= 0.01).mean())}
                              for k, v in resid_df.items()}  # fmt: skip
    S["R2_pass"] = all(v["share_le_0.01"] >= 0.99 for v in S["R2_residual_rms_mV"].values())
    rank_sv = np.linalg.svd(centred[::20], compute_uv=False)  # (n/20, 12)
    S["svd_variance_share_top8"] = float(
        np.median((rank_sv[:, :8] ** 2).sum(1) / (rank_sv**2).sum(1))
    )
    S["svd_variance_share_top2_limb"] = float(np.median(
        (np.linalg.svd(centred[::20, :, :6], compute_uv=False)[:, :2] ** 2).sum(1)
        / (np.linalg.svd(centred[::20, :, :6], compute_uv=False) ** 2).sum(1)))  # fmt: skip

    # Per-record descriptors (heart rate via Phase 0's detector)
    with Pool(args.workers) as pool:
        hr = pool.map(heart_rate, list(sig), chunksize=64)
    meta["hr_bpm"] = hr
    meta["mean_lead_sd"] = sd.mean(1)
    S["hr_bpm"] = {k: float(v) for k, v in meta.hr_bpm.describe().items()}

    # ------------------------------------------------------------ R1 fold 9 vs folds 1-8
    meta["female"] = meta.sex == 1
    meta["rh_SR"] = meta.rhythm_codes.apply(lambda s: "SR" in s)
    meta["rh_AFIB"] = meta.rhythm_codes.apply(lambda s: "AFIB" in s)
    variables = ["age_years", "female", "age_over_89", *[f"sc_{c}" for c in SUPERCLASSES],
                 "rh_SR", "rh_AFIB", *[f"q_{f}" for f in QUALITY_FLAGS], "hr_bpm", "mean_lead_sd"]  # fmt: skip
    tr, va = meta[meta.strat_fold.isin(TRAIN_FOLDS)], meta[meta.strat_fold.isin(VAL_FOLDS)]
    smd_rows = [{"variable": v, "train_mean": float(tr[v].astype(float).mean()),
                 "val_mean": float(va[v].astype(float).mean()), "smd": smd(va[v], tr[v])}
                for v in variables]  # fmt: skip
    smd_df = pd.DataFrame(smd_rows)
    smd_df.to_csv(out / "table_fold9_vs_train_smd.csv", index=False, float_format="%.4f")
    # Context: the same SMD for every other fold vs the remaining training folds.
    spread = {}
    for f in TRAIN_FOLDS:
        a, b = meta[meta.strat_fold == f], meta[meta.strat_fold.isin(set(TRAIN_FOLDS) - {f})]
        spread[f] = max(abs(smd(a[v], b[v])) for v in variables)
    S["R1_max_abs_smd_fold9"] = float(smd_df.smd.abs().max())
    S["R1_worst_variable"] = smd_df.loc[smd_df.smd.abs().idxmax(), "variable"]
    S["R1_pass"] = S["R1_max_abs_smd_fold9"] < 0.1
    S["max_abs_smd_each_train_fold_vs_rest"] = spread

    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    meta[["patient_id", "strat_fold", "hr_bpm", "mean_lead_sd"]].join(
        resid_df.add_prefix("resid_")
    ).to_csv(args.cache / "phase2_per_record.csv")
    figures(meta, sig, sd, offset, lead_stats, corr, resid_df, smd_df, per_patient, gaps, out)
    S["posthoc"] = posthoc(meta, sig, sd, resid_df, out)
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    print(json.dumps(S, indent=2, default=str))


def posthoc(meta, sig, sd, resid_df, out, phase0=Path("results/phase0/per_record.csv")) -> dict:
    """Analyses added after seeing the pre-registered results (labelled post hoc)."""
    from scipy.signal import welch

    P: dict = {}
    bad = resid_df[(resid_df > 0.01).any(axis=1)]
    P["limb_relation_violators"] = {
        "n": len(bad),
        "ecg_ids": bad.index.tolist(),
        "devices": meta.loc[bad.index].device.str.split().str[0].value_counts().to_dict(),
    }
    P["records_with_lead_sd_below_0.01mV"] = int((sd < 0.01).any(1).sum())
    P["records_with_lead_sd_below_0.02mV"] = int((sd < 0.02).any(1).sum())
    P["implausible_anthropometrics"] = {
        "height_lt_100cm": int((meta.height < 100).sum()),
        "height_lt_100cm_age_lt_18": int(((meta.height < 100) & (meta.age_years < 18)).sum()),
        "weight_lt_20kg": int((meta.weight < 20).sum()),
        "weight_gt_180kg": int((meta.weight > 180).sum()),
    }
    val = meta.strat_fold == 9
    human = meta.validated_by_human.astype(bool)
    P["fold9_vs_train"] = {
        "validated_by_human_folds1_8": float(human[~val].mean()),
        "validated_by_human_fold9": float(human[val].mean()),
        "baseline_drift_if_human_validated_folds1_8": float(
            meta[~val & human].q_baseline_drift.mean()
        ),
        "baseline_drift_if_not_human_validated_folds1_8": float(
            meta[~val & ~human].q_baseline_drift.mean()
        ),
    }

    # Device family: composition, spectrum, flags and (if Phase 0 ran) forecastability.
    dev = meta.device.str.split().str[0]
    f, pxx = welch(sig, fs=FS, nperseg=256, axis=1)
    pxx = pxx.mean(axis=2)
    lf = pxx[:, f < 0.5].sum(1) / pxx.sum(1)
    d = pd.DataFrame({"dev": dev, "val": val, "lf": lf, "sd": sd.mean(1)}, index=meta.index)
    d = d.join(meta[["q_baseline_drift", "q_static_noise"]])
    g = d.groupby("dev")
    table = pd.DataFrame(
        {
            "records": g.size(),
            "share_folds1_8": d[~d.val].dev.value_counts(normalize=True),
            "share_fold9": d[d.val].dev.value_counts(normalize=True),
            "power_below_0.5Hz_median": g.lf.median(),
            "mean_lead_sd_median": g.sd.median(),
            "baseline_drift_flag": g.q_baseline_drift.mean(),
            "static_noise_flag": g.q_static_noise.mean(),
        }
    )
    a = d[d.val].dev.value_counts(normalize=True).reindex(table.index, fill_value=0)
    b = d[~d.val].dev.value_counts(normalize=True).reindex(table.index, fill_value=0)
    P["device_smd_fold9_vs_train"] = ((a - b) / np.sqrt((a * (1 - a) + b * (1 - b)) / 2)).to_dict()
    if phase0.exists():
        cols = ["mae_median", "mae_mean", "mae_sn4", "mae_oracle"]
        d = d.join(pd.read_csv(phase0, index_col="ecg_id")[cols])
        g = d.groupby("dev")
        table["mae_median_forecast"] = g.mae_median.mean()
        table["sn4_skill_vs_mean"] = 1 - g.mae_sn4.sum() / g.mae_mean.sum()
        table["oracle_skill_vs_mean"] = 1 - g.mae_oracle.sum() / g.mae_mean.sum()
        tr = ~d.val.to_numpy()
        w = (a / b).reindex(d.dev).to_numpy()
        P["phase0_median_forecast_mae"] = {
            "folds1_8": float(d.mae_median[tr].mean()),
            "fold9": float(d.mae_median[~tr].mean()),
            "folds1_8_reweighted_to_fold9_device_mix": float(
                np.average(d.mae_median[tr], weights=w[tr])
            ),
        }
    table.to_csv(out / "table_device_effects.csv", float_format="%.4f")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    table[["share_folds1_8", "share_fold9"]].plot.bar(ax=ax[0], rot=0, color=["#4c72b0", "#dd8452"])
    ax[0].set(ylabel="share of records", title="Device family by split", xlabel="")
    for name, grp in d.groupby("dev"):
        v = np.sort(grp.lf)
        ax[1].plot(v, np.linspace(0, 1, len(v)), label=name)
    ax[1].set(xscale="log", xlabel="share of power below 0.5 Hz", ylabel="cumulative share",
              title="Low-frequency content by device")  # fmt: skip
    ax[1].legend(fontsize=8)
    if "oracle_skill_vs_mean" in table:
        table[["sn4_skill_vs_mean", "oracle_skill_vs_mean"]].plot.bar(
            ax=ax[2], rot=0, color=["#2171b5", "#d94801"]
        )
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set(
            ylabel="skill vs history mean (Phase 0)", title="Forecastability by device", xlabel=""
        )
    fig.tight_layout()
    fig.savefig(out / "fig8_device.png", dpi=150)
    return P


def figures(meta, sig, sd, offset, lead_stats, corr, resid_df, smd_df, per_patient, gaps, out):
    n = len(meta)
    # Fig 1: patients and repeat recordings
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    vc = per_patient.value_counts().sort_index()
    ax[0].bar(vc.index.astype(str), vc.values, color="#4c72b0")
    ax[0].set(
        yscale="log",
        xlabel="records per patient",
        ylabel="patients (log)",
        title="Records per patient",
    )
    ax[1].hist(np.log10(gaps.clip(lower=1e-3)), bins=50, color="#4c72b0")
    ax[1].set(xlabel="days between a patient's consecutive records (log10)", ylabel="pairs",
              title=f"Repeat-recording gaps (n={len(gaps)})")  # fmt: skip
    for x, lab in ((0, "1 d"), (np.log10(365), "1 y")):
        ax[1].axvline(x, color="k", ls="--", lw=0.8)
        ax[1].text(x, ax[1].get_ylim()[1] * 0.9, lab, ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_patients.png", dpi=150)

    # Fig 2: demographics
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for s, col in (("male", "#4c72b0"), ("female", "#dd8452")):
        ax[0].hist(meta.age_years[meta.sex_label == s].dropna(), bins=np.arange(0, 91, 3), alpha=0.6,
                   label=s, color=col)  # fmt: skip
    ax[0].bar(92, meta.age_over_89.sum(), width=3, color="grey", label="> 89 (coded 300)")
    ax[0].set(xlabel="age (years)", ylabel="records", title="Age by sex")
    ax[0].legend(fontsize=8)
    miss = pd.Series({"age": meta.age.isna().mean(), "sex": meta.sex.isna().mean(),
                      "height": meta.height.isna().mean(), "weight": meta.weight.isna().mean(),
                      "site": meta.site.isna().mean(), "nurse": meta.nurse.isna().mean()})  # fmt: skip
    ax[1].barh(miss.index, 100 * miss.values, color="#c44e52")
    ax[1].set(xlabel="% missing", title="Metadata missingness")
    hw = meta[["height", "weight"]].dropna()
    ax[2].scatter(hw.height, hw.weight, s=2, alpha=0.3)
    ax[2].set(xlabel="height (cm)", ylabel="weight (kg)", title=f"Height vs weight (n={len(hw)})")
    fig.tight_layout()
    fig.savefig(out / "fig2_demographics.png", dpi=150)

    # Fig 3: labels
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    scp = pd.Series({c: meta[f"sc_{c}"].mean() * 100 for c in SUPERCLASSES})
    ax[0].bar(scp.index, scp.values, color="#55a868")
    ax[0].set(ylabel="% of records", title="Diagnostic superclasses (multi-label)")
    rh = pd.Series([x for s in meta.rhythm_codes for x in s]).value_counts() / n * 100
    ax[1].barh(rh.index[::-1], rh.values[::-1], color="#8172b2")
    ax[1].set(xscale="log", xlabel="% of records (log)", title="Rhythm statements")
    sub = pd.Series([x for s in meta.diag_subclasses for x in s]).value_counts().head(15) / n * 100
    ax[2].barh(sub.index[::-1], sub.values[::-1], color="#55a868")
    ax[2].set(xlabel="% of records", title="Top 15 diagnostic subclasses")
    fig.tight_layout()
    fig.savefig(out / "fig3_labels.png", dpi=150)

    # Fig 4: acquisition
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    yr = meta.recording_dt.dt.year.value_counts().sort_index()
    ax[0].bar(yr.index, yr.values, color="#64b5cd")
    ax[0].set(xlabel="year", ylabel="records", title="Recordings per year")
    dev = meta.device.value_counts()
    ax[1].barh(dev.index[::-1], dev.values[::-1], color="#64b5cd")
    ax[1].set(xscale="log", xlabel="records (log)", title="Device")
    site = meta.site.value_counts().head(15)
    ax[2].bar(site.index.astype(int).astype(str), site.values, color="#64b5cd")
    ax[2].set(
        xlabel="site id (top 15)", ylabel="records", title=f"Sites ({meta.site.nunique()} total)"
    )
    ax[2].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4_acquisition.png", dpi=150)

    # Fig 5: signal statistics
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].boxplot([sd[:, j] for j in range(12)], tick_labels=LEADS, showfliers=False)
    ax[0].set(ylabel="per-record SD (mV)", title="Amplitude by lead")
    ax[1].boxplot([offset[:, j] for j in range(12)], tick_labels=LEADS, showfliers=False)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(ylabel="per-record mean (mV)", title="DC offset by lead")
    im = ax[2].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[2].set_xticks(range(12), LEADS, rotation=90, fontsize=8)
    ax[2].set_yticks(range(12), LEADS, fontsize=8)
    ax[2].set_title("Inter-lead correlation (record-centred)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig5_signals.png", dpi=150)

    # Fig 6: lead relationships and fold 9 representativeness
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for name in resid_df:
        v = np.sort(resid_df[name].clip(lower=1e-5))
        ax[0].plot(v, np.linspace(0, 1, len(v)), label=name)
    ax[0].axvline(0.01, color="k", ls="--", lw=0.8)
    ax[0].set(xscale="log", xlabel="RMS residual vs Einthoven/Goldberger relation (mV)",
              ylabel="cumulative share of records", title="Limb-lead redundancy")  # fmt: skip
    ax[0].legend(fontsize=8)
    s = smd_df.iloc[::-1]
    ax[1].barh(s.variable, s.smd, color=np.where(s.smd.abs() < 0.1, "#55a868", "#c44e52"))
    for x in (-0.1, 0.1):
        ax[1].axvline(x, color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="SMD, fold 9 vs folds 1-8", title="Validation-fold representativeness")
    ax[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_leads_and_folds.png", dpi=150)

    # Fig 7: one example 12-lead ECG per common superclass (training folds only)
    rng = np.random.default_rng(20261122)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    picks = [("NORM", meta.sc_NORM & (meta.diag_superclasses.apply(len) == 1)),
             ("MI", meta.sc_MI & (meta.diag_superclasses.apply(len) == 1)),
             ("AFIB rhythm", meta.rhythm_codes.apply(lambda r: "AFIB" in r))]  # fmt: skip
    t = np.arange(1000) / FS
    for a, (label, mask) in zip(axes, picks, strict=True):
        pool = np.flatnonzero((mask & ~meta.q_any & meta.strat_fold.isin(TRAIN_FOLDS)).to_numpy())
        i = int(rng.choice(pool))
        for j in range(12):
            a.plot(t, sig[i, :, j] - 2.0 * j, lw=0.7, color="k")
        a.axvline(7, color="r", ls="--", lw=0.8)
        a.set(title=f"{label} (ecg_id {meta.index[i]})", xlabel="time (s)")
    axes[0].set_yticks(-2.0 * np.arange(12), LEADS)
    fig.tight_layout()
    fig.savefig(out / "fig7_examples.png", dpi=120)


if __name__ == "__main__":
    main()
