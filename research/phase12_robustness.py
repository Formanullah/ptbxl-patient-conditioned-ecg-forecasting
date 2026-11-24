"""Phase 12: robustness of the final configuration (PREREGISTRATION.md).

Out-of-fold (OOF) forecasts from the 9-fold CV (3-seed ensemble per fold) cover every
cleaned record of folds 1-9 once. They are analysed by stratum (device, quality,
baseline wander, rhythm, heart rate, human validation) and the final model is also
tested under input perturbations on fold 9 (6 saved models).

    python research/phase12_robustness.py   # after the training queue (run_12_18.py)
"""

from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase8_validation as p8  # noqa: E402
import phase9_models as p9  # noqa: E402
import phase10_objectives as p10  # noqa: E402
import phase11_tuning as p11  # noqa: E402

OUT = Path("reports/phase12_robustness")
CACHE = Path(".cache/data/phase12")
H, F = 700, 290
SEEDS6, CV_SEEDS, SUITE = p9.SEEDS6, p9.CV_SEEDS, p9.SUITE
MIN_PATIENTS = 200  # strata smaller than this are reported, not tested
PERTURB = {"noise 0.02 mV": ("noise", 0.02), "noise 0.05 mV": ("noise", 0.05),
           "lead II missing": ("drop", 1)}  # fmt: skip
EXTRA = ["r_amp_err", "st_err", "t_amp_err", "n_pred_beats", "n_true_beats"]


# ------------------------------------------------------------------ training specs
def keep_path(s: int) -> Path:
    return CACHE / "fold9_models" / f"final_k9_s{s}.npz"


def specs() -> list[dict]:
    """9-fold CV of the final configuration (none needed if it is the Phase 10 base,
    whose CV runs exist) and 6 fold-9 runs with saved weights for perturbation tests."""
    c = p11.final_config()
    (CACHE / "fold9_models").mkdir(parents=True, exist_ok=True)
    out = [p11.spec(c, k, s) for k in range(1, 10) for s in CV_SEEDS]
    out += [
        {"config": c, "k": 9, "s": s, "path": str(keep_path(s)), "keep_model": True} for s in SEEDS6
    ]
    return out


# ------------------------------------------------------------------ out-of-fold scores
def oof_paths() -> dict:
    c = p11.final_config()
    lr_l1 = p10.LR
    return {
        "final": {k: [str(p11.run_path(c, k, s)) for s in CV_SEEDS] for k in range(1, 10)},
        "l1": {k: [str(p9.run_path("cnn_lstm", lr_l1, False, k, s)) for s in CV_SEEDS] for k in range(1, 10)},
    }  # fmt: skip


def oof_scores():
    """Per-record scores (all metrics) of final, l1, flat and sn4 on folds 1-9, with
    metadata. Returns (table, meta, fold_rows)."""
    meta, sig, fold_rows, pids, base = p9.setup_scoring()
    paths = oof_paths()
    jobs = {(a, k): (k, paths[a][k]) for a in paths for k in range(1, 10)}
    cache = CACHE / "scores_oof.pkl"
    old = p10.CACHE / "scores_stage_c.pkl"
    if not cache.exists() and p11.final_config() == p11.BASE and old.exists():
        # same runs as Phase 10 stage C (identical paths): reuse its scores
        prev = pickle.load(old.open("rb"))
        seeded = {
            ("final", k): prev[("sn4_gated", k)] for k in range(1, 10) if ("sn4_gated", k) in prev
        }
        seeded.update({("l1", k): prev[("l1", k)] for k in range(1, 10) if ("l1", k) in prev})
        pickle.dump(seeded, cache.open("wb"))
    sc = p9.score_many(jobs, cache)
    parts = []
    for k in range(1, 10):
        ids = meta.index[fold_rows[k]]
        arms = {
            "final": sc[("final", k)],
            "l1": sc[("l1", k)],
            "flat": base[("flat", k)],
            "sn4": base[("sn4", k)],
        }
        for a, df in arms.items():
            d = df.reset_index(drop=True)[[*SUITE, *EXTRA]].copy()
            d.insert(0, "arm", a)
            d.insert(1, "ecg_id", ids)
            d.insert(2, "fold", k)
            parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    return t, meta, fold_rows


def strata(meta: pd.DataFrame) -> pd.DataFrame:
    """One categorical column per stratification factor, indexed by ecg_id."""
    p2 = pd.read_csv(".cache/data/phase2_per_record.csv", index_col="ecg_id")
    p3 = pd.read_csv(".cache/data/phase3_per_record.csv", index_col="ecg_id")
    s = pd.DataFrame(index=meta.index)
    s["patient_id"] = meta.patient_id
    s["device"] = p3.device_family.reindex(meta.index)
    s["quality_flag"] = np.where(meta.q_any, "flagged", "clean")
    bw = p3.bw_pp_med.reindex(meta.index)
    s["baseline_wander"] = pd.qcut(bw, 3, labels=["low", "mid", "high"]).astype(str)

    def rhythm(codes: set) -> str:
        if codes & {"AFIB", "AFLT"}:
            return "AF/flutter"
        if "SR" in codes:
            return "sinus"
        if codes & {"STACH", "SBRAD", "SARRH"}:
            return "sinus tachy/brady/arrhythmia"
        return "other"

    s["rhythm"] = meta.rhythm_codes.apply(rhythm)
    s.loc[meta.scp.apply(lambda c: "PACE" in c), "rhythm"] = "paced"
    hr = p2.hr_bpm.reindex(meta.index)
    s["heart_rate"] = pd.cut(
        hr, [0, 60, 100, 1000], labels=["<60", "60-100", ">100"], right=False
    ).astype(str)
    s["validated"] = np.where(meta.validated_by_human, "human-validated", "not validated")
    return s


def stratum_table(t: pd.DataFrame, s: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    from ptbxl_forecast.stats import compare, patient_macro

    wide = {a: g.set_index("ecg_id") for a, g in t.groupby("arm")}
    rows = []
    for fac in factors:
        for lev, ids in s.groupby(fac).groups.items():
            ids = pd.Index(ids)
            pid = s.loc[ids, "patient_id"].to_numpy()
            row = {
                "factor": fac,
                "level": lev,
                "records": len(ids),
                "patients": len(np.unique(pid)),
            }
            for a in ("final", "l1", "flat", "sn4"):
                for m in SUITE:
                    row[f"{a}_{m}"] = patient_macro(wide[a].loc[ids, m], pid)
            if row["patients"] >= MIN_PATIENTS:
                fin = wide["final"].loc[ids].reset_index()
                for ref in ("flat", "l1", "sn4"):
                    c = compare(fin, wide[ref].loc[ids].reset_index(), pid)
                    row[f"beats_{ref}"] = c["beats_reference"]
                    row[f"d_mae_vs_{ref}_uV"] = c["mae"]["diff"] * 1000
                    row[f"d_timing_vs_{ref}_ms"] = c["beat_timing_ms"]["diff"]
                    row[f"d_timing_vs_{ref}_lo"] = c["beat_timing_ms"]["lo"]
                    row[f"d_timing_vs_{ref}_hi"] = c["beat_timing_ms"]["hi"]
                    row[f"d_template_vs_{ref}_uV"] = c["template_mae"]["diff"] * 1000
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ perturbations
def perturb(x: np.ndarray, kind: str, v, rng) -> np.ndarray:
    x = x.astype(np.float64).copy()
    if kind == "noise":
        x += rng.normal(0, v, x.shape)
    elif kind == "drop":
        x[:, :, v] = 0.0
    return x


def _predict_saved(args) -> np.ndarray:
    import torch

    from ptbxl_forecast.models import GatedBaselineCNNLSTM
    from ptbxl_forecast.preprocessing import pack_baseline
    from ptbxl_forecast.training import predict

    path, hist, sn4 = args
    torch.set_num_threads(4)
    c = p11.final_config()
    pipe = pickle.load(Path(path).with_suffix(".pipe.pkl").open("rb"))
    model = GatedBaselineCNNLSTM(pipe.in_channels, 12, F, c["width"], c["hidden"], c["layers"])
    model.load_state_dict(torch.load(Path(path).with_suffix(".pt")))
    packed = pack_baseline(np.concatenate([hist, np.zeros((len(hist), F, 12))], axis=1), sn4, H)[
        :, :H
    ]
    return predict(model, pipe, packed).astype(np.float32)


def perturbation_analysis(meta, fold_rows) -> pd.DataFrame:
    from ptbxl_forecast.stats import compare_runs

    rows9 = fold_rows[9]
    sig = p8._G["sig"]
    pid = meta.patient_id.to_numpy()[rows9]
    clean = sig[rows9, :H].astype(np.float64)
    saved = [str(keep_path(s)) for s in SEEDS6]
    # saved models reproduce the CV/fold-9 runs bitwise
    c = p11.final_config()
    same = all(np.array_equal(np.load(p)["pred"], np.load(p11.run_path(c, 9, s))["pred"])
               for p, s in zip(saved, SEEDS6, strict=True))  # fmt: skip
    out = []
    cache = CACHE / "perturb_scores.pkl"
    done = pickle.load(cache.open("rb")) if cache.exists() else {}
    for name, (kind, v) in {"clean": ("none", 0), **PERTURB}.items():
        if name not in done:
            hist = perturb(clean, kind, v, np.random.default_rng(0)) if kind != "none" else clean
            with Pool(38) as pool:
                sn4 = np.stack(pool.map(p10._sn4, list(hist), chunksize=32))
            with Pool(6) as pool:
                preds = pool.map(_predict_saved, [(p, hist, sn4) for p in saved])
            flat = np.broadcast_to(
                np.median(hist, axis=1, keepdims=True), (len(hist), F, 12)
            ).copy()
            with Pool(8) as pool:
                scored = pool.map(p8._score, [(9, x) for x in [*preds, sn4, flat]])
            done[name] = {"runs": scored[:6], "sn4": scored[6], "flat": scored[7]}
            pickle.dump(done, cache.open("wb"))
        d = done[name]
        row = {
            "perturbation": name,
            "final_mae": np.mean([p9.macro(r, pid)["mae"] for r in d["runs"]]),
        }
        for m in SUITE:
            row[f"final_{m}"] = np.mean([p9.macro(r, pid)[m] for r in d["runs"]])
            row[f"sn4_{m}"] = p9.macro(d["sn4"], pid)[m]
            row[f"flat_{m}"] = p9.macro(d["flat"], pid)[m]
        for ref in ("flat", "sn4"):
            cr = compare_runs(d["runs"], [d[ref]], pid)
            row[f"beats_{ref}"] = cr["beats_reference"]
            row[f"d_mae_vs_{ref}_uV"] = cr["mae"]["diff"] * 1000
        out.append(row)
    t = pd.DataFrame(out)
    t.attrs["saved_models_bitwise_equal"] = same
    return t


# ------------------------------------------------------------------ main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t, meta, fold_rows = oof_scores()
    t.to_parquet(CACHE / "oof_scores.parquet")
    s = strata(meta)
    s.to_parquet(CACHE / "strata.parquet")
    factors = ["device", "quality_flag", "baseline_wander", "rhythm", "heart_rate", "validated"]
    st = stratum_table(t, s, factors)
    st.to_csv(OUT / "table_strata.csv", index=False, float_format="%.6g")
    folds = t.groupby(["arm", "fold"])[list(SUITE)].mean().reset_index()  # record-mean per fold
    pt = perturbation_analysis(meta, fold_rows)
    pt.to_csv(OUT / "table_perturbations.csv", index=False, float_format="%.6g")
    tested = st[st.patients >= MIN_PATIENTS]
    S = {
        "H12_1_beats_flat_all_strata": bool(tested.beats_flat.all()),
        "H12_2_beats_l1_all_strata": bool(tested.beats_l1.all()),
        "strata_not_beating_flat": tested[~tested.beats_flat.astype(bool)][["factor", "level"]].values.tolist(),
        "strata_not_beating_l1": tested[~tested.beats_l1.astype(bool)][["factor", "level"]].values.tolist(),
        "strata_beating_sn4": tested[tested.beats_sn4.astype(bool)][["factor", "level"]].values.tolist(),
        "perturb_beats_flat": dict(zip(pt.perturbation, pt.beats_flat.astype(bool), strict=True)),
        "saved_models_bitwise_equal": bool(pt.attrs["saved_models_bitwise_equal"]),
        "fold_sd_final_mae": float(folds[folds.arm == "final"].mae.std()),
    }  # fmt: skip
    af = st[(st.factor == "rhythm") & (st.level == "AF/flutter")]
    if len(af) and "d_timing_vs_sn4_hi" in af:
        S["H12_3_af_timing_not_worse_than_sn4"] = bool(af.d_timing_vs_sn4_lo.iloc[0] <= 0)
        S["af_timing_final_minus_sn4_ms"] = float(af.d_timing_vs_sn4_ms.iloc[0])
    json.dump(S, (OUT / "summary.json").open("w"), indent=2, default=str)
    figure(st)
    pd.set_option("display.width", 250)
    print(
        st.round(4).to_string(),
        "\n",
        pt.round(4).to_string(),
        "\n",
        json.dumps(S, indent=2, default=str),
    )


def figure(st: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    lab = st.factor + ": " + st.level.astype(str) + " (n=" + st.patients.astype(str) + ")"
    y = np.arange(len(st))
    for a, m, title in zip(ax, ("mae", "template_mae", "beat_timing_ms"),
                           ("MAE (mV)", "template MAE (mV)", "beat timing (ms)"), strict=True):  # fmt: skip
        for arm, c, mk in (
            ("final", "#d62728", "o"),
            ("l1", "#999999", "s"),
            ("flat", "k", "|"),
            ("sn4", "#2ca02c", "^"),
        ):
            a.scatter(st[f"{arm}_{m}"], y, color=c, marker=mk, label=arm, s=25)
        a.set(title=f"OOF folds 1-9: {title}")
        a.grid(alpha=0.3)
    ax[0].set_yticks(y, lab, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_strata.png", dpi=120)


if __name__ == "__main__":
    main()
