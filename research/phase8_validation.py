"""Phase 8: validation strategy (PREREGISTRATION.md).

E1  fold 9 representativeness: fold 9 vs human-validated / other records of folds 1-8
E2  9-fold cross-validation over the official folds 1-9 vs the fold-9 holdout
E3  training randomness: 18 seeds x 3 configurations; calibration of comparison protocols
E4  Phase 5 adoption decisions re-tested with the selected protocol (stats.compare_runs)

python research/phase8_validation.py      # ~1 h on 40 cores (CNN runs are cached)
"""

from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import Pool, get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("reports/phase8_validation")
CACHE = Path(".cache/data/phase8")
P5 = Path(".cache/data/phase5")
H, F = 700, 290
WORKERS, THREADS = 9, 4
DEC = "decompose_0.67"
CV_SEEDS = (42, 123, 999)
S18 = (42, 123, 999, 7, 2024, 31337, *range(101, 113))
N_DRAWS, N_BOOT = 500, 1000
MODELS = ["flat", "sn4", "ridge", "cnn_raw", "cnn_dec", "cnn_dec20"]
LEARNED = MODELS[2:]
CNN_CFG = {"cnn_raw": ("none", 1.0), "cnn_dec": (DEC, 1.0), "cnn_dec20": (DEC, 0.2)}
E3_CFG = {"A": (DEC, 1.0), "R": ("none", 1.0), "S": (DEC, 0.4)}
SUITE = ("mae", "dtw_mae", "template_mae", "beat_timing_ms")


# ------------------------------------------------------------------ training
def split_rows(meta: pd.DataFrame, k: int, fraction: float, subset_seed: int):
    """Train on folds != k minus the 20% internal patient holdout (seed 0); nested
    patient subset of the pool as in Phase 7; evaluate on fold k."""
    from ptbxl_forecast.splits import patient_holdout

    parts = patient_holdout(meta[meta.strat_fold != k], 0.2, seed=0)
    pool = parts["train"]
    patients = np.sort(pool.patient_id.unique())
    order = np.random.default_rng(subset_seed).permutation(patients)
    sub = pool[pool.patient_id.isin(set(order[: max(1, round(fraction * len(patients)))]))]
    idx = meta.index.get_indexer
    return idx(sub.index), idx(parts["val"].index), np.flatnonzero(meta.strat_fold == k), sub


def spec(k: int, filt: str, f: float, ss: int, s: int) -> dict:
    return {"k": k, "filter": filt, "f": f, "ss": 0 if f == 1.0 else ss, "s": s}


def run_path(sp: dict) -> Path:
    """Phase 5 caches are reused for identical runs (bitwise reproducible, D7.1)."""
    if sp["k"] == 9 and sp["f"] == 1.0 and sp["s"] in S18[:6] and not sp.get("verify"):
        p5 = P5 / f"cnn_{'default' if sp['filter'] == 'none' else 'filter_' + DEC}_s{sp['s']}.npz"
        if p5.exists():
            return p5
    tag = "verify_" if sp.get("verify") else ""
    return CACHE / f"{tag}cnn_k{sp['k']}_{sp['filter']}_f{sp['f']:.2f}_ss{sp['ss']}_s{sp['s']}.npz"


def cnn_job(sp: dict) -> str:
    path = run_path(sp)
    if path.exists():
        return str(path)
    import torch

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.models.cnn import CNNForecaster
    from ptbxl_forecast.preprocessing import Pipeline
    from ptbxl_forecast.training import predict, train_forecaster

    torch.set_num_threads(THREADS)
    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    rows, rows_ho, rows_ev, sub = split_rows(meta, sp["k"], sp["f"], sp["ss"])
    pipe = Pipeline(sp["filter"]).fit(sig[rows, :H].astype(np.float64), sub.index)
    model = CNNForecaster(in_channels=pipe.in_channels, out_leads=pipe.out_leads, horizon=F)
    info = train_forecaster(
        model, pipe, sig[rows], sig[rows_ho], H, F, seed=sp["s"], threads=THREADS
    )
    pred = predict(model, pipe, sig[rows_ev, :H].astype(np.float64))
    info.update(sp, n_records=len(rows))
    np.savez(path, pred=pred.astype(np.float32), info=np.array(info, dtype=object))
    return str(path)


def ridge_fold(k: int, meta, sig) -> np.ndarray:
    from ptbxl_forecast.models.ridge import RidgeForecaster

    path = CACHE / f"ridge_k{k}.npz"
    if not path.exists():
        rows, _, rows_ev, sub = split_rows(meta, k, 1.0, 0)
        x = sig[rows, :H].astype(np.float64).reshape(len(rows), -1)
        y = sig[rows, H : H + F].astype(np.float64).reshape(len(rows), -1)
        model = RidgeForecaster(seed=42).fit(x, y, sub.patient_id.to_numpy())
        xe = sig[rows_ev, :H].astype(np.float64).reshape(len(rows_ev), -1)
        pred = model.predict(xe).reshape(len(rows_ev), F, 12)
        np.savez(path, pred=pred.astype(np.float32), lam=model.lam)
    return np.load(path)["pred"].astype(np.float64)


# ------------------------------------------------------------------ scoring
_G: dict = {}


def _peaks(i: int) -> np.ndarray:
    from ptbxl_forecast.metrics import detect_beats

    return detect_beats(
        _G["sig"][i, :H].astype(np.float64), _G["sig"][i, H : H + F].astype(np.float64)
    )


def _score(job: tuple) -> pd.DataFrame:
    from ptbxl_forecast import metrics as M

    k, pred = job
    rows = _G["fold_rows"][k]
    s = _G["sig"][rows].astype(np.float64)
    return M.evaluate(s[:, :H], s[:, H : H + F], pred, true_peaks=[_G["peaks"][i] for i in rows])


def ensemble(paths: list[str]) -> np.ndarray:
    return np.mean([np.load(p)["pred"].astype(np.float64) for p in paths], axis=0)


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(float).dropna(), b.astype(float).dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# ------------------------------------------------------------------ E3 calibration
def _one_thread() -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _draw(args: tuple) -> list[dict]:
    """One draw: disjoint seed sets for the two arms; every protocol on the same draw."""
    scen, cx, cy, d = args
    rng = np.random.default_rng(10_000 + d)
    perm = rng.permutation(len(S18))
    ia, ib = perm[:6], perm[6:12]
    RX, RY = _G["res"][cx], _G["res"][cy]
    codes, counts, C = _G["codes"], _G["counts"], _G["C"]
    n_pat = len(counts)

    def pm(rec):  # per-patient means of a per-record vector
        return np.bincount(codes, weights=rec, minlength=n_pat) / counts

    def ens_mae(R, idx):
        return np.abs(R[idx].mean(axis=0, dtype=np.float64)).mean(axis=(1, 2))

    def ci(diff_p):
        b = C @ diff_p / n_pat
        return float(diff_p.mean()), *np.percentile(b, [2.5, 97.5])

    out = []

    def add(protocol, est, lo, hi, claim=None):
        claim = (lo > 0 or hi < 0) if claim is None else claim
        out.append({"scenario": scen, "draw": d, "protocol": protocol, "diff": est, "lo": lo,
                    "hi": hi, "claim": bool(claim), "sign": int(np.sign(est)) if claim else 0})  # fmt: skip

    halves = []
    for a_idx, b_idx in ((ia[:3], ib[:3]), (ia[3:], ib[3:])):
        halves.append(ci(pm(ens_mae(RX, a_idx) - ens_mae(RY, b_idx))))
    add("P0-3", *halves[0])
    add("P0-6", *ci(pm(ens_mae(RX, ia) - ens_mae(RY, ib))))
    c0, c1 = ((h[1] > 0 or h[2] < 0) for h in halves)
    same = np.sign(halves[0][0]) == np.sign(halves[1][0])
    add("P1", (halves[0][0] + halves[1][0]) / 2, np.nan, np.nan, claim=c0 and c1 and same)
    MX, MY = _G["mpat"][cx], _G["mpat"][cy]  # (18, patients) single-model patient means
    for n in (3, 6):
        wa = rng.multinomial(n, [1 / n] * n, size=N_BOOT)
        wb = rng.multinomial(n, [1 / n] * n, size=N_BOOT)
        diff = (wa @ MX[ia[:n]] - wb @ MY[ib[:n]]) / n
        boots = (C * diff).sum(axis=1) / n_pat
        est = float((MX[ia[:n]].mean(0) - MY[ib[:n]].mean(0)).mean())
        add(f"P2-{n}", est, *np.percentile(boots, [2.5, 97.5]))
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    from scipy.stats import kendalltau, ttest_1samp

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import compare, patient_macro

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    va_meta, _, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    assert (meta.index[meta.strat_fold == 9] == va_meta.index).all()
    folds = list(range(1, 10))
    S: dict = {}

    # ---------------- training (CNN in parallel, then ridge without contention)
    e2 = [spec(k, *CNN_CFG[m], s, s) for k in folds for m in CNN_CFG for s in CV_SEEDS]
    e3 = [spec(9, *E3_CFG[c], 0, s) for c in E3_CFG for s in S18]
    verify = {**spec(9, DEC, 1.0, 0, 42), "verify": True}
    specs = list({json.dumps(x, sort_keys=True): x for x in e2 + e3 + [verify]}.values())
    specs.sort(key=lambda x: -x["f"])  # longest first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
        list(pool.map(cnn_job, specs))
    S["cnn_minutes"] = (time.perf_counter() - t0) / 60
    S["cnn_runs"] = len(specs)
    S["reuse_verified_bitwise"] = bool(
        np.array_equal(
            np.load(run_path(verify))["pred"], np.load(run_path(spec(9, DEC, 1.0, 0, 42)))["pred"]
        )
    )
    print(f"CNN runs: {len(specs)} in {S['cnn_minutes']:.1f} min", flush=True)
    t0 = time.perf_counter()
    preds = {("ridge", k): ridge_fold(k, meta, sig) for k in folds}
    S["ridge_minutes"] = (time.perf_counter() - t0) / 60
    for m, (filt, f) in CNN_CFG.items():
        for k in folds:
            preds[(m, k)] = ensemble([str(run_path(spec(k, filt, f, s, s))) for s in CV_SEEDS])

    # ---------------- scoring
    fold_rows = {k: np.flatnonzero(meta.strat_fold == k) for k in folds}
    _G.update(sig=sig, fold_rows=fold_rows)
    pk = CACHE / "dev_true_peaks.pkl"
    if not pk.exists():
        with Pool(30) as p:
            pickle.dump(p.map(_peaks, range(len(sig)), chunksize=64), pk.open("wb"))
    _G["peaks"] = pickle.load(pk.open("rb"))
    keys = sorted(preds)
    sc_path = CACHE / "cv_scores.pkl"
    if sc_path.exists():
        scored = pickle.load(sc_path.open("rb"))
    else:
        with Pool(min(30, len(keys))) as p:
            scored = dict(
                zip(keys, p.map(_score, [(k, preds[(m, k)]) for m, k in keys]), strict=True)
            )
        pickle.dump(scored, sc_path.open("wb"))
    p6 = pd.read_parquet(".cache/data/phase6_scores.parquet")
    p6["ecg_id"] = meta.index.to_numpy()[p6.row]
    for name, probe in (("flat", "median (flat)"), ("sn4", "sn4 (repeat cycles)")):
        ref = p6[p6.probe == probe].set_index("ecg_id")
        for k in folds:
            scored[(name, k)] = ref.loc[meta.index[fold_rows[k]]].reset_index(drop=True)

    # ---------------- E1: fold 9 representativeness
    fam = meta.device.str.split().str[0]
    drift = meta.baseline_drift.notna() & (meta.baseline_drift.astype(str).str.strip() != "")
    rec = pd.DataFrame({f"device {d}": (fam == d).astype(float) for d in sorted(fam.unique())})
    rec.index = meta.index
    rec["baseline-drift flag"] = drift.astype(float)
    for name, probe in (("flat", "median (flat)"), ("sn4", "sn4 (repeat cycles)")):
        ref = p6[p6.probe == probe].set_index("ecg_id").loc[meta.index]
        rec[f"{name} MAE"] = ref.mae.to_numpy()
        if name == "sn4":
            rec["sn4 beat timing"] = ref.beat_timing_ms.to_numpy()
    g9 = meta.strat_fold == 9
    gv = ~g9 & meta.validated_by_human.astype(bool)
    gn = ~g9 & ~meta.validated_by_human.astype(bool)
    e1 = pd.DataFrame({
        "fold9": rec[g9].mean(), "validated_1_8": rec[gv].mean(), "other_1_8": rec[gn].mean(),
        "smd_f9_vs_validated": [smd(rec.loc[g9, c], rec.loc[gv, c]) for c in rec],
        "smd_f9_vs_other": [smd(rec.loc[g9, c], rec.loc[gn, c]) for c in rec],
    })  # fmt: skip
    e1.to_csv(OUT / "table_e1_representativeness.csv", float_format="%.4f")
    S["E1_n"] = {"fold9": int(g9.sum()), "validated_1_8": int(gv.sum()), "other_1_8": int(gn.sum())}
    S["H8_1_max_abs_smd_f9_vs_validated"] = float(e1.smd_f9_vs_validated.abs().max())
    S["H8_1_max_abs_smd_f9_vs_other"] = float(e1.smd_f9_vs_other.abs().max())
    S["H8_1_pass"] = bool(
        S["H8_1_max_abs_smd_f9_vs_validated"] < 0.1 and S["H8_1_max_abs_smd_f9_vs_other"] >= 0.2
    )

    # ---------------- E2: holdout vs CV
    pids = {k: meta.patient_id.to_numpy()[fold_rows[k]] for k in folds}
    cv = pd.DataFrame([
        {"model": m, "fold": k, **{x: patient_macro(scored[(m, k)][x], pids[k]) for x in SUITE},
         "share_timing_le_50ms": float((scored[(m, k)].beat_timing_ms <= 50).mean())}
        for m in MODELS for k in folds
    ])  # fmt: skip
    cv.to_csv(OUT / "table_cv.csv", index=False, float_format="%.5f")
    pm = cv.pivot(index="fold", columns="model", values="mae")[MODELS]
    pair_rows = []
    for a, b in combinations(MODELS, 2):
        d = pm[a] - pm[b]
        t = ttest_1samp(d, 0)
        c9 = compare(scored[(a, 9)], scored[(b, 9)], pids[9], n_boot=1000, seed=0)
        pair_rows.append({"a": a, "b": b, "cv_mean_dmae": d.mean(), "cv_sd_dmae": d.std(ddof=1),
                          "cv_p": float(t.pvalue), "cv_significant": bool(t.pvalue < 0.05),
                          "fold9_dmae": d.loc[9], "fold9_lo": c9["mae"]["lo"], "fold9_hi": c9["mae"]["hi"],
                          "folds_same_sign_as_cv_mean": int((np.sign(d) == np.sign(d.mean())).sum()),
                          "fold9_agrees": bool(np.sign(d.loc[9]) == np.sign(d.mean()))})  # fmt: skip
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "table_cv_pairs.csv", index=False, float_format="%.6f")
    cv_mean = pm.mean()
    taus = {k: float(kendalltau(pm.loc[k], cv_mean).statistic) for k in folds}
    sig_pairs = pairs[pairs.cv_significant]
    lo18 = pm.loc[1:8]
    S.update({
        "E2_kendall_tau_by_fold": taus,
        "E2_cv_mean_mae": cv_mean.to_dict(),
        "E2_fold9_mae": pm.loc[9].to_dict(),
        "E2_between_fold_sd_mae_1_8": lo18.std(ddof=1).to_dict(),
        "E2_significant_pairs": int(len(sig_pairs)),
        "E2_significant_pairs_fold9_disagrees": sig_pairs[~sig_pairs.fold9_agrees][["a", "b"]].values.tolist(),
        "H8_2a_pass": bool(sig_pairs.fold9_agrees.all()),
        "H8_2b_pass": bool(taus[9] >= 0.8),
        "H8_3_by_model": {m: bool(pm.loc[9, m] > lo18[m].mean() + 2 * lo18[m].std(ddof=1)) for m in LEARNED},
    })  # fmt: skip
    S["H8_2_pass"] = S["H8_2a_pass"] and S["H8_2b_pass"]
    S["H8_3_pass"] = all(S["H8_3_by_model"].values())

    # ---------------- E3: training randomness
    rows9 = fold_rows[9]
    y9 = sig[rows9, H : H + F].astype(np.float32)
    codes, uniq = pd.factorize(meta.patient_id.to_numpy()[rows9])
    counts = np.bincount(codes).astype(float)
    res, mpat, single = {}, {}, {}
    for c, (filt, f) in E3_CFG.items():
        res[c] = np.stack([np.load(run_path(spec(9, filt, f, 0, s)))["pred"] - y9 for s in S18])
        rec_mae = np.abs(res[c]).mean(axis=(2, 3))  # (18, records)
        mpat[c] = np.stack([np.bincount(codes, weights=r) / counts for r in rec_mae])
        single[c] = mpat[c].mean(axis=1)  # patient-macro MAE per seed
    rng = np.random.default_rng(0)
    C = rng.multinomial(len(counts), [1 / len(counts)] * len(counts), size=N_BOOT).astype(float)
    _G.update(res=res, mpat=mpat, codes=codes, counts=counts, C=C)
    scen = {
        "A vs A (null)": ("A", "A"),
        "A vs S (moderate)": ("A", "S"),
        "A vs R (small)": ("A", "R"),
    }
    jobs = [(name, cx, cy, d) for name, (cx, cy) in scen.items() for d in range(N_DRAWS)]
    draws_path = CACHE / "e3_draws.csv"
    if draws_path.exists():
        draws = pd.read_csv(draws_path)
    else:
        with Pool(30, initializer=_one_thread) as p:
            draws = pd.DataFrame([r for out in p.map(_draw, jobs, chunksize=4) for r in out])
        draws.to_csv(draws_path, index=False)
    runs_per_arm = {"P0-3": 3, "P0-6": 6, "P1": 6, "P2-3": 3, "P2-6": 6}
    truth = {"A vs A (null)": 0.0, "A vs S (moderate)": float((single["A"] - single["S"]).mean()),
             "A vs R (small)": float((single["A"] - single["R"]).mean())}  # fmt: skip
    prot = []
    for (sc, pr), g in draws.groupby(["scenario", "protocol"]):
        true_sign = np.sign(truth[sc])
        prot.append({"scenario": sc, "protocol": pr, "runs_per_arm": runs_per_arm[pr],
                     "claim_rate": g.claim.mean(),
                     "correct_sign_rate": (g.sign == true_sign).mean() if true_sign else np.nan,
                     "wrong_sign_rate": ((g.sign != 0) & (g.sign != true_sign)).mean() if true_sign else np.nan,
                     "mean_diff_uV": 1000 * g["diff"].mean(), "sd_diff_uV": 1000 * g["diff"].std(ddof=1),
                     "mean_ci_width_uV": 1000 * (g.hi - g.lo).mean()})  # fmt: skip
    protocols = pd.DataFrame(prot)
    protocols.to_csv(OUT / "table_protocols.csv", index=False, float_format="%.4f")
    fpr = protocols[protocols.scenario == "A vs A (null)"].set_index("protocol").claim_rate
    power = (
        protocols[protocols.scenario == "A vs S (moderate)"].set_index("protocol").correct_sign_rate
    )
    ok = [p for p in fpr.index if fpr[p] <= 0.05]
    chosen = None
    if ok:
        best = max(power[p] for p in ok)
        near = [p for p in ok if power[p] >= best - 0.05]
        chosen = min(near, key=lambda p: (runs_per_arm[p], -power[p]))
    S.update({
        "E3_seed_sd_single_model_mae_uV": {c: 1000 * float(np.std(v, ddof=1)) for c, v in single.items()},
        "E3_mean_single_model_mae": {c: float(np.mean(v)) for c, v in single.items()},
        "E3_true_effect_uV": {k: 1000 * v for k, v in truth.items()},
        "E3_fpr": fpr.to_dict(), "E3_power_moderate": power.to_dict(),
        "E3_power_small": protocols[protocols.scenario == "A vs R (small)"].set_index("protocol").correct_sign_rate.to_dict(),
        "H8_4_fpr_P0_3": float(fpr["P0-3"]), "H8_4_pass": bool(fpr["P0-3"] > 0.05),
        "D8_3_selected_protocol": chosen,
    })  # fmt: skip
    S["E4_retro_check"] = retro_check(meta, fold_rows[9])
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
    print(json.dumps(S, indent=2, default=str))
    figures(e1, pm, taus, draws, protocols)


def retro_check(meta: pd.DataFrame, rows9: np.ndarray) -> dict:
    """Phase 5 adoption decisions re-tested with the selected protocol (compare_runs, full
    suite on single-model scores)."""
    from ptbxl_forecast.stats import compare_runs

    paths = {
        "decomposition": [run_path(spec(9, DEC, 1.0, 0, s)) for s in S18],
        "raw": [run_path(spec(9, "none", 1.0, 0, s)) for s in S18],
        "gaussian_noise": [P5 / f"cnn_aug_gaussian_noise_s{s}.npz" for s in S18[:6]],
    }
    sc_path = CACHE / "run_scores.pkl"
    if sc_path.exists():
        runs = pickle.load(sc_path.open("rb"))
    else:
        flat = [(name, p) for name, ps in paths.items() for p in ps]
        with Pool(30, initializer=_one_thread) as pool:
            dfs = pool.map(_score, [(9, np.load(p)["pred"].astype(np.float64)) for _, p in flat])
        runs = {
            name: [d for (n, _), d in zip(flat, dfs, strict=True) if n == name] for name in paths
        }
        pickle.dump(runs, sc_path.open("wb"))
    pid = meta.patient_id.to_numpy()[rows9]
    tests = {
        "decomposition vs raw, 18 vs 18 runs": (runs["decomposition"], runs["raw"]),
        "decomposition vs raw, 6 vs 6 (Phase 5 seeds)": (
            runs["decomposition"][:6],
            runs["raw"][:6],
        ),
        "gaussian noise vs raw, 6 vs 6 (Phase 5 seeds)": (runs["gaussian_noise"], runs["raw"][:6]),
    }
    out = {}
    for name, (a, b) in tests.items():
        c = compare_runs(a, b, pid)
        out[name] = {m: {k: round(c[m][k] * (1 if m == "beat_timing_ms" else 1000), 3) for k in ("diff", "lo", "hi")}
                     for m in SUITE} | {"beats_reference": c["beats_reference"]}  # fmt: skip
    return out


def figures(e1, pm, taus, draws, protocols) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    y = np.arange(len(e1))
    ax[0].barh(y - 0.2, e1.smd_f9_vs_validated, 0.4, label="fold 9 vs human-validated (folds 1–8)")
    ax[0].barh(y + 0.2, e1.smd_f9_vs_other, 0.4, label="fold 9 vs other records (folds 1–8)")
    for v in (-0.1, 0.1):
        ax[0].axvline(v, color="grey", ls=":", lw=0.8)
    ax[0].set_yticks(y, e1.index, fontsize=8)
    ax[0].set_xlim(-1.6, 1.6)
    for i, v in enumerate(e1.smd_f9_vs_other):
        if abs(v) > 1.6:
            ax[0].annotate(f"{v:.1f} (clipped)", (np.sign(v) * 1.55, i + 0.2), fontsize=7,
                           ha="left" if v < 0 else "right", va="center")  # fmt: skip
    ax[0].set(xlabel="standardized mean difference", title="E1: what makes fold 9 different?")
    ax[0].legend(fontsize=7)
    for m in pm.columns:
        ax[1].plot(pm.index, pm[m], marker="o", label=m)
    ax[1].set(xlabel="held-out official fold", ylabel="patient-macro MAE (mV)",
              title="E2: MAE by held-out fold (CNN: 3-seed ensembles)")  # fmt: skip
    ax[1].set_xticks(pm.index)
    ax[1].legend(fontsize=7, ncol=2)
    for k, t in taus.items():
        ax[1].annotate(f"τ={t:.2f}", (k, ax[1].get_ylim()[0]), fontsize=7, ha="center", va="bottom")
    p = protocols.pivot(index="protocol", columns="scenario", values="claim_rate")
    cs = protocols.pivot(index="protocol", columns="scenario", values="correct_sign_rate")
    x = np.arange(len(p))
    ax[2].bar(
        x - 0.27, p["A vs A (null)"], 0.27, label="false-positive rate (A vs A)", color="#d62728"
    )
    ax[2].bar(x, cs["A vs R (small)"], 0.27, label="power, small effect (A vs R)", color="#9ecae1")
    ax[2].bar(
        x + 0.27,
        cs["A vs S (moderate)"],
        0.27,
        label="power, moderate effect (A vs S)",
        color="#2171b5",
    )
    ax[2].axhline(0.05, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(x, p.index)
    ax[2].set(ylim=(0, 1), ylabel="share of 500 draws", title="E3: comparison protocols")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_validation.png", dpi=140)

    fig, ax = plt.subplots(figsize=(7, 4))
    null = draws[(draws.scenario == "A vs A (null)") & (draws.protocol == "P0-3")]
    ax.hist(1000 * null["diff"], bins=40, color="#999999")
    half = 500 * (null.hi - null.lo).mean()
    for v in (-half, half):
        ax.axvline(v, color="#d62728", ls="--")
    ax.set(xlabel="ΔMAE between two 3-seed ensembles of the same configuration (µV)",
           ylabel="draws", title="Run-to-run spread vs the patient-bootstrap CI (dashed: mean half-width)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig2_null_spread.png", dpi=140)


if __name__ == "__main__":
    main()
