"""Phase 10: training objectives for CNN-LSTM (PREREGISTRATION.md).

    python research/phase10_objectives.py --stage sn4      # cache causal sn4 forecasts, check vs Phase 6
    python research/phase10_objectives.py --stage A        # loss-weight pilot (internal holdout) + L1 rerun check
    python research/phase10_objectives.py --stage B        # fold-9 screening, 6 seeds per objective
    python research/phase10_objectives.py --stage C        # 9-fold CV of the finalists (D8.6)
    python research/phase10_objectives.py --stage report

Runs are cached; stages read the earlier stages' outputs. The L1 reference is the Phase 9
CNN-LSTM (same seeds, splits and learning rate).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402
import phase8_validation as p8  # noqa: E402
import phase9_models as p9  # noqa: E402

OUT = Path("reports/phase10_objectives")
CACHE = Path(".cache/data/phase10")
SN4 = CACHE / "sn4_dev.npy"
H, F, THREADS, WORKERS = 700, 290, 4, 10
LR, MAX_EPOCHS = 1e-3, 100  # CNN-LSTM learning rate (D9.2) and epoch cap (D9.1)
AUX_OBJ = ("derivative", "spectral")
OBJECTIVES = (*AUX_OBJ, "sn4_gated")
LAMS = (0.3, 1.0, 3.0)
MAE_TOLERANCE = 1.02  # pilot: holdout MAE at most 2% above the L1 run
SEEDS6, CV_SEEDS, SUITE = p9.SEEDS6, p9.CV_SEEDS, p9.SUITE
SCALE_STRIDE = 7  # aux_scale uses every 7th training record


# ------------------------------------------------------------------ sn4 forecasts
def _sn4(x: np.ndarray) -> np.ndarray:
    hist = x[:H].astype(np.float64)
    period, _ = p0.estimate_period(hist, p0.rpeaks(hist[:, 1]))
    return p0.seasonal(hist, period, F, p0.K_CYCLES)


def stage_sn4() -> None:
    """sn4 from each record's history only (as Phase 6), checked against Phase 6 MAE."""
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset

    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    if not SN4.exists():
        with Pool(38) as pool:
            out = pool.map(_sn4, [sig[i, :H] for i in range(len(sig))], chunksize=64)
        np.save(SN4, np.stack(out).astype(np.float32))
    sn4 = np.load(SN4)
    p6 = pd.read_parquet(".cache/data/phase6_scores.parquet")
    p6 = p6[p6.probe == "sn4 (repeat cycles)"].set_index("row")
    rows = p6.index.to_numpy()
    mae = np.abs(sn4[rows].astype(np.float64) - sig[rows, H : H + F]).mean(axis=(1, 2))
    err = float(np.max(np.abs(mae - p6.mae.to_numpy())))
    json.dump({"records": len(sn4), "phase6_rows_checked": len(rows), "max_abs_mae_diff_mV": err},
              (OUT / "sn4_check.json").open("w"), indent=2)  # fmt: skip
    print("sn4 cached:", sn4.shape, "max |ΔMAE| vs Phase 6:", err)
    assert err < 1e-5, err


# ------------------------------------------------------------------ runs
def run_path(obj: str, lam: float, k: int, s: int, verify: bool = False) -> Path:
    if obj == "l1" and not verify:
        return p9.run_path("cnn_lstm", LR, False, k, s)
    tag = "verify_" if verify else ""
    return CACHE / f"{tag}{obj}_lam{lam:g}_k{k}_s{s}.npz"


def spec(obj, lam, k, s, verify=False) -> dict:
    return {"obj": obj, "lam": lam, "k": k, "s": s, "path": str(run_path(obj, lam, k, s, verify))}


def job(sp: dict) -> str:
    path = Path(sp["path"])
    if path.exists():
        return str(path)
    import torch

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.losses import aux_scale, make_objective
    from ptbxl_forecast.models import MODELS, GatedBaselineCNNLSTM
    from ptbxl_forecast.preprocessing import BaselineConditioned, Pipeline, pack_baseline
    from ptbxl_forecast.training import predict, train_forecaster

    torch.set_num_threads(THREADS)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    rows, rows_ho, rows_ev, sub = p8.split_rows(meta, sp["k"], 1.0, 0)
    p = cfg.preprocessing
    pipe = Pipeline(p.filter, p.scaler, False)
    data = sig
    extra = {}
    if sp["obj"] == "sn4_gated":
        data = pack_baseline(sig, np.load(SN4), H)
        pipe = BaselineConditioned(pipe)
        model = GatedBaselineCNNLSTM(pipe.in_channels, horizon=F)
        loss = make_objective("l1")
    else:
        model = MODELS["cnn_lstm"](in_channels=pipe.in_channels, out_leads=12, horizon=F)
        scale = 1.0
        if sp["obj"] in AUX_OBJ:
            r = rows[::SCALE_STRIDE]
            scale = aux_scale(
                sp["obj"], sig[r, H : H + F].astype(np.float64), sig[r, :H].astype(np.float64)
            )
        loss = make_objective(sp["obj"], sp["lam"], scale)
        extra["aux_scale"] = scale
    pipe.fit(data[rows, :H].astype(np.float64), sub.index)
    info = train_forecaster(model, pipe, data[rows], data[rows_ho], H, F, seed=sp["s"], lr=LR,
                            epochs=MAX_EPOCHS, threads=THREADS, loss=loss)  # fmt: skip
    pred = predict(model, pipe, data[rows_ev, :H].astype(np.float64))
    info.update({k: v for k, v in sp.items() if k != "path"}, **extra,
                params=sum(q.numel() for q in model.parameters()), n_records=len(rows))  # fmt: skip
    out = {"pred": pred.astype(np.float32), "info": np.array(info, dtype=object)}
    if sp["k"] == 9 and sp["s"] == SEEDS6[0]:  # pilot runs: also internal-holdout forecasts
        out["pred_holdout"] = predict(model, pipe, data[rows_ho, :H].astype(np.float64)).astype(
            np.float32
        )
    np.savez(path, **out)
    return str(path)


def train(specs: list[dict]) -> float:
    todo = [s for s in specs if not Path(s["path"]).exists()]
    todo.sort(key=lambda s: (s["obj"] != "spectral", s["k"], s["s"]))
    t0 = time.perf_counter()
    if todo:
        print(f"training {len(todo)} runs on {WORKERS} workers", flush=True)
        with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
            for p in pool.map(job, todo):
                print(time.strftime("%H:%M"), "done", Path(p).name, flush=True)
    return (time.perf_counter() - t0) / 60


# ------------------------------------------------------------------ stage A
_HO: dict = {}


def _ho_peaks(i: int) -> np.ndarray:
    from ptbxl_forecast import metrics as M

    s = _HO["sig"]
    return M.detect_beats(s[i, :H].astype(np.float64), s[i, H : H + F].astype(np.float64))


def _ho_score(pred: np.ndarray) -> pd.DataFrame:
    from ptbxl_forecast import metrics as M

    s = _HO["sig"][_HO["rows"]].astype(np.float64)
    return M.evaluate(s[:, :H], s[:, H : H + F], pred.astype(np.float64), true_peaks=_HO["peaks"])


def stage_a() -> None:
    """Loss weight per auxiliary objective on the internal holdout (fold 9 unused), plus
    a bitwise check that the modified training loop reproduces Phase 9's L1 run."""
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import patient_macro

    s0 = SEEDS6[0]
    specs = [spec(o, lam, 9, s0) for o in AUX_OBJ for lam in LAMS]
    specs += [spec("l1", 0.0, 9, s0, verify=True), *(spec("sn4_gated", 0.0, 9, s) for s in SEEDS6)]
    minutes = train(specs)
    verify = np.array_equal(np.load(run_path("l1", 0.0, 9, s0, True))["pred"],
                            np.load(run_path("l1", 0.0, 9, s0))["pred"])  # fmt: skip
    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    _, rows_ho, _, _ = p8.split_rows(meta, 9, 1.0, 0)
    pid = meta.patient_id.to_numpy()[rows_ho]
    _HO.update(sig=sig, rows=rows_ho)
    pk = CACHE / "holdout_k9_true_peaks.pkl"
    if not pk.exists():
        with Pool(38) as pool:
            pickle.dump(pool.map(_ho_peaks, rows_ho, chunksize=16), pk.open("wb"))
    _HO["peaks"] = pickle.load(pk.open("rb"))
    runs = {("l1", 0.0): run_path("l1", 0.0, 9, s0, True)}
    runs.update({(o, lam): run_path(o, lam, 9, s0) for o in AUX_OBJ for lam in LAMS})
    runs[("sn4_gated", 0.0)] = run_path("sn4_gated", 0.0, 9, s0)
    keys = list(runs)
    with Pool(len(keys)) as pool:
        scores = pool.map(_ho_score, [np.load(runs[k])["pred_holdout"] for k in keys])
    rows = []
    for (o, lam), df in zip(keys, scores, strict=True):
        i = p9.info(runs[(o, lam)])
        rows.append({"objective": o, "lam": lam, **{m: patient_macro(df[m], pid) for m in SUITE},
                     "best_epoch": i["best_epoch"], "epochs_run": i["epochs_run"],
                     "minutes": i["seconds"] / 60, "aux_scale": i.get("aux_scale")})  # fmt: skip
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_stage_a_pilot.csv", index=False, float_format="%.6g")
    ref = t[t.objective == "l1"].iloc[0]
    choice = {}
    for o in AUX_OBJ:
        g = t[t.objective == o]
        ok = g[g.mae <= MAE_TOLERANCE * ref.mae]
        choice[o] = float(
            (ok.sort_values("template_mae") if len(ok) else g.sort_values("mae")).lam.iloc[0]
        )
    choice["sn4_gated"] = 0.0
    json.dump({"lam": choice, "l1_rerun_bitwise_identical": bool(verify), "minutes": minutes,
               "l1_holdout_mae": float(ref.mae)}, (OUT / "stage_a.json").open("w"), indent=2)  # fmt: skip
    pd.set_option("display.width", 250)
    print(t.round(5).to_string(), "\n", choice, "bitwise:", verify)


def lam_of() -> dict:
    return json.load((OUT / "stage_a.json").open())["lam"]


# ------------------------------------------------------------------ stage B
def stage_b() -> None:
    from ptbxl_forecast.stats import compare_runs

    lam = lam_of()
    minutes = train([spec(o, lam[o], 9, s) for o in OBJECTIVES for s in SEEDS6])
    meta, sig, fold_rows, pids, base = p9.setup_scoring()
    pid = pids[9]
    arms = {"l1": 0.0, **lam}
    paths = {o: [str(run_path(o, arms[o], 9, s)) for s in SEEDS6] for o in arms}
    jobs = {(o, s): (9, [p]) for o in arms for s, p in zip(SEEDS6, paths[o], strict=True)}
    jobs.update({(o, "ens"): (9, paths[o]) for o in arms})
    sc = p9.score_many(jobs, CACHE / "scores_stage_b.pkl")
    rows, cmps = [], []
    for o in arms:
        runs = [sc[(o, s)] for s in SEEDS6]
        single = pd.DataFrame([p9.macro(r, pid) for r in runs])
        infos = [p9.info(p) for p in paths[o]]
        rows.append({"objective": o, "lam": arms[o], "params": infos[0].get("params"),
                     **{f"{m}_seed_mean": single[m].mean() for m in SUITE},
                     "share_le_50ms_seed_mean": single.share_timing_le_50ms.mean(),
                     "mae_seed_sd": single.mae.std(ddof=1),
                     **{f"ens_{m}": v for m, v in p9.macro(sc[(o, "ens")], pid).items()},
                     "best_epoch_mean": np.mean([i["best_epoch"] for i in infos]),
                     "capped_runs": int(sum(i["epochs_run"] == MAX_EPOCHS for i in infos)),
                     "minutes_mean": np.mean([i["seconds"] for i in infos]) / 60})  # fmt: skip
        refs = {"flat": [base[("flat", 9)]], "sn4": [base[("sn4", 9)]]}
        if o != "l1":
            refs["l1"] = [sc[("l1", s)] for s in SEEDS6]
        for rname, ref in refs.items():
            cmps.append({"objective": o, "vs": rname, **p9.cmp_row(compare_runs(runs, ref, pid))})
    table, comps = pd.DataFrame(rows), pd.DataFrame(cmps)
    table.to_csv(OUT / "table_stage_b_screening.csv", index=False, float_format="%.6g")
    comps.to_csv(OUT / "table_stage_b_comparisons.csv", index=False, float_format="%.5g")
    c = comps.set_index(["objective", "vs"])
    eligible = [o for o in OBJECTIVES if c.loc[(o, "flat"), "beats"]]
    order = table.set_index("objective").loc[eligible].sort_values("mae_seed_mean").index.tolist()
    S = {
        "minutes": minutes,
        "eligible_beats_flat": eligible,
        "finalists": order[:2],
        "template_better_than_l1": [o for o in OBJECTIVES if c.loc[(o, "l1"), "d_template_mae_hi"] < 0],
        "beats_l1": [o for o in OBJECTIVES if c.loc[(o, "l1"), "beats"]],
        "beats_sn4": [o for o in OBJECTIVES if c.loc[(o, "sn4"), "beats"]],
    }  # fmt: skip
    json.dump(S, (OUT / "stage_b.json").open("w"), indent=2)
    pd.set_option("display.width", 250)
    print(
        table.round(5).to_string(), "\n", comps.round(3).to_string(), "\n", json.dumps(S, indent=2)
    )


# ------------------------------------------------------------------ stage C
def stage_c() -> None:
    """Finalists vs flat, sn4 and L1 CNN-LSTM across folds 1-9 (3-seed ensembles per fold)."""
    from scipy.stats import ttest_1samp

    lam, sb = lam_of(), json.load((OUT / "stage_b.json").open())
    fin = sb["finalists"]
    if not fin:
        json.dump({"finalists": [], "note": "no objective beat flat on fold 9; no CV run"},
                  (OUT / "stage_c.json").open("w"), indent=2)  # fmt: skip
        print("no finalists")
        return
    arms = {"l1": 0.0, **{o: lam[o] for o in fin}}
    minutes = train([spec(o, lam[o], k, s) for o in fin for k in range(1, 10) for s in CV_SEEDS])
    meta, sig, fold_rows, pids, base = p9.setup_scoring()
    jobs = {(o, k): (k, [str(run_path(o, arms[o], k, s)) for s in CV_SEEDS])
            for o in arms for k in range(1, 10)}  # fmt: skip
    sc = p9.score_many(jobs, CACHE / "scores_stage_c.pkl")
    cv = pd.DataFrame([{"arm": o, "fold": k, **p9.macro(sc[(o, k)], pids[k])}
                       for o in arms for k in range(1, 10)])  # fmt: skip
    for name in ("flat", "sn4"):
        cv = pd.concat([cv, pd.DataFrame([{"arm": name, "fold": k, **p9.macro(base[(name, k)], pids[k])}
                                          for k in range(1, 10)])])  # fmt: skip
    cv.to_csv(OUT / "table_stage_c_cv.csv", index=False, float_format="%.6g")
    by = {a: g.set_index("fold") for a, g in cv.groupby("arm")}
    rows = []
    for o in fin:
        for ref in ("flat", "sn4", "l1"):
            row = {"arm": o, "vs": ref}
            for m in SUITE:
                d = by[o][m] - by[ref][m]
                row[f"mean_d_{m}"], row[f"p_{m}"] = d.mean(), float(ttest_1samp(d, 0).pvalue)
                row[f"folds_better_{m}"] = int((d < 0).sum())
            row["beats"] = bool(row["mean_d_mae"] < 0 and row["p_mae"] < 0.05 and all(
                not (row[f"mean_d_{g}"] > 0 and row[f"p_{g}"] < 0.05) for g in SUITE[1:]))  # fmt: skip
            rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "table_stage_c_pairs.csv", index=False, float_format="%.5g")
    p = pairs.set_index(["arm", "vs"])
    S = {"minutes": minutes, "finalists": fin,
         "cv_mean": cv.groupby("arm")[list(SUITE)].mean().to_dict(orient="index"),
         "confirmed_beats_flat": [o for o in fin if p.loc[(o, "flat"), "beats"]],
         "confirmed_beats_l1": [o for o in fin if p.loc[(o, "l1"), "beats"]],
         "confirmed_beats_sn4": [o for o in fin if p.loc[(o, "sn4"), "beats"]]}  # fmt: skip
    json.dump(S, (OUT / "stage_c.json").open("w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    print(pairs.round(5).to_string(), "\n", json.dumps(S, indent=2, default=float))


# ------------------------------------------------------------------ report figures
def report() -> None:
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset

    lam = {"l1": 0.0, **lam_of()}
    t = pd.read_csv(OUT / "table_stage_b_screening.csv")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.arange(len(t))
    for a, (m, lab) in zip(ax, (("mae", "MAE (mV)"), ("template_mae", "template MAE (mV)"),
                                ("beat_timing_ms", "beat timing (ms)")), strict=True):  # fmt: skip
        a.bar(x, t[f"{m}_seed_mean"], color="#2171b5")
        a.scatter(x, t[f"ens_{m}"], color="k", zorder=3, label="6-run ensemble")
        a.set_xticks(x, t.objective)
        a.set(ylabel=lab, title=f"Fold 9: {lab}")
    ref = pd.read_csv("reports/phase9_models/table_stage_c_cv.csv")
    for a, m in zip(ax, ("mae", "template_mae", "beat_timing_ms"), strict=True):
        for name, c, ls in (("flat", "grey", "--"), ("sn4", "#2ca02c", ":")):
            a.axhline(
                ref[(ref.family == name) & (ref.fold == 9)][m].iloc[0], color=c, ls=ls, label=name
            )
        a.legend(fontsize=8)
    ax[0].set_ylim(0.08, 0.125)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_objectives.png", dpi=140)

    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    rows9 = np.flatnonzero(meta.strat_fold == 9)
    pick = np.sort(np.random.default_rng(0).choice(len(rows9), 4, replace=False))  # as Phase 9
    preds = {o: np.mean([np.load(run_path(o, lam[o], 9, s))["pred"][pick] for s in SEEDS6], axis=0)
             for o in lam}  # fmt: skip
    sn4 = np.load(SN4)[rows9[pick]]
    t_ = np.arange(990) / 100
    colors = {
        "l1": "#999999",
        "derivative": "#ff7f0e",
        "spectral": "#9467bd",
        "sn4_gated": "#d62728",
    }
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for a, i, r in zip(ax, range(4), rows9[pick], strict=True):
        a.plot(t_[500:], sig[r, 500:990, 1], color="k", lw=1.1, label="truth (lead II)")
        a.plot(t_[H:990], sn4[i, :, 1], color="#2ca02c", lw=0.8, ls=":", label="sn4")
        for o, c in colors.items():
            a.plot(t_[H:990], preds[o][i, :, 1], color=c, lw=1, label=f"{o} (6-run ensemble)")
        a.axvline(7, color="grey", ls="--", lw=0.8)
        a.set(ylabel="mV", title=f"ecg_id {meta.index[r]}")
    ax[0].legend(fontsize=8, ncol=3)
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_example_forecasts.png", dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sn4", "A", "B", "C", "report"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"sn4": stage_sn4, "A": stage_a, "B": stage_b, "C": stage_c, "report": report}[args.stage]()


if __name__ == "__main__":
    main()
