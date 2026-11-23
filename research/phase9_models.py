"""Phase 9: model families under a fixed L1 objective (PREREGISTRATION.md).

    python research/phase9_models.py --stage A   # learning rate per family (internal holdout)
    python research/phase9_models.py --stage B   # fold-9 screening, 6 seeds per family
    python research/phase9_models.py --stage C   # 9-fold CV of the shortlist (D8.6)
    python research/phase9_models.py --stage D   # 8-lead output on the best family
    python research/phase9_models.py --stage report

Runs are cached; stages read the earlier stages' outputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
import phase8_validation as p8  # noqa: E402

OUT = Path("reports/phase9_models")
CACHE = Path(".cache/data/phase9")
H, F, THREADS, WORKERS = 700, 290, 4, 9
MAX_EPOCHS = 100  # amended from 40 after stage A (PREREGISTRATION.md, amendment 1)
FAMILIES = ("lstm", "cnn_lstm", "patchtst", "transformer")
ALL = ("cnn", *FAMILIES)
LRS = (1e-3, 3e-4)
SEEDS6 = p8.S18[:6]
CV_SEEDS = p8.CV_SEEDS
SUITE = ("mae", "dtw_mae", "template_mae", "beat_timing_ms")
COST = {"patchtst": 5, "transformer": 4.5, "cnn_lstm": 3, "lstm": 2.5, "cnn": 0.5}


# ------------------------------------------------------------------ runs
def run_path(fam: str, lr: float, leads8: bool, k: int, s: int, verify: bool = False) -> Path:
    """The CNN reference (lr 1e-3, 12 leads) reuses the Phase 5/8 caches (bitwise
    reproducible, checked by a verification rerun)."""
    if fam == "cnn" and lr == 1e-3 and not leads8 and not verify:
        cached = p8.run_path(p8.spec(k, p8.DEC, 1.0, 0, s))
        if cached.exists():
            return cached
    tag = "verify_" if verify else ""
    return CACHE / f"{tag}{fam}_lr{lr:g}_{'8' if leads8 else '12'}lead_k{k}_s{s}.npz"


def spec(fam, lr, leads8, k, s, verify=False) -> dict:
    return {"fam": fam, "lr": lr, "leads8": leads8, "k": k, "s": s,
            "path": str(run_path(fam, lr, leads8, k, s, verify))}  # fmt: skip


def job(sp: dict) -> str:
    path = Path(sp["path"])
    if path.exists():
        return str(path)
    import torch

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.models import MODELS
    from ptbxl_forecast.preprocessing import Pipeline
    from ptbxl_forecast.training import predict, train_forecaster

    torch.set_num_threads(THREADS)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    rows, rows_ho, rows_ev, sub = p8.split_rows(meta, sp["k"], 1.0, 0)
    p = cfg.preprocessing
    pipe = Pipeline(p.filter, p.scaler, sp["leads8"]).fit(
        sig[rows, :H].astype(np.float64), sub.index
    )
    model = MODELS[sp["fam"]](in_channels=pipe.in_channels, out_leads=pipe.out_leads, horizon=F)
    info = train_forecaster(model, pipe, sig[rows], sig[rows_ho], H, F, seed=sp["s"], lr=sp["lr"],
                            epochs=MAX_EPOCHS, threads=THREADS)  # fmt: skip
    pred = predict(model, pipe, sig[rows_ev, :H].astype(np.float64))
    info.update({k: v for k, v in sp.items() if k != "path"},
                params=sum(q.numel() for q in model.parameters()), n_records=len(rows))  # fmt: skip
    np.savez(path, pred=pred.astype(np.float32), info=np.array(info, dtype=object))
    return str(path)


def train(specs: list[dict]) -> float:
    todo = [s for s in specs if not Path(s["path"]).exists()]
    todo.sort(key=lambda s: -COST[s["fam"]])
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
            for p in pool.map(job, todo):
                print("done", Path(p).name, flush=True)
    return (time.perf_counter() - t0) / 60


def info(path) -> dict:
    return np.load(path, allow_pickle=True)["info"].item()


# ------------------------------------------------------------------ scoring
def _score_paths(job_: tuple) -> pd.DataFrame:
    k, paths = job_
    pred = np.mean([np.load(p)["pred"].astype(np.float64) for p in paths], axis=0)
    return p8._score((k, pred))


def setup_scoring():
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset

    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    fold_rows = {k: np.flatnonzero(meta.strat_fold == k) for k in range(1, 10)}
    p8._G.update(sig=sig, fold_rows=fold_rows,
                 peaks=pickle.load((p8.CACHE / "dev_true_peaks.pkl").open("rb")))  # fmt: skip
    p6 = pd.read_parquet(".cache/data/phase6_scores.parquet")
    p6["ecg_id"] = meta.index.to_numpy()[p6.row]
    base = {}
    for name, probe in (("flat", "median (flat)"), ("sn4", "sn4 (repeat cycles)")):
        ref = p6[p6.probe == probe].set_index("ecg_id")
        for k in range(1, 10):
            base[(name, k)] = ref.loc[meta.index[fold_rows[k]]].reset_index(drop=True)
    pids = {k: meta.patient_id.to_numpy()[fold_rows[k]] for k in range(1, 10)}
    return meta, sig, fold_rows, pids, base


def score_many(jobs: dict, cache: Path) -> dict:
    """jobs: key -> (fold, [paths]); cached by key."""
    done = pickle.load(cache.open("rb")) if cache.exists() else {}
    todo = [k for k in jobs if k not in done]
    if todo:
        with Pool(min(30, len(todo))) as pool:
            for key, df in zip(todo, pool.map(_score_paths, [jobs[k] for k in todo]), strict=True):
                done[key] = df
        pickle.dump(done, cache.open("wb"))
    return {k: done[k] for k in jobs}


def macro(df, pid) -> dict:
    from ptbxl_forecast.stats import patient_macro

    out = {m: patient_macro(df[m], pid) for m in SUITE}
    out["share_timing_le_50ms"] = float((df.beat_timing_ms <= 50).mean())
    return out


def cmp_row(c: dict) -> dict:
    row = {}
    for m in SUITE:
        scale = 1 if m == "beat_timing_ms" else 1000  # µV for mV metrics
        row.update({f"d_{m}": c[m]["diff"] * scale, f"d_{m}_lo": c[m]["lo"] * scale,
                    f"d_{m}_hi": c[m]["hi"] * scale})  # fmt: skip
    row["beats"] = c["beats_reference"]
    return row


# ------------------------------------------------------------------ stages
def stage_a() -> None:
    specs = [spec(f, lr, False, 9, 42) for f in ALL for lr in LRS]
    specs.append(spec("cnn", 1e-3, False, 9, 42, verify=True))
    minutes = train(specs)
    verify = np.array_equal(np.load(run_path("cnn", 1e-3, False, 9, 42, True))["pred"],
                            np.load(run_path("cnn", 1e-3, False, 9, 42))["pred"])  # fmt: skip
    rows = []
    for f in ALL:
        for lr in LRS:
            i = info(run_path(f, lr, False, 9, 42))
            rows.append({"family": f, "lr": lr, "best_holdout_l1": i["best_holdout_l1"],
                         "best_epoch": i["best_epoch"], "epochs_run": i["epochs_run"],
                         "minutes": i["seconds"] / 60, "params": i.get("params")})  # fmt: skip
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_stage_a_lr.csv", index=False, float_format="%.6g")
    choice = {f: float(g.sort_values("best_holdout_l1").lr.iloc[0]) for f, g in t.groupby("family")}
    json.dump({"lr": choice, "cnn_reuse_verified_bitwise": bool(verify), "minutes": minutes},
              (OUT / "stage_a.json").open("w"), indent=2)  # fmt: skip
    print(t.to_string(), "\n", choice, "verify:", verify)


def lr_of() -> dict:
    return json.load((OUT / "stage_a.json").open())["lr"]


def stage_b() -> None:
    from ptbxl_forecast.stats import compare_runs

    lr = lr_of()
    minutes = train([spec(f, lr[f], False, 9, s) for f in ALL for s in SEEDS6])
    meta, sig, fold_rows, pids, base = setup_scoring()
    pid = pids[9]
    paths = {f: [str(run_path(f, lr[f], False, 9, s)) for s in SEEDS6] for f in ALL}
    jobs = {(f, s): (9, [p]) for f in ALL for s, p in zip(SEEDS6, paths[f], strict=True)}
    jobs.update({(f, "ens"): (9, paths[f]) for f in ALL})
    sc = score_many(jobs, CACHE / "scores_stage_b.pkl")
    rows, cmps = [], []
    for f in ALL:
        runs = [sc[(f, s)] for s in SEEDS6]
        single = pd.DataFrame([macro(r, pid) for r in runs])
        infos = [info(p) for p in paths[f]]
        row = {"family": f, "lr": lr[f], "params": infos[0].get("params"),
               **{f"{m}_seed_mean": single[m].mean() for m in SUITE}, "mae_seed_sd": single.mae.std(ddof=1),
               **{f"ens_{m}": v for m, v in macro(sc[(f, "ens")], pid).items()},
               "best_epoch_mean": np.mean([i["best_epoch"] for i in infos]),
               "minutes_mean": np.mean([i["seconds"] for i in infos]) / 60}  # fmt: skip
        rows.append(row)
        refs = {"flat": [base[("flat", 9)]], "sn4": [base[("sn4", 9)]]}
        if f != "cnn":
            refs["cnn"] = [sc[("cnn", s)] for s in SEEDS6]
        for rname, ref in refs.items():
            cmps.append({"family": f, "vs": rname, **cmp_row(compare_runs(runs, ref, pid))})
    table, comps = pd.DataFrame(rows), pd.DataFrame(cmps)
    table.to_csv(OUT / "table_stage_b_screening.csv", index=False, float_format="%.6g")
    comps.to_csv(OUT / "table_stage_b_comparisons.csv", index=False, float_format="%.5g")
    new = table[table.family.isin(FAMILIES)].sort_values("mae_seed_mean")
    S = {
        "minutes": minutes,
        "shortlist": new.family.iloc[:2].tolist(),
        "best_family_fold9": table.sort_values("mae_seed_mean").family.iloc[0],
        "H9_1_any_beats_sn4": bool(comps[comps.vs == "sn4"].beats.any()),
        "H9_1_min_ens_timing_ms": float(table.ens_beat_timing_ms.min()),
        "H9_2_best_vs_cnn_rel": float(table.mae_seed_mean.min() / table.set_index("family").loc["cnn", "mae_seed_mean"] - 1),
    }  # fmt: skip
    S["H9_1_pass"] = (not S["H9_1_any_beats_sn4"]) and S["H9_1_min_ens_timing_ms"] > 150
    S["H9_2_pass"] = abs(S["H9_2_best_vs_cnn_rel"]) <= 0.02
    json.dump(S, (OUT / "stage_b.json").open("w"), indent=2)
    pd.set_option("display.width", 250)
    print(
        table.round(5).to_string(), "\n", comps.round(3).to_string(), "\n", json.dumps(S, indent=2)
    )


def stage_c() -> None:
    from scipy.stats import ttest_1samp

    from ptbxl_forecast.stats import compare_runs

    lr, sb = lr_of(), json.load((OUT / "stage_b.json").open())
    fams = ["cnn", *sb["shortlist"]]
    minutes = train(
        [spec(f, lr[f], False, k, s) for f in fams for k in range(1, 10) for s in CV_SEEDS]
    )
    meta, sig, fold_rows, pids, base = setup_scoring()
    jobs = {
        (f, k): (k, [str(run_path(f, lr[f], False, k, s)) for s in CV_SEEDS])
        for f in fams
        for k in range(1, 10)
    }
    sc = score_many(jobs, CACHE / "scores_stage_c.pkl")
    cv = pd.DataFrame(
        [{"family": f, "fold": k, **macro(sc[(f, k)], pids[k])} for f in fams for k in range(1, 10)]
    )
    for name in ("flat", "sn4"):
        cv = pd.concat([cv, pd.DataFrame([{"family": name, "fold": k, **macro(base[(name, k)], pids[k])}
                                          for k in range(1, 10)])])  # fmt: skip
    cv.to_csv(OUT / "table_stage_c_cv.csv", index=False, float_format="%.6g")
    b = pickle.load((CACHE / "scores_stage_b.pkl").open("rb"))
    pairs = []
    for a_, b_ in combinations(fams, 2):
        row = {"a": a_, "b": b_}
        f9 = compare_runs([b[(a_, s)] for s in SEEDS6], [b[(b_, s)] for s in SEEDS6], pids[9])
        row["fold9_a_beats_b"], row["fold9_b_beats_a"] = f9["beats_reference"], compare_runs(
            [b[(b_, s)] for s in SEEDS6], [b[(a_, s)] for s in SEEDS6], pids[9])["beats_reference"]  # fmt: skip
        for m in SUITE:
            d = cv[cv.family == a_].set_index("fold")[m] - cv[cv.family == b_].set_index("fold")[m]
            row[f"cv_mean_d_{m}"], row[f"cv_p_{m}"] = d.mean(), float(ttest_1samp(d, 0).pvalue)
        pairs.append(row)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "table_stage_c_pairs.csv", index=False, float_format="%.5g")

    def beats(x, y) -> bool:
        r = pairs[((pairs.a == x) & (pairs.b == y)) | ((pairs.a == y) & (pairs.b == x))].iloc[0]
        sign = 1 if r.a == x else -1
        fold9 = r.fold9_a_beats_b if r.a == x else r.fold9_b_beats_a
        cv_ok = sign * r.cv_mean_d_mae < 0 and r.cv_p_mae < 0.05
        gk_ok = all(
            not (sign * r[f"cv_mean_d_{g}"] > 0 and r[f"cv_p_{g}"] < 0.05) for g in SUITE[1:]
        )
        return bool(fold9 and cv_ok and gk_ok)

    winners = [x for x in fams if all(beats(x, y) for y in fams if y != x)]
    cv_mean = cv[cv.family.isin(fams)].groupby("family").mae.mean().sort_values()
    f9_rank = cv[(cv.family.isin(fams)) & (cv.fold == 9)].set_index("family").mae.sort_values()
    S = {"minutes": minutes, "families": fams, "cv_mean_mae": cv_mean.to_dict(),
         "fold9_mae_3seed_ens": f9_rank.to_dict(), "best_family_under_L1": winners[0] if winners else None,
         "carried_to_phase10": cv_mean.index[:2].tolist(),
         "H9_3_pass": cv_mean.index.tolist() == f9_rank.index.tolist()}  # fmt: skip
    json.dump(S, (OUT / "stage_c.json").open("w"), indent=2)
    print(pairs.round(5).to_string(), "\n", json.dumps(S, indent=2))


def stage_d() -> None:
    from phase5_cnn import limb_residual

    from ptbxl_forecast.stats import compare_runs

    lr, sb = lr_of(), json.load((OUT / "stage_b.json").open())
    fam = sb["best_family_fold9"]
    minutes = train([spec(fam, lr[fam], True, 9, s) for s in SEEDS6])
    meta, sig, fold_rows, pids, base = setup_scoring()
    p12 = [str(run_path(fam, lr[fam], False, 9, s)) for s in SEEDS6]
    p8l = [str(run_path(fam, lr[fam], True, 9, s)) for s in SEEDS6]
    jobs = {("8", s): (9, [p]) for s, p in zip(SEEDS6, p8l, strict=True)}
    jobs[("8", "ens")] = (9, p8l)
    sc = score_many(jobs, CACHE / "scores_stage_d.pkl")
    b = pickle.load((CACHE / "scores_stage_b.pkl").open("rb"))
    c = compare_runs([sc[("8", s)] for s in SEEDS6], [b[(fam, s)] for s in SEEDS6], pids[9])
    res = {}
    for name, ps in (("12-lead", p12), ("8-lead", p8l)):
        r = limb_residual(np.mean([np.load(p)["pred"] for p in ps], axis=0).astype(np.float64))
        res[name] = {"median_mV": float(np.median(r)), "share_gt_0.01": float((r > 0.01).mean())}
    row = cmp_row(c)
    S = {"family": fam, "minutes": minutes, "delta_8_minus_12": row, "limb_residual": res,
         "ens_8lead": macro(sc[("8", "ens")], pids[9]), "ens_12lead": macro(b[(fam, "ens")], pids[9]),
         "H9_4_pass": bool(row["d_mae_hi"] <= 0.5 and all(row[f"d_{g}_lo"] <= 0 for g in SUITE[1:]))}  # fmt: skip
    json.dump(S, (OUT / "stage_d.json").open("w"), indent=2, default=float)
    print(json.dumps(S, indent=2, default=float))


def report() -> None:
    t = pd.read_csv(OUT / "table_stage_b_screening.csv")
    cv = pd.read_csv(OUT / "table_stage_c_cv.csv")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(t))
    for a, (m, lab) in zip(
        ax[:2], (("mae", "MAE (mV)"), ("beat_timing_ms", "beat timing (ms)")), strict=True
    ):
        a.bar(
            x, t[f"{m}_seed_mean"], color=["#999999" if f == "cnn" else "#2171b5" for f in t.family]
        )
        a.scatter(x, t[f"ens_{m}"], color="k", zorder=3, label="6-run ensemble")
        a.set_xticks(x, t.family)
        a.set(ylabel=lab, title=f"Fold 9: {lab} (bars: single-model mean of 6 seeds)")
        a.legend(fontsize=8)
    lo = min(t.mae_seed_mean.min(), t.ens_mae.min())
    ax[0].set_ylim(lo * 0.98, t.mae_seed_mean.max() * 1.01)
    sn4 = cv[(cv.family == "sn4") & (cv.fold == 9)].beat_timing_ms.iloc[0]
    flat = cv[(cv.family == "flat") & (cv.fold == 9)].beat_timing_ms.iloc[0]
    ax[1].axhline(sn4, color="#2ca02c", ls=":", label="sn4")
    ax[1].axhline(flat, color="grey", ls="--", label="flat")
    ax[1].legend(fontsize=8)
    for f, g in cv.groupby("family"):
        ax[2].plot(g.fold, g.mae, marker="o", label=f)
    ax[2].set(
        xlabel="held-out official fold",
        ylabel="patient-macro MAE (mV)",
        title="9-fold CV (3-seed ensembles)",
    )
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_families.png", dpi=140)
    examples()


def examples() -> None:
    """Lead II of 6-run ensemble forecasts for 4 random fold-9 records (seeded choice)."""
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset

    lr = lr_of()
    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    rows9 = np.flatnonzero(meta.strat_fold == 9)
    pick = np.sort(np.random.default_rng(0).choice(len(rows9), 4, replace=False))
    preds = {f: np.mean([np.load(run_path(f, lr[f], False, 9, s))["pred"][pick] for s in SEEDS6], axis=0)
             for f in ("cnn", "cnn_lstm", "transformer")}  # fmt: skip
    t = np.arange(990) / 100
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for a, i, r in zip(ax, range(4), rows9[pick], strict=True):
        a.plot(t[500:], sig[r, 500:990, 1], color="k", lw=1.1, label="truth (lead II)")
        for f, c in (("cnn", "#999999"), ("cnn_lstm", "#d62728"), ("transformer", "#2171b5")):
            a.plot(t[H:990], preds[f][i, :, 1], color=c, lw=1, label=f"{f} (6-run ensemble)")
        a.axvline(7, color="grey", ls="--", lw=0.8)
        a.set(ylabel="mV", title=f"ecg_id {meta.index[r]}")
    ax[0].legend(fontsize=8, ncol=4)
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_example_forecasts.png", dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C", "D", "report"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d, "report": report}[args.stage]()


if __name__ == "__main__":
    main()
