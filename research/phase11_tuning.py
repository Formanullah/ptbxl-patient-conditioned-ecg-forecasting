"""Phase 11: hyperparameters of the sn4-gated CNN-LSTM (PREREGISTRATION.md).

    python research/phase11_tuning.py --stage A   # one-change-at-a-time grid, internal holdout, seed 42
    python research/phase11_tuning.py --stage B   # fold-9 confirmation of the top 2 (6 seeds, compare_runs)

``train_config`` is the shared trainer for Phases 11-18: a configuration dict, a CV fold
``k`` (``k = 10`` trains on all of folds 1-9 and returns the model) and a seed.
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase8_validation as p8  # noqa: E402
import phase9_models as p9  # noqa: E402
import phase10_objectives as p10  # noqa: E402

OUT = Path("reports/phase11_tuning")
CACHE = Path(".cache/data/phase11")
H, F, THREADS, WORKERS = 700, 290, 4, 10
SEEDS6, SUITE = p9.SEEDS6, p9.SUITE
BASE = {"model": "gated", "width": 64, "hidden": 128, "layers": 2, "lr": 1e-3, "batch": 64,
        "aux": "l1", "lam": 0.0}  # fmt: skip
CANDIDATES = {
    "base": {},
    "lr3e-4": {"lr": 3e-4},
    "hidden256": {"hidden": 256},
    "hidden64": {"hidden": 64},
    "layers3": {"layers": 3},
    "width128": {"width": 128},
    "spectral0.1": {"aux": "spectral", "lam": 0.1},
    "spectral0.3": {"aux": "spectral", "lam": 0.3},
}
MIN_GAIN_MV = 0.0005  # stage A: holdout MAE must improve on base by >= 0.5 µV


def config(name: str) -> dict:
    return {**BASE, **CANDIDATES[name]}


def tag(c: dict) -> str:
    return "_".join(f"{k}{c[k]:g}" if isinstance(c[k], float) else f"{k}{c[k]}" for k in sorted(c))


# ------------------------------------------------------------------ shared trainer
def train_config(
    c: dict, k: int, s: int, path: Path, holdout_pred: bool = False, keep_model: bool = False
):
    """Train one run of configuration ``c``. ``c['model']``: 'gated' (sn4-gated CNN-LSTM),
    'additive' (sn4 + correction, no gate), 'input' (sn4 as extra input channels of a plain
    CNN-LSTM) or 'plain' (no sn4). ``c['filter']`` overrides the input filter."""
    import torch

    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.losses import aux_scale, make_objective
    from ptbxl_forecast.models import CNNLSTMForecaster, GatedBaselineCNNLSTM
    from ptbxl_forecast.preprocessing import BaselineConditioned, Pipeline, pack_baseline
    from ptbxl_forecast.training import predict, train_forecaster

    torch.set_num_threads(THREADS)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    rows, rows_ho, rows_ev, sub = p8.split_rows(meta, k, 1.0, 0)
    p = cfg.preprocessing
    pipe = Pipeline(c.get("filter", p.filter), p.scaler, False)
    size = {"width": c["width"], "hidden": c["hidden"], "layers": c["layers"]}
    data = sig
    if c["model"] == "plain":
        model = CNNLSTMForecaster(pipe.in_channels, 12, F, **size)
    else:
        data = pack_baseline(sig, np.load(p10.SN4), H)
        pipe = BaselineConditioned(pipe)
        if c["model"] == "input":
            model = CNNLSTMForecaster(pipe.in_channels + 12, 12, F, **size)
        else:
            model = GatedBaselineCNNLSTM(
                pipe.in_channels, 12, F, additive=c["model"] == "additive", **size
            )
    scale = 1.0
    if c["aux"] != "l1":
        r = rows[:: p10.SCALE_STRIDE]
        scale = aux_scale(
            c["aux"], sig[r, H : H + F].astype(np.float64), sig[r, :H].astype(np.float64)
        )
    pipe.fit(data[rows, :H].astype(np.float64), sub.index)
    info = train_forecaster(model, pipe, data[rows], data[rows_ho], H, F, seed=s, lr=c["lr"],
                            batch_size=c["batch"], epochs=p10.MAX_EPOCHS, threads=THREADS,
                            loss=make_objective(c["aux"], c["lam"], scale))  # fmt: skip
    info.update(config=c, k=k, s=s, aux_scale=scale, n_records=len(rows),
                params=sum(q.numel() for q in model.parameters()))  # fmt: skip
    out = {"info": np.array(info, dtype=object)}
    if len(rows_ev):
        out["pred"] = predict(model, pipe, data[rows_ev, :H].astype(np.float64)).astype(np.float32)
    if holdout_pred:
        out["pred_holdout"] = predict(model, pipe, data[rows_ho, :H].astype(np.float64)).astype(
            np.float32
        )
    if keep_model:
        torch.save(model.state_dict(), path.with_suffix(".pt"))
        pickle.dump(pipe, path.with_suffix(".pipe.pkl").open("wb"))
    np.savez(path, **out)
    return str(path)


def _job(sp: dict) -> str:
    path = Path(sp["path"])
    if path.exists():
        return str(path)
    return train_config(sp["config"], sp["k"], sp["s"], path, sp.get("holdout_pred", False),
                        sp.get("keep_model", False))  # fmt: skip


def run_many(specs: list[dict], log=print) -> float:
    todo = [s for s in specs if not Path(s["path"]).exists()]
    todo.sort(key=lambda s: -s["config"]["hidden"] * s["config"]["layers"])  # slow first
    t0 = time.perf_counter()
    if todo:
        log(f"training {len(todo)} runs on {WORKERS} workers")
        with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
            for p in pool.map(_job, todo):
                log(time.strftime("%H:%M") + " done " + Path(p).name)
    return (time.perf_counter() - t0) / 60


def run_path(c: dict, k: int, s: int, cache: Path = CACHE) -> Path:
    """Base configuration reuses Phase 10's sn4_gated runs (same code path, checked)."""
    if c == BASE and cache == CACHE:
        old = p10.run_path("sn4_gated", 0.0, k, s)
        if old.exists():
            return old
    return cache / f"{tag(c)}_k{k}_s{s}.npz"


def spec(c, k, s, cache: Path = CACHE, **kw) -> dict:
    return {"config": c, "k": k, "s": s, "path": str(run_path(c, k, s, cache)), **kw}


# ------------------------------------------------------------------ stages
def stage_a() -> None:
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import patient_macro

    s0 = SEEDS6[0]
    specs = [spec(config(n), 9, s0, holdout_pred=True) for n in CANDIDATES if n != "base"]
    specs.append(spec(BASE, 9, s0, CACHE / "verify", holdout_pred=True))  # bitwise reuse check
    (CACHE / "verify").mkdir(parents=True, exist_ok=True)
    minutes = run_many(specs)
    verify = np.array_equal(
        np.load(specs[-1]["path"])["pred"], np.load(run_path(BASE, 9, s0))["pred"]
    )
    meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
    _, rows_ho, _, _ = p8.split_rows(meta, 9, 1.0, 0)
    pid = meta.patient_id.to_numpy()[rows_ho]
    p10._HO.update(
        sig=sig,
        rows=rows_ho,
        peaks=pickle.load((p10.CACHE / "holdout_k9_true_peaks.pkl").open("rb")),
    )
    paths = {n: run_path(config(n), 9, s0) for n in CANDIDATES}
    with Pool(len(paths)) as pool:
        scores = pool.map(p10._ho_score, [np.load(paths[n])["pred_holdout"] for n in paths])
    rows = []
    for n, df in zip(paths, scores, strict=True):
        i = p9.info(paths[n])
        rows.append({"candidate": n, **{m: patient_macro(df[m], pid) for m in SUITE},
                     "best_epoch": i["best_epoch"], "epochs_run": i["epochs_run"],
                     "minutes": i["seconds"] / 60, "params": i.get("params")})  # fmt: skip
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_stage_a_grid.csv", index=False, float_format="%.6g")
    base = t.set_index("candidate").loc["base", "mae"]
    better = t[(t.candidate != "base") & (t.mae <= base - MIN_GAIN_MV)].sort_values("mae")
    S = {"shortlist": better.candidate.iloc[:2].tolist(), "base_holdout_mae": float(base),
         "base_rerun_bitwise_identical": bool(verify), "minutes": minutes}  # fmt: skip
    json.dump(S, (OUT / "stage_a.json").open("w"), indent=2)
    pd.set_option("display.width", 250)
    print(t.round(5).to_string(), "\n", json.dumps(S, indent=2))


def stage_b() -> None:
    from ptbxl_forecast.stats import compare_runs

    short = json.load((OUT / "stage_a.json").open())["shortlist"]
    names = ["base", *short]
    minutes = run_many([spec(config(n), 9, s) for n in short for s in SEEDS6])
    meta, sig, fold_rows, pids, base = p9.setup_scoring()
    pid = pids[9]
    paths = {n: [str(run_path(config(n), 9, s)) for s in SEEDS6] for n in names}
    jobs = {(n, s): (9, [p]) for n in names for s, p in zip(SEEDS6, paths[n], strict=True)}
    jobs.update({(n, "ens"): (9, paths[n]) for n in names})
    sc = p9.score_many(jobs, CACHE / "scores_stage_b.pkl")
    rows, cmps = [], []
    for n in names:
        runs = [sc[(n, s)] for s in SEEDS6]
        single = pd.DataFrame([p9.macro(r, pid) for r in runs])
        rows.append({"candidate": n, **{f"{m}_seed_mean": single[m].mean() for m in SUITE},
                     **{f"ens_{m}": v for m, v in p9.macro(sc[(n, "ens")], pid).items()}})  # fmt: skip
        if n != "base":
            c = compare_runs(runs, [sc[("base", s)] for s in SEEDS6], pid)
            cmps.append({"candidate": n, "vs": "base", **p9.cmp_row(c)})
    t, comps = pd.DataFrame(rows), pd.DataFrame(cmps)
    t.to_csv(OUT / "table_stage_b_confirm.csv", index=False, float_format="%.6g")
    comps.to_csv(OUT / "table_stage_b_comparisons.csv", index=False, float_format="%.5g")
    winners = comps[comps.beats].candidate.tolist() if len(comps) else []
    final = t.set_index("candidate").loc[winners].mae_seed_mean.idxmin() if winners else "base"
    S = {"winners": winners, "final": final, "final_config": config(final), "minutes": minutes}
    json.dump(S, (OUT / "stage_b.json").open("w"), indent=2)
    pd.set_option("display.width", 250)
    print(t.round(5).to_string(), "\n", comps.round(3).to_string(), "\n", json.dumps(S, indent=2))


def final_config() -> dict:
    return json.load((OUT / "stage_b.json").open())["final_config"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"A": stage_a, "B": stage_b}[args.stage]()


if __name__ == "__main__":
    main()
