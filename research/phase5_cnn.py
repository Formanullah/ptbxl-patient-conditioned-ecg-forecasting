"""Phase 5, CNN tier: shortlisted preprocessing, augmentation, and the combined pipeline.

Each job trains the small CNN reference (3 seeds) on folds 1-8 minus an internal patient
holdout (early stopping), predicts fold 9, and is scored with the Phase 6 suite.
Variants are judged on their 3-seed ensemble against the default with the D6.2 rule.

    python research/phase5_cnn.py --stage shortlist|augment|combined [--combined-spec JSON]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/phase5_preprocessing")
CACHE = Path(".cache/data/phase5")
H, F, SEEDS, WORKERS, THREADS = 700, 290, (42, 123, 999), 9, 4


def job(spec: dict) -> dict:
    """Train one (variant, seed) and save fold-9 predictions. Runs in a spawned process."""
    import torch

    from ptbxl_forecast.augment import AUGMENTATIONS
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.models.cnn import CNNForecaster
    from ptbxl_forecast.preprocessing import Pipeline
    from ptbxl_forecast.splits import patient_holdout
    from ptbxl_forecast.training import predict, train_forecaster

    path = CACHE / f"cnn_{spec['name']}_s{spec['seed']}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {"name": spec["name"], "seed": spec["seed"], "info": d["info"].item()}
    torch.set_num_threads(THREADS)
    cfg = load_config()
    meta, sig, _ = load_dataset(cfg, "train", cache_dir=".cache/data")
    va_meta, va_sig, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    parts = patient_holdout(meta, 0.2, seed=0)  # the same internal holdout for every run
    rows_tr, rows_ho = (
        meta.index.get_indexer(parts["train"].index),
        meta.index.get_indexer(parts["val"].index),
    )
    p = spec["pipeline"]
    pipe = Pipeline(
        p.get("filter_name", "none"), p.get("scaler_name", "none"), p.get("leads8", False)
    )
    pipe.fit(sig[rows_tr, :H].astype(np.float64), parts["train"].index)
    augs = [AUGMENTATIONS[a] for a in spec.get("augment", [])]

    def augment(rec, h, g):
        for a in augs:
            rec = a(rec, h, g)
        return rec

    model = CNNForecaster(in_channels=pipe.in_channels, out_leads=pipe.out_leads, horizon=F)
    info = train_forecaster(model, pipe, sig[rows_tr], sig[rows_ho], H, F, seed=spec["seed"],
                            augment=augment if augs else None, threads=THREADS)  # fmt: skip
    pred = predict(model, pipe, va_sig[:, :H].astype(np.float64))
    info.update(params=sum(q.numel() for q in model.parameters()))
    np.savez(path, pred=pred.astype(np.float32), info=np.array(info, dtype=object))
    return {"name": spec["name"], "seed": spec["seed"], "info": info}


def specs_for(stage: str, combined: dict | None) -> list[dict]:
    default = [
        {"name": "default", "pipeline": {}, "seed": s} for s in SEEDS
    ]  # cached after the first run
    if stage == "augment":
        from ptbxl_forecast.augment import AUGMENTATIONS

        return default + [{"name": f"aug_{a}", "pipeline": {}, "augment": [a], "seed": s}
                          for a in AUGMENTATIONS for s in SEEDS]  # fmt: skip
    if stage == "combined":
        return default + [{"name": "combined", "pipeline": combined["pipeline"],
                           "augment": combined.get("augment", []), "seed": s} for s in SEEDS]  # fmt: skip
    ridge = pd.read_csv(OUT / "table_ridge.csv")
    sig = pd.read_csv(OUT / "table_signal_level.csv", index_col=0)
    eligible = [v for v in ridge[ridge.category == "filter"].sort_values("mae").variant
                if v != "none" and (v == "decompose_0.67" or (sig.loc[v, "pass_S1"] and sig.loc[v, "pass_S2"]))]  # fmt: skip
    best_scaler = ridge[ridge.category == "scaling"].sort_values("mae").variant.iloc[0]
    variants = {"default": {}}
    variants.update({f"filter_{f}": {"filter_name": f} for f in eligible[:2]})
    variants[f"scaling_{best_scaler}"] = {"scaler_name": best_scaler}
    variants["leads_8_independent"] = {"leads8": True}
    return [{"name": n, "pipeline": p, "seed": s} for n, p in variants.items() for s in SEEDS]


def evaluate(names: list[str]) -> pd.DataFrame:
    from ptbxl_forecast import metrics as M
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import cluster_bootstrap, compare

    cfg = load_config()
    va_meta, va_sig, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    x, y = va_sig[:, :H].astype(np.float64), va_sig[:, H : H + F].astype(np.float64)
    pid = va_meta.patient_id.to_numpy()
    peaks = pickle.load((CACHE / "val_true_peaks.pkl").open("rb"))
    scores, rows = {}, []
    for name in ["default", *[n for n in names if n != "default"]]:
        preds, infos, seed_mae = [], [], []
        for s in SEEDS:
            d = np.load(CACHE / f"cnn_{name}_s{s}.npz", allow_pickle=True)
            preds.append(d["pred"].astype(np.float64))
            infos.append(d["info"].item())
            seed_mae.append(
                float(pd.Series(np.abs(preds[-1] - y).mean(axis=(1, 2))).groupby(pid).mean().mean())
            )
        path = CACHE / f"cnn_scores_{name}.pkl"
        if path.exists():
            df = pickle.load(path.open("rb"))
        else:
            df = M.evaluate(x, y, np.mean(preds, axis=0), true_peaks=peaks)
            pickle.dump(df, path.open("wb"))
        scores[name] = df
        row = {"variant": name, "mae_seed_mean": float(np.mean(seed_mae)), "mae_seed_sd": float(np.std(seed_mae, ddof=1)),
               "best_epoch_mean": float(np.mean([i["best_epoch"] for i in infos])),
               "train_minutes_mean": float(np.mean([i["seconds"] for i in infos]) / 60), "params": infos[0]["params"]}  # fmt: skip
        for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms", "rmse", "pearson"):
            row[f"ens_{m}"] = cluster_bootstrap(df[m], pid, n_boot=200, seed=0)[0]
        row["ens_share_timing_le_50ms"] = float((df.beat_timing_ms <= 50).mean())
        if name != "default":
            c = compare(df, scores["default"], pid, n_boot=1000, seed=0)
            for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms"):
                row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = (
                    c[m]["diff"],
                    c[m]["lo"],
                    c[m]["hi"],
                )
            row["beats_default"] = c["beats_reference"]
        rows.append(row)
    return pd.DataFrame(rows)


def limb_residual(rec: np.ndarray) -> np.ndarray:
    """Per-record worst RMS violation (mV) of III=II-I, aVR=-(I+II)/2, aVL=I-II/2, aVF=II-I/2."""
    i, ii, iii, avr, avl, avf = (rec[..., k] for k in range(6))
    res = [iii - (ii - i), avr + (i + ii) / 2, avl - (i - ii / 2), avf - (ii - i / 2)]
    return np.max([np.sqrt((r**2).mean(axis=1)) for r in res], axis=0)


def vs_baselines() -> None:
    """Default-CNN ensemble vs the Phase 6 flat and sn4 baselines (D6.2 rule) and its
    limb-lead consistency; writes cnn_vs_baselines.json."""
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import compare

    cfg = load_config()
    va_meta, va_sig, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    dev, _, _ = load_dataset(cfg, "dev", cache_dir=".cache/data")
    pid = va_meta.patient_id.to_numpy()
    p6 = pd.read_parquet(".cache/data/phase6_scores.parquet")
    p6["ecg_id"] = dev.index.to_numpy()[p6.row]
    cnn = pickle.load((CACHE / "cnn_scores_default.pkl").open("rb"))
    out = {}
    for probe in ("median (flat)", "sn4 (repeat cycles)"):
        ref = p6[p6.probe == probe].set_index("ecg_id").loc[va_meta.index].reset_index(drop=True)
        c = compare(cnn, ref, pid, n_boot=1000, seed=0)
        out[probe] = {m: {k: round(float(c[m][k]), 5) for k in ("diff", "lo", "hi")}
                      for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms")}  # fmt: skip
        out[probe]["beats_baseline"] = bool(c["beats_reference"])
    pred = np.mean([np.load(CACHE / f"cnn_default_s{s}.npz")["pred"] for s in SEEDS], axis=0)
    r, t = (
        limb_residual(pred.astype(np.float64)),
        limb_residual(va_sig[:, H : H + F].astype(np.float64)),
    )
    out["cnn_limb_lead_residual_mV"] = {"median": float(np.median(r)), "p95": float(np.quantile(r, 0.95)),
                                        "share_gt_0.01": float((r > 0.01).mean()),
                                        "truth_median": float(np.median(t))}  # fmt: skip
    (OUT / "cnn_vs_baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["shortlist", "augment", "combined"], required=True)
    ap.add_argument("--combined-spec", type=str, default=None)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    combined = json.loads(args.combined_spec) if args.combined_spec else None
    specs = specs_for(args.stage, combined)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
        for r in pool.map(job, specs):
            print(r["name"], r["seed"], {k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in r["info"].items() if k != "log"}, flush=True)  # fmt: skip
    print(f"training done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    names = sorted({s["name"] for s in specs} | {"default"})
    table = evaluate(names)
    table.to_csv(OUT / f"table_cnn_{args.stage}.csv", index=False, float_format="%.5f")
    print(table.to_string())
    vs_baselines()


if __name__ == "__main__":
    main()
