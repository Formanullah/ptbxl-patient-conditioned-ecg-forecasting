"""Phase 5, ridge tier: every preprocessing variant with the ridge reference learner.

Fits on folds 1-8 (λ on an internal patient holdout), scores fold 9 with the Phase 6
suite, compares each variant with the default pipeline using the D6.2 rule, and adds a
t-SNE view of history representations (analysis only).

    python research/phase5_ridge.py
"""

from __future__ import annotations

import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from ptbxl_forecast import features as FE  # noqa: E402
from ptbxl_forecast import metrics as M  # noqa: E402
from ptbxl_forecast.config import load_config  # noqa: E402
from ptbxl_forecast.data import load_dataset  # noqa: E402
from ptbxl_forecast.leakage import assert_fitted_on  # noqa: E402
from ptbxl_forecast.models.ridge import RidgeForecaster  # noqa: E402
from ptbxl_forecast.preprocessing import Pipeline  # noqa: E402
from ptbxl_forecast.stats import cluster_bootstrap, compare  # noqa: E402

OUT = Path("reports/phase5_preprocessing")
CACHE = Path(".cache/data/phase5")
H, F, SEED = 700, 290, 20261122
VARIANTS = (
    [("filter", f, {"filter_name": f}) for f in
     ["none", "causal_hp_0.5", "causal_bp_0.5_40", "zero_phase_bp_0.5_40_history", "causal_fir_lp_40",
      "wavelet_db4", "median_baseline", "decompose_0.67"]]
    + [("scaling", s, {"scaler_name": s}) for s in
       ["global_standard", "lead_standard", "lead_robust", "lead_minmax", "instance_history"]]
    + [("leads", "8_independent_leads", {"leads8": True})]
    + [("features", f, {"features": f}) for f in ["statistical", "morphological", "frequency", "hybrid"]]
    + [("dimensionality", f"pca_{v}", {"pca": v}) for v in (0.95, 0.99)]
)  # fmt: skip


def true_peaks(args):
    x, y = args
    return M.detect_beats(x, y)


def feature_block(name: str, x: np.ndarray, morph: np.ndarray) -> np.ndarray:
    parts = {"statistical": [FE.statistical(x)], "frequency": [FE.frequency(x)], "morphological": [morph],
             "hybrid": [FE.statistical(x), FE.frequency(x), morph]}[name]  # fmt: skip
    return np.concatenate(parts, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    tr_meta, tr_sig, _ = load_dataset(cfg, "train", cache_dir=".cache/data")
    va_meta, va_sig, _ = load_dataset(cfg, "val", cache_dir=".cache/data")
    xtr, ytr = tr_sig[:, :H].astype(np.float64), tr_sig[:, H : H + F].astype(np.float64)
    xva, yva = va_sig[:, :H].astype(np.float64), va_sig[:, H : H + F].astype(np.float64)
    groups, va_pid = tr_meta.patient_id.to_numpy(), va_meta.patient_id.to_numpy()

    pk = CACHE / "val_true_peaks.pkl"
    if not pk.exists():
        with Pool(38) as pool:
            pickle.dump(
                pool.map(true_peaks, list(zip(xva, yva, strict=True)), chunksize=32), pk.open("wb")
            )
    peaks = pickle.load(pk.open("rb"))
    mf = CACHE / "morph_features.npz"
    if not mf.exists():
        np.savez(mf, tr=FE.morphological(xtr, workers=38), va=FE.morphological(xva, workers=38))
    morph = np.load(mf)

    results, scores = [], {}
    for category, name, opts in VARIANTS:
        path = CACHE / f"ridge_{name}.pkl"
        if path.exists():
            rec = pickle.load(path.open("rb"))
        else:
            t0 = time.perf_counter()
            pipe = Pipeline(
                opts.get("filter_name", "none"),
                opts.get("scaler_name", "none"),
                opts.get("leads8", False),
            )
            pipe.fit(xtr, tr_meta.index)
            assert_fitted_on(pipe.fit_ids, tr_meta.index)
            a_tr, a_va = (
                pipe.inputs(xtr).reshape(len(xtr), -1),
                pipe.inputs(xva).reshape(len(xva), -1),
            )
            extra = {}
            if "features" in opts:
                ftr = feature_block(opts["features"], xtr, morph["tr"])
                fva = feature_block(opts["features"], xva, morph["va"])
                mu, sd = ftr.mean(0), ftr.std(0) + 1e-8  # standardized with training statistics
                a_tr, a_va = np.hstack([a_tr, (ftr - mu) / sd]), np.hstack([a_va, (fva - mu) / sd])
            if "pca" in opts:
                pca = PCA(n_components=opts["pca"], svd_solver="full", random_state=0).fit(a_tr)
                a_tr, a_va = pca.transform(a_tr), pca.transform(a_va)
                extra["pca_components"] = int(pca.n_components_)
            ttr = pipe.targets(ytr, xtr).reshape(len(xtr), -1)
            model = RidgeForecaster(seed=SEED).fit(a_tr, ttr, groups)
            pred = pipe.outputs(model.predict(a_va).reshape(len(xva), F, pipe.out_leads), xva)
            df = M.evaluate(xva, yva, pred, true_peaks=peaks)
            rec = {"category": category, "variant": name, "lambda": model.lam, "n_inputs": a_tr.shape[1],
                   "fit_seconds": time.perf_counter() - t0, "scores": df, **extra}  # fmt: skip
            pickle.dump(rec, path.open("wb"))
            np.save(CACHE / f"ridge_pred_{name}.npy", pred.astype(np.float32))
        scores[name] = rec["scores"]
        row = {k: v for k, v in rec.items() if k != "scores"}
        for m in (
            "mae",
            "dtw_mae",
            "template_mae",
            "beat_timing_ms",
            "rmse",
            "mae_8lead",
            "pearson",
        ):
            row[m] = cluster_bootstrap(rec["scores"][m], va_pid, n_boot=200, seed=SEED)[0]
        row["share_timing_le_50ms"] = float((rec["scores"].beat_timing_ms <= 50).mean())
        results.append(row)
        print(
            category,
            name,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
            flush=True,
        )

    base = scores["none"]
    for row in results:
        if row["variant"] == "none":
            continue
        c = compare(scores[row["variant"]], base, va_pid, n_boot=1000, seed=SEED)
        row.update(
            {f"d_{m}": c[m]["diff"] for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms")}
        )
        row.update(
            {f"d_{m}_lo": c[m]["lo"] for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms")}
        )
        row.update(
            {f"d_{m}_hi": c[m]["hi"] for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms")}
        )
        row["beats_default"] = c["beats_reference"]
    table = pd.DataFrame(results)
    table.to_csv(OUT / "table_ridge.csv", index=False, float_format="%.5f")
    (OUT / "ridge.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2, default=str) + "\n"
    )
    tsne(tr_meta, xtr)


def tsne(meta: pd.DataFrame, x: np.ndarray) -> None:
    """t-SNE of instance-normalized 8-lead histories (analysis only): do devices/rhythms cluster?"""
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(x), 4000, replace=False))
    z = x[idx][..., [0, 1, 6, 7, 8, 9, 10, 11]]
    z = ((z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-3))[:, ::2].reshape(
        len(idx), -1
    )
    emb = TSNE(2, init="pca", random_state=0, perplexity=40).fit_transform(
        PCA(50, random_state=0).fit_transform(z)
    )
    m = meta.iloc[idx]
    groups = {"device family": m.device.str.split().str[0].to_numpy(),
              "rhythm": m.rhythm_codes.apply(lambda r: "AFIB" if "AFIB" in r else "PACE" if "PACE" in r
                                             else "SR" if "SR" in r else "other").to_numpy(),
              "superclass": m.diag_superclasses.apply(lambda s: "NORM only" if s == {"NORM"} else "abnormal").to_numpy()}  # fmt: skip
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a, (title, lab) in zip(ax, groups.items(), strict=True):
        for v in sorted(set(lab)):
            k = lab == v
            a.scatter(emb[k, 0], emb[k, 1], s=3, alpha=0.5, label=f"{v} ({k.sum()})")
        a.set(title=f"t-SNE of histories, coloured by {title}", xticks=[], yticks=[])
        a.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_tsne.png", dpi=130)


if __name__ == "__main__":
    main()
