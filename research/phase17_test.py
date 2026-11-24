"""Phase 17: ONE-TIME evaluation on the sealed fold 10 (PREREGISTRATION.md).

Requires explicit user approval. Refuses to run twice (lock file). The frozen models of
Phase 16 are verified against MANIFEST.json before fold 10 is loaded.

    python research/phase17_test.py --dry-run                    # code check on fold 9 (in-sample; numbers meaningless)
    python research/phase17_test.py --unseal-fold-10-once        # the real, one-time test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase9_models as p9  # noqa: E402
import phase10_objectives as p10  # noqa: E402

OUT = Path("reports/phase17_test")
LOCK = OUT / "UNSEALED.lock"
MANIFEST = Path("reports/phase16_final/MANIFEST.json")
H, F = 700, 290
SUITE = p9.SUITE
_G: dict = {}


def _peaks(i):
    from ptbxl_forecast import metrics as M

    s = _G["sig"]
    return M.detect_beats(s[i, :H].astype(np.float64), s[i, H : H + F].astype(np.float64))


def _score(pred):
    from ptbxl_forecast import metrics as M

    s = _G["sig"].astype(np.float64)
    return M.evaluate(s[:, :H], s[:, H : H + F], pred.astype(np.float64), true_peaks=_G["peaks"])


def _predict(args):
    import torch

    from ptbxl_forecast.models import CNNLSTMForecaster, GatedBaselineCNNLSTM
    from ptbxl_forecast.preprocessing import pack_baseline
    from ptbxl_forecast.training import predict

    run, c = args
    torch.set_num_threads(4)
    pipe = pickle.load(Path(run["pipeline"]).open("rb"))
    size = (c["width"], c["hidden"], c["layers"])
    sig = _G["sig"].astype(np.float64)
    if c["model"] == "plain":
        model, x = CNNLSTMForecaster(pipe.in_channels, 12, F, *size), sig[:, :H]
    else:
        model = GatedBaselineCNNLSTM(
            pipe.in_channels, 12, F, *size, additive=c["model"] == "additive"
        )
        x = pack_baseline(sig, _G["sn4"].astype(np.float64), H)[:, :H]
    model.load_state_dict(torch.load(run["weights"]))
    return predict(model, pipe, x)


def verify_manifest(man: dict) -> None:
    for arm in man["runs"]:
        for r in man["runs"][arm]:
            for key in ("weights", "pipeline"):
                h = hashlib.sha256(Path(r[key]).read_bytes()).hexdigest()
                if h != r[f"sha256_{key}"]:
                    raise SystemExit(f"{r[key]} does not match the frozen manifest")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--unseal-fold-10-once", action="store_true")
    args = ap.parse_args()
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import compare_runs

    man = json.load(MANIFEST.open())
    verify_manifest(man)
    if args.dry_run:
        out = Path(".cache/data/phase17_dryrun")
        meta, sig, _ = load_dataset(load_config(), "dev", cache_dir=".cache/data")
        keep = np.flatnonzero(meta.strat_fold == 9)
        meta, sig = meta.iloc[keep], sig[keep]
    else:
        if LOCK.exists():
            raise SystemExit(
                f"fold 10 was already evaluated ({LOCK.read_text().strip()}); refusing to rerun"
            )
        out = OUT
        OUT.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(f"unsealed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        cfg = load_config(overrides=["data.allow_test=true"])
        meta, sig, _ = load_dataset(cfg, "test", cache_dir=".cache/data")
        assert set(meta.strat_fold) == {10}
    out.mkdir(parents=True, exist_ok=True)
    _G["sig"] = np.asarray(sig)
    with Pool(38) as pool:
        _G["peaks"] = pool.map(_peaks, range(len(sig)), chunksize=16)
        _G["sn4"] = np.stack(
            pool.map(p10._sn4, [sig[i, :H] for i in range(len(sig))], chunksize=32)
        ).astype(np.float32)
    flat = np.broadcast_to(np.median(sig[:, :H], axis=1, keepdims=True), (len(sig), F, 12)).astype(
        np.float64
    )
    preds = {}
    with Pool(10) as pool:
        for arm in ("final", "l1"):
            preds[arm] = pool.map(_predict, [(r, man["configs"][arm]) for r in man["runs"][arm]])
    jobs = {(a, i): p for a in preds for i, p in enumerate(preds[a])}
    jobs.update({("final", "ens"): np.mean(preds["final"], 0), ("l1", "ens"): np.mean(preds["l1"], 0),
                 ("sn4", 0): _G["sn4"], ("flat", 0): flat})  # fmt: skip
    keys = list(jobs)
    with Pool(min(30, len(keys))) as pool:
        sc = dict(zip(keys, pool.map(_score, [jobs[k] for k in keys]), strict=True))
    pid = meta.patient_id.to_numpy()
    fin, l1 = [sc[("final", i)] for i in range(6)], [sc[("l1", i)] for i in range(6)]
    rows = []
    for name, runs in (("final", fin), ("l1", l1)):
        single = pd.DataFrame([p9.macro(r, pid) for r in runs])
        rows.append({"arm": name, **{f"{m}_single_mean": single[m].mean() for m in SUITE},
                     **{f"ens_{m}": v for m, v in p9.macro(sc[(name, "ens")], pid).items()}})  # fmt: skip
    for name in ("flat", "sn4"):
        rows.append(
            {"arm": name, **{f"ens_{m}": v for m, v in p9.macro(sc[(name, 0)], pid).items()}}
        )
    cmps = [{"vs": "flat", **p9.cmp_row(compare_runs(fin, [sc[("flat", 0)]], pid))},
            {"vs": "sn4", **p9.cmp_row(compare_runs(fin, [sc[("sn4", 0)]], pid))},
            {"vs": "l1", **p9.cmp_row(compare_runs(fin, l1, pid))}]  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "table_test.csv", index=False, float_format="%.6g")
    pd.DataFrame(cmps).to_csv(out / "table_test_comparisons.csv", index=False, float_format="%.5g")
    per = pd.concat([sc[(a, k)].assign(arm=a, ecg_id=meta.index.to_numpy(), patient_id=pid)
                     for a, k in (("final", "ens"), ("l1", "ens"), ("sn4", 0), ("flat", 0))])  # fmt: skip
    per.to_parquet(out / "per_record_scores.parquet")
    S = {"mode": "dry-run (fold 9, in-sample)" if args.dry_run else "fold 10 one-time test",
         "records": len(meta), "patients": int(len(np.unique(pid))),
         "final_beats": {c["vs"]: bool(c["beats"]) for c in cmps}}  # fmt: skip
    json.dump(S, (out / "summary.json").open("w"), indent=2)
    pd.set_option("display.width", 250)
    print(
        pd.DataFrame(rows).round(5).to_string(),
        "\n",
        pd.DataFrame(cmps).round(2).to_string(),
        "\n",
        S,
    )


if __name__ == "__main__":
    main()
