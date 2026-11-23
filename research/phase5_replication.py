"""Phase 5 addendum: replicate the two CNN variants whose verdict changed after the
initialization fix, with three new seeds (REPLICATION_PREREGISTRATION.md).

python research/phase5_replication.py
"""

from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from phase5_cnn import CACHE, OUT, job  # noqa: E402

H, F = 700, 290
OLD, NEW = (42, 123, 999), (7, 2024, 31337)
VARIANTS = {
    "default": {"pipeline": {}},
    "aug_gaussian_noise": {"pipeline": {}, "augment": ["gaussian_noise"]},
    "filter_decompose_0.67": {"pipeline": {"filter_name": "decompose_0.67"}},
}


def main() -> None:
    from ptbxl_forecast import metrics as M
    from ptbxl_forecast.config import load_config
    from ptbxl_forecast.data import load_dataset
    from ptbxl_forecast.stats import compare

    specs = [{"name": n, **v, "seed": s} for n, v in VARIANTS.items() for s in NEW]
    with ProcessPoolExecutor(len(specs), mp_context=get_context("spawn")) as pool:
        list(pool.map(job, specs))

    va_meta, va_sig, _ = load_dataset(load_config(), "val", cache_dir=".cache/data")
    x, y = va_sig[:, :H].astype(np.float64), va_sig[:, H : H + F].astype(np.float64)
    pid = va_meta.patient_id.to_numpy()
    peaks = pickle.load((CACHE / "val_true_peaks.pkl").open("rb"))

    def scores(name: str, seeds: tuple[int, ...]):
        pred = np.mean([np.load(CACHE / f"cnn_{name}_s{s}.npz")["pred"].astype(np.float64)
                        for s in seeds], axis=0)  # fmt: skip
        return M.evaluate(x, y, pred, true_peaks=peaks)

    out = {}
    for label, seeds in (("new_seeds", NEW), ("six_seeds", OLD + NEW)):
        ref = scores("default", seeds)
        for name in ("aug_gaussian_noise", "filter_decompose_0.67"):
            c = compare(scores(name, seeds), ref, pid, n_boot=1000, seed=0)
            out.setdefault(name, {})[label] = {
                **{
                    m: {k: round(float(c[m][k]), 6) for k in ("diff", "lo", "hi")}
                    for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms")
                },  # fmt: skip
                "beats_default": bool(c["beats_reference"]),
            }
    for v in out.values():
        v["adopt"] = v["new_seeds"]["beats_default"] and v["six_seeds"]["beats_default"]
    (OUT / "replication.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
