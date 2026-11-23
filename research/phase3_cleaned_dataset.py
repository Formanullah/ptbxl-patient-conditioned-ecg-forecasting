"""Phase 3 completion: dataset statistics after the cleaning decisions D3.1 and D3.4.

Checks that the package's defect rules (``ptbxl_forecast.quality``) reproduce the
Phase 3 exclusion list, reports per-fold counts before/after exclusion, and measures
the effect of trimming the last 10 samples on the Phase 0 baselines (unmodified 300
vs trimmed 290 target samples).

    python research/phase3_cleaned_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_forecast.config import load_config
from ptbxl_forecast.data import load_dataset, load_metadata

sys.path.insert(0, str(Path(__file__).parent))
import phase0_feasibility as p0  # noqa: E402

OUT = Path("reports/phase3_quality")


def main() -> None:
    cfg = load_config()
    meta_all = load_metadata(cfg.paths.data, range(1, 10))
    meta, sig, defects = load_dataset(cfg, "dev", cache_dir=".cache/data")
    listed = set(pd.read_csv(OUT / "excluded_ecg_ids.csv").ecg_id)
    flagged = set(defects.index[defects["any"]])
    R: dict = {
        "package_rules_reproduce_phase3_list": flagged == listed,
        "excluded_records": len(flagged),
        "excluded_by_rule": {
            k: int(defects[k].sum()) for k in ("dropout", "plateau", "lead_relation")
        },
        "signals_shape_after_cleaning": list(sig.shape),
    }
    before = meta_all.groupby("strat_fold").agg(
        records=("patient_id", "size"), patients=("patient_id", "nunique")
    )
    after = meta.groupby("strat_fold").agg(
        records=("patient_id", "size"), patients=("patient_id", "nunique")
    )
    table = before.join(after, lsuffix="_before", rsuffix="_after")
    table.loc["total"] = [
        len(meta_all),
        meta_all.patient_id.nunique(),
        len(meta),
        meta.patient_id.nunique(),
    ]
    table.to_csv(OUT / "table_cleaned_counts.csv")
    R["patients_lost_entirely"] = int(meta_all.patient_id.nunique() - meta.patient_id.nunique())
    R["counts"] = table.astype(int).to_dict(orient="index")

    # Trimming impact on Phase 0 baselines (per-step MAE stored by Phase 0).
    arr = np.load("results/phase0/arrays.npz")
    abs_h, ids = arr["abs_h"], arr["ecg_id"]
    keep = np.isin(ids, meta.index)
    impact = {}
    for name in ("median", "mean", "sn4", "oracle"):
        k = p0.PREDICTORS.index(name)
        e = abs_h[keep, k]
        ok = ~np.isnan(e).any(axis=1)
        impact[name] = {
            "mae_300_unmodified": float(e[ok].mean()),
            "mae_290_trimmed": float(e[ok, :290].mean()),
            "mae_last_10_samples": float(e[ok, 290:].mean()),
            "last10_over_first290": float(e[ok, 290:].mean() / e[ok, :290].mean()),
        }
    R["trimming_impact_phase0_baselines_mV"] = impact
    (OUT / "cleaned_dataset.json").write_text(json.dumps(R, indent=2, default=str) + "\n")
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
