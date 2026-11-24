"""Phase 16: final models (PREREGISTRATION.md). Trains the final configuration and the
L1 CNN-LSTM reference on all cleaned records of folds 1-9 (minus the 20% internal patient
holdout used for early stopping), 6 seeds each, and freezes them in a manifest with
SHA-256 hashes. Fold 10 is not touched.

    python research/phase16_final.py   # after the training queue (run_12_18.py)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import phase9_models as p9  # noqa: E402
import phase11_tuning as p11  # noqa: E402

OUT = Path("reports/phase16_final")
CACHE = Path(".cache/data/phase16")
SEEDS6 = p9.SEEDS6
L1_CONFIG = {**p11.BASE, "model": "plain"}  # Phase 9 CNN-LSTM
# D12.1 (user, 2026-11-24, before fold 10 was opened): the 9-fold CV of Phase 12 showed
# hidden 256 (Phase 11 fold-9 choice) worse than hidden 128 on template MAE and timing,
# so the final model reverts to the Phase 10 configuration (hidden 128). The hidden-256
# models are kept as superseded (MANIFEST_superseded_hidden256.json).
FINAL_CONFIG = dict(p11.BASE)


def final_config() -> dict:
    return FINAL_CONFIG


def path(arm: str, s: int) -> Path:
    tag = "final128" if arm == "final" else arm
    return CACHE / f"{tag}_k10_s{s}.npz"


def specs() -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = {"final": final_config(), "l1": L1_CONFIG}
    return [{"config": cfg[a], "k": 10, "s": s, "path": str(path(a, s)), "keep_model": True}
            for a in cfg for s in SEEDS6]  # fmt: skip


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {"final": final_config(), "l1": L1_CONFIG}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    runs = {}
    for a in cfg:
        runs[a] = []
        for s in SEEDS6:
            p = path(a, s)
            i = p9.info(p)
            runs[a].append({"seed": s, "weights": str(p.with_suffix(".pt")), "sha256_weights": sha(p.with_suffix(".pt")),
                            "pipeline": str(p.with_suffix(".pipe.pkl")), "sha256_pipeline": sha(p.with_suffix(".pipe.pkl")),
                            "best_epoch": i["best_epoch"], "epochs_run": i["epochs_run"],
                            "best_holdout_l1": i["best_holdout_l1"], "minutes": i["seconds"] / 60,
                            "n_train_records": i["n_records"], "params": i["params"]})  # fmt: skip
    sn4_code = sha(Path("research/phase0_feasibility.py"))
    manifest = {"frozen_at_commit": commit, "configs": cfg, "seeds": list(SEEDS6), "runs": runs,
                "training_data": "cleaned folds 1-9 minus 20% internal patient holdout (seed 0)",
                "prediction": "mean of the 6 seeds' forecasts (raw mV, 12 leads, 290 samples after 700)",
                "sn4_source_sha256": sn4_code, "fold10_touched": False,
                "decision": "D12.1: hidden 128 (Phase 10 configuration) replaces hidden 256"}  # fmt: skip
    json.dump(manifest, (OUT / "MANIFEST.json").open("w"), indent=2, default=float)
    caps = sum(r["epochs_run"] == 100 for a in runs for r in runs[a])
    print(
        json.dumps({a: [round(r["best_holdout_l1"], 5) for r in runs[a]] for a in runs}, indent=2),
        "capped:",
        caps,
    )
    assert np.isfinite([r["best_holdout_l1"] for a in runs for r in runs[a]]).all()


if __name__ == "__main__":
    main()
