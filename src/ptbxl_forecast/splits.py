"""Patient-level splits (Phase 4). Record-level splits are deliberately not offered."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .leakage import assert_disjoint


def fold_assignments(root: str | Path) -> pd.DataFrame:
    """``patient_id`` and ``strat_fold`` for *all* records, including the sealed fold.

    Only identifiers are read: no signals, labels or demographics. This exists solely
    to verify patient disjointness across train/validation/test.
    """
    return pd.read_csv(
        Path(root) / "ptbxl_database.csv",
        usecols=["ecg_id", "patient_id", "strat_fold"],
        index_col="ecg_id",
    )


def official_split(meta: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    """Train/validation partitions from the configured official folds (test excluded)."""
    parts = {
        "train": meta[meta.strat_fold.isin(cfg.data.train_folds)],
        "val": meta[meta.strat_fold.isin(cfg.data.val_folds)],
    }
    assert_disjoint(parts)
    return parts


def patient_holdout(meta: pd.DataFrame, val_fraction: float, seed: int) -> dict[str, pd.DataFrame]:
    """Random patient-level holdout: each patient's records all go to one side."""
    patients = np.sort(meta.patient_id.unique())
    rng = np.random.default_rng(seed)
    val_patients = set(rng.choice(patients, round(val_fraction * len(patients)), replace=False))
    is_val = meta.patient_id.isin(val_patients)
    parts = {"train": meta[~is_val], "val": meta[is_val]}
    assert_disjoint(parts)
    return parts


def group_kfold(meta: pd.DataFrame, k: int, seed: int) -> list[dict[str, pd.DataFrame]]:
    """K patient-level folds; patients are shuffled, then assigned round-robin."""
    patients = np.sort(meta.patient_id.unique())
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(patients), np.arange(len(patients)) % k, strict=True))
    fold = meta.patient_id.map(fold_of).to_numpy()
    splits = []
    for f in range(k):
        parts = {"train": meta[fold != f], "val": meta[fold == f]}
        assert_disjoint(parts)
        splits.append(parts)
    return splits
