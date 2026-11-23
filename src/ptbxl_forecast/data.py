"""PTB-XL metadata and signal access with a sealed test fold.

Records are indexed via ``ptbxl_database.csv`` (never ``RECORDS``, which is malformed
upstream; decision D1.5.3). Requesting any test fold raises ``SealedFoldError``
unless ``allow_test`` is set.
"""

from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Iterable
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
QUALITY_FLAGS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems")
AGE_OVER_89_CODE = 300  # PTB-XL v1.0.2 changelog: ages above 89 are stored as 300
N_SAMPLES, N_LEADS = 1000, 12


class SealedFoldError(PermissionError):
    pass


def split_folds(cfg, split: str) -> list[int]:
    """Folds for 'train', 'val', 'dev' (train + val) or 'test', from ``cfg.data``."""
    d = cfg.data
    folds = {
        "train": list(d.train_folds),
        "val": list(d.val_folds),
        "dev": list(d.train_folds) + list(d.val_folds),
        "test": list(d.test_folds),
    }
    return folds[split]


def _guard(folds: Iterable[int], test_folds: Iterable[int], allow_test: bool) -> None:
    sealed = set(folds) & set(test_folds)
    if sealed and not allow_test:
        raise SealedFoldError(f"fold(s) {sorted(sealed)} are sealed; set data.allow_test=true")


def load_metadata(
    root: str | Path,
    folds: Iterable[int],
    test_folds: Iterable[int] = (10,),
    allow_test: bool = False,
) -> pd.DataFrame:
    """Metadata for the requested folds, with parsed labels and derived columns.

    Added columns: ``age_over_89``, ``age_years`` (NaN when missing or > 89),
    ``sex_label``, ``scp`` (dict), ``diag_superclasses``/``diag_subclasses`` (sets),
    ``rhythm_codes``/``form_codes`` (sets), one boolean ``sc_<CLASS>`` per superclass,
    one boolean ``q_<flag>`` per quality flag, ``q_any`` and ``recording_dt``.
    """
    folds = list(folds)
    _guard(folds, test_folds, allow_test)
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db = db[db.strat_fold.isin(folds)].copy()  # filter before deriving anything

    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]
    rhythm, form = set(scp.index[scp.rhythm == 1]), set(scp.index[scp.form == 1])
    db["scp"] = db.scp_codes.apply(ast.literal_eval)
    db["diag_superclasses"] = db.scp.apply(
        lambda c: {diag.diagnostic_class[k] for k in c if k in diag.index}
    )
    db["diag_subclasses"] = db.scp.apply(
        lambda c: {diag.diagnostic_subclass[k] for k in c if k in diag.index}
    )
    db["rhythm_codes"] = db.scp.apply(lambda c: set(c) & rhythm)
    db["form_codes"] = db.scp.apply(lambda c: set(c) & form)
    for cls in SUPERCLASSES:
        db[f"sc_{cls}"] = db.diag_superclasses.apply(lambda s, cls=cls: cls in s)

    db["age_over_89"] = db.age == AGE_OVER_89_CODE
    db["age_years"] = db.age.where(db.age != AGE_OVER_89_CODE)
    db["sex_label"] = db.sex.map({0: "male", 1: "female"})
    for flag in QUALITY_FLAGS:
        db[f"q_{flag}"] = db[flag].notna()
    db["q_any"] = db[[f"q_{f}" for f in QUALITY_FLAGS]].any(axis=1)
    db["recording_dt"] = pd.to_datetime(db.recording_date)
    return db


def _read(path: str) -> np.ndarray:
    import wfdb

    sig = wfdb.rdsamp(path)[0].astype(np.float32)
    if sig.shape != (N_SAMPLES, N_LEADS):
        raise ValueError(f"{path}: unexpected shape {sig.shape}")
    return sig


def load_signals(
    root: str | Path,
    meta: pd.DataFrame,
    cache_dir: str | Path | None = None,
    test_folds: Iterable[int] = (10,),
    allow_test: bool = False,
    workers: int | None = None,
) -> np.ndarray:
    """Raw 100 Hz signals in mV, float32, shape (n_records, 1000, 12), in ``meta`` order.

    With ``cache_dir`` the array is stored once per distinct record list and memory-mapped
    afterwards (read-only).
    """
    _guard(meta.strat_fold.unique(), test_folds, allow_test)
    root = Path(root)
    key = hashlib.sha256(",".join(map(str, meta.index)).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"signals100_{key}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")

    paths = [str(root / f) for f in meta.filename_lr]
    n_workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if n_workers == 1:
        signals = np.stack([_read(p) for p in paths])
    else:
        with Pool(n_workers) as pool:
            signals = np.stack(pool.map(_read, paths, chunksize=64))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npy")
        np.save(tmp, signals)
        os.replace(tmp, cache)
    return signals


def split_window(signals: np.ndarray, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` = observed history and forecast target. Models and every fitted or
    per-record preprocessing step may only see ``X`` (Phase 4 leakage rules)."""
    if signals.shape[1] < history + horizon:
        raise ValueError(f"record of {signals.shape[1]} samples < history + horizon")
    return signals[:, :history], signals[:, history : history + horizon]


def load_dataset(cfg, split: str, cache_dir: str | Path | None = None, workers: int | None = None):
    """Metadata and signals for a split, with the configured cleaning rules applied.

    Defect rules (``data.exclude_defects``, D3.1) are evaluated on the full record;
    then ``data.trim_end`` samples are removed from the end (D3.4). Returns
    ``(meta, signals, defects)``; ``signals`` has shape (n, 1000 - trim_end, 12) and
    ``defects`` lists every record of the split, including excluded ones.
    """
    from .quality import defect_table

    d = cfg.data
    meta = load_metadata(cfg.paths.data, split_folds(cfg, split), d.test_folds, d.allow_test)
    signals = load_signals(cfg.paths.data, meta, cache_dir, d.test_folds, d.allow_test, workers)
    defects = defect_table(signals, meta.index)
    keep = ~defects["any"].to_numpy() if d.exclude_defects else np.ones(len(meta), bool)
    meta, signals = meta[keep], signals[keep]
    end = N_SAMPLES - d.trim_end
    return meta, np.asarray(signals[:, :end]), defects
