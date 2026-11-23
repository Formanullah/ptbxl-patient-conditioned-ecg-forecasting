from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptbxl_forecast.config import load_config
from ptbxl_forecast.data import (
    LEADS,
    SealedFoldError,
    load_dataset,
    load_metadata,
    load_signals,
    split_window,
)
from ptbxl_forecast.quality import record_defects

wfdb = pytest.importorskip("wfdb")
BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


@pytest.fixture
def ptbxl(tmp_path):
    """Three-record PTB-XL-shaped dataset: folds 1, 9 and 10."""
    rows = [
        (1, 101, 45.0, 0, "{'NORM': 100.0, 'SR': 0.0}", 1, None),
        (2, 102, 300.0, 1, "{'IMI': 50.0, 'AFIB': 0.0}", 9, ", I-AVR,"),
        (3, 103, 60.0, 0, "{'LVH': 100.0}", 10, None),
    ]
    db = pd.DataFrame(
        rows,
        columns=["ecg_id", "patient_id", "age", "sex", "scp_codes", "strat_fold", "static_noise"],
    )
    db["filename_lr"] = [f"records100/00000/{i:05d}_lr" for i in db.ecg_id]
    db["recording_date"] = "1990-01-01 10:00:00"
    for col in ("baseline_drift", "burst_noise", "electrodes_problems"):
        db[col] = None
    db.to_csv(tmp_path / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, None, None],
            "form": [None, None, None, None, None],
            "rhythm": [None, None, None, 1, 1],
            "diagnostic_class": ["NORM", "MI", "HYP", None, None],
            "diagnostic_subclass": ["NORM", "IMI", "LVH", None, None],
        },
        index=["NORM", "IMI", "LVH", "SR", "AFIB"],
    ).to_csv(tmp_path / "scp_statements.csv")
    rec_dir = tmp_path / "records100" / "00000"
    rec_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in db.ecg_id:
        sig = physiological_leads(rng)
        if i == 2:
            sig[:, 6] = 0.1  # V1 constant for the whole record: a dropout defect
        wfdb.wrsamp(
            f"{i:05d}_lr", fs=100, units=["mV"] * 12, sig_name=list(LEADS), p_signal=sig,
            fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(rec_dir),
        )  # fmt: skip
    return tmp_path


def physiological_leads(rng: np.random.Generator) -> np.ndarray:
    """Random 12-lead signal whose limb leads obey the Einthoven/Goldberger relations."""
    sig = rng.normal(0, 0.2, (1000, 12))
    i, ii = sig[:, 0], sig[:, 1]
    sig[:, 2], sig[:, 3], sig[:, 4], sig[:, 5] = ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2
    return np.round(sig, 3)


def test_test_fold_is_sealed(ptbxl):
    with pytest.raises(SealedFoldError):
        load_metadata(ptbxl, [1, 10])
    meta = load_metadata(ptbxl, [1, 9, 10], allow_test=True)
    assert list(meta.index) == [1, 2, 3]
    with pytest.raises(SealedFoldError):
        load_signals(ptbxl, meta)


def test_metadata_derivations(ptbxl):
    meta = load_metadata(ptbxl, [1, 9])
    assert list(meta.index) == [1, 2]
    assert meta.loc[2, "age_over_89"] and np.isnan(meta.loc[2, "age_years"])
    assert meta.loc[1, "age_years"] == 45 and meta.loc[2, "sex_label"] == "female"
    assert meta.loc[1, "sc_NORM"] and meta.loc[2, "sc_MI"] and not meta.loc[2, "sc_NORM"]
    assert meta.loc[2, "rhythm_codes"] == {"AFIB"} and meta.loc[2, "diag_subclasses"] == {"IMI"}
    assert meta.loc[2, "q_static_noise"] and not meta.loc[1, "q_any"]


def test_signals_match_wfdb_and_are_cached(ptbxl, tmp_path):
    meta = load_metadata(ptbxl, [1, 9])
    cache = tmp_path / "cache"
    sig = load_signals(ptbxl, meta, cache_dir=cache, workers=1)
    assert sig.shape == (2, 1000, 12) and sig.dtype == np.float32
    ref = wfdb.rdsamp(str(ptbxl / meta.filename_lr.iloc[1]))[0]
    np.testing.assert_allclose(sig[1], ref, atol=1e-6)
    assert len(list(cache.glob("*.npy"))) == 1
    np.testing.assert_array_equal(load_signals(ptbxl, meta, cache_dir=cache), sig)


def test_load_dataset_excludes_defects_and_trims(ptbxl):
    overrides = [f"paths.data={ptbxl}", "data.train_folds=[1]", "data.val_folds=[9]",
                 "data.test_folds=[10]"]  # fmt: skip
    cfg = load_config([BASE], overrides)
    meta, sig, defects = load_dataset(cfg, "dev", workers=1)
    assert list(defects.index[defects["any"]]) == [2] and defects.loc[2, "dropout"]
    assert list(meta.index) == [1] and sig.shape == (1, 990, 12)
    x, y = split_window(sig, cfg.data.history, cfg.data.horizon)
    assert x.shape == (1, 700, 12) and y.shape == (1, 290, 12)
    np.testing.assert_array_equal(np.concatenate([x, y], axis=1), sig)
    meta, sig, _ = load_dataset(load_config([BASE], [*overrides, "data.exclude_defects=false",
                                                     "data.trim_end=0"]), "dev", workers=1)  # fmt: skip
    assert list(meta.index) == [1, 2] and sig.shape == (2, 1000, 12)
    with pytest.raises(SealedFoldError):
        load_dataset(cfg, "test", workers=1)


def test_defect_rules():
    rng = np.random.default_rng(1)
    x = physiological_leads(rng)
    assert not any(record_defects(x).values())
    clipped = x.copy()
    clipped[100:106, 7] = 3.0  # 60 ms flat at the lead maximum
    assert record_defects(clipped)["plateau"]
    swapped = x.copy()
    swapped[:, 2] = -swapped[:, 2]  # lead III inverted: breaks III = II - I
    assert record_defects(swapped)["lead_relation"]
