import numpy as np
import pandas as pd
import pytest
from scipy import signal as sps

from ptbxl_forecast.leakage import (
    LeakageError,
    assert_disjoint,
    assert_fitted_on,
    causality_violation,
    check_input_columns,
)
from ptbxl_forecast.splits import group_kfold, patient_holdout


@pytest.fixture
def meta():
    rng = np.random.default_rng(0)
    patients = rng.integers(0, 300, 1000)  # many patients with repeat records
    return pd.DataFrame({"patient_id": patients}, index=pd.RangeIndex(1, 1001, name="ecg_id"))


def test_assert_disjoint_detects_shared_patient(meta):
    assert_disjoint({"train": meta.iloc[:500], "val": meta.iloc[500:]}, key="index")
    with pytest.raises(LeakageError):
        assert_disjoint({"train": meta.iloc[:500], "val": meta.iloc[500:]})


def test_patient_splits_are_disjoint_and_complete(meta):
    parts = patient_holdout(meta, 0.2, seed=1)
    assert len(parts["train"]) + len(parts["val"]) == len(meta)
    folds = group_kfold(meta, 5, seed=1)
    val_ids = np.concatenate([f["val"].index for f in folds])
    assert sorted(val_ids) == sorted(meta.index)  # every record validated exactly once
    assert not set(folds[0]["val"].patient_id) & set(folds[0]["train"].patient_id)


def test_causality_check_separates_causal_from_zero_phase_filters():
    rng = np.random.default_rng(0)
    record = rng.normal(0, 1, (990, 12)).cumsum(axis=0)
    sos = sps.butter(2, 0.67, "lowpass", fs=100, output="sos")
    assert causality_violation(lambda x: sps.sosfilt(sos, x, axis=0), record, 700) == 0
    assert causality_violation(lambda x: sps.sosfiltfilt(sos, x, axis=0), record, 700) > 1e-3
    # Whole-record z-scoring also leaks: the future changes the mean/SD applied to the history.
    assert causality_violation(lambda x: (x - x.mean(0)) / x.std(0), record, 700) > 1e-3


def test_fit_and_metadata_guards():
    assert_fitted_on([1, 2], [1, 2, 3])
    with pytest.raises(LeakageError):
        assert_fitted_on([1, 4], [1, 2, 3])
    check_input_columns(["age_years", "sex", "device_family"])
    for cols in (["age_years", "scp_codes"], ["sc_MI"], ["recording_date"]):
        with pytest.raises(LeakageError):
            check_input_columns(cols)
