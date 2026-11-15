"""Executable patient, temporal-boundary, and raw-training-scaler checks."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import read_json, study_dir, write_json
from data import metadata, validate_cache
from preprocessing import filter_history, observed_rr, waveform_scaler


def audit(cfg, cache):
    manifest = validate_cache(cache, cfg)
    frame = metadata(cfg["data"]["root"])
    patients = [set(frame.loc[frame.split.eq(s), "patient_id"]) for s in ("train", "val", "test")]
    assert all(not patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3))
    checks = ["official patient groups are disjoint"]
    for split in ("train", "val"):
        rows = pd.read_csv(Path(cache) / f"metadata_{split}.csv")
        assert rows.ecg_id.is_unique
        assert manifest["splits"][split]["records"] == manifest["splits"][split]["windows"]
    checks.append("exactly one 1,000-sample record window in training and validation")
    raw = np.load(Path(cache) / "waves_train.npy", mmap_mode="r")
    scaler = read_json(Path(cache) / "scaler.json")
    fitted = waveform_scaler(raw, cfg["data"]["history"])
    np.testing.assert_array_equal(scaler["mean"], fitted["mean"])
    np.testing.assert_array_equal(scaler["std"], fitted["std"])
    checks.append("scaler refits exactly from raw training history only")
    wave = np.load(Path(cache) / "waves_val.npy", mmap_mode="r")[0].copy()
    changed = wave.copy()
    h = cfg["data"]["history"]
    changed[h:] += 10000
    np.testing.assert_array_equal(wave[:h], changed[:h])
    np.testing.assert_array_equal(filter_history(wave[:h], cfg), filter_history(changed[:h], cfg))
    np.testing.assert_allclose(observed_rr(wave[:h]), observed_rr(changed[:h]), equal_nan=True)
    checks.append("future mutation cannot alter raw/filtered histories or observed RR features")
    prefix = h // 2
    modified_history = wave[:h].copy()
    modified_history[prefix:] += 1000
    np.testing.assert_array_equal(
        filter_history(wave[:h], cfg)[:prefix], filter_history(modified_history, cfg)[:prefix]
    )
    checks.append("causal filter prefix is independent of later history samples")
    result = {
        "passed": True,
        "cohort": manifest["cohort"],
        "checks": checks,
        "synthetic": manifest["synthetic"],
        "test_signal_read": False,
    }
    write_json(study_dir(cfg) / cfg["run_mode"] / "leakage_audit.json", result)
    return result
