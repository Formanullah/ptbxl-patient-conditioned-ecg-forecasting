<<<<<<< HEAD
"""Verify real-cache isolation and future-mutation invariance before optimization."""

import argparse
=======
"""Executable patient, temporal-boundary, and raw-training-scaler checks."""

>>>>>>> origin/work/manual-src-workflow
from pathlib import Path

import numpy as np
import pandas as pd
<<<<<<< HEAD
import torch

from config import load_config, write_json
from models import augment
from preprocessing import fit_scaling, input_filter, metadata, observed_features, read_wave, validate_cache
from evaluate import authorize_split


def audit(cfg):
    manifest = validate_cache(cfg)
    cache = Path(cfg["data"]["cache"])
    checks = []
    frames = {s: pd.read_csv(cache / f"metadata_{s}.csv") for s in ("train", "val", "test")}
    for column in ("patient_id", "ecg_id", "window_id"):
        values = [set(frame[column]) for frame in frames.values()]
        assert all(not values[i] & values[j] for i in range(3) for j in range(i + 1, 3)), column
        checks.append(f"PASS: {column} is isolated between splits")
    for split in ("val", "test"):
        assert frames[split].ecg_id.is_unique
    checks.append("PASS: one validation/test window per recording")
    raw = np.load(cache / "raw_X_train.npy", mmap_mode="r")
    temporary = cache / "_audit_filtered.npy"
    if len(raw) > cfg["data"].get("memmap_threshold", 5000):
        filtered = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=raw.shape
        )
        for i, x in enumerate(raw):
            filtered[i] = input_filter(x, cfg)
    else:
        filtered = np.stack([input_filter(x, cfg) for x in raw])
    center, scale = fit_scaling(
        filtered, cfg["preprocessing"]["scaling"], np.ones(len(raw), dtype=bool)
    )
    if isinstance(filtered, np.memmap):
        filtered._mmap.close()
        temporary.unlink()
    np.testing.assert_array_equal(center, manifest["mean_mV"])
    np.testing.assert_array_equal(scale, manifest["std_mV"])
    checks.append("PASS: normalization refits exactly from training observed windows")
    all_meta = metadata(cfg["data"]["root"])
    selected_ids = frames["train"].ecg_id.unique()
    age = pd.to_numeric(all_meta.loc[all_meta.ecg_id.isin(selected_ids), "age"], errors="coerce")
    age = age.where(age.between(0, 120)).dropna()
    expected = float(age.median()) if len(age) else 0.0
    assert expected == manifest["static_statistics"]["age_median"]
    checks.append("PASS: age imputation matches selected training records only")
    row = all_meta.set_index("ecg_id").loc[int(frames["val"].ecg_id.min())]
    wave = read_wave(row, cfg).T
    cut = cfg["data"]["start"] + cfg["data"]["history"]
    changed = wave.copy()
    changed[:, cut:] += 10000
    st = cfg["data"]["start"]
    for kind in ("none", "highpass", "bandpass"):
        settings = {**cfg, "preprocessing": {**cfg["preprocessing"], "filter": kind}}
        a, b = [input_filter(x[:, st:cut], settings) for x in (wave, changed)]
        np.testing.assert_array_equal(a, b)
        f1, _ = observed_features(wave[:, st:cut], cfg["data"]["sampling_rate"])
        f2, _ = observed_features(changed[:, st:cut], cfg["data"]["sampling_rate"])
        np.testing.assert_allclose(f1, f2, equal_nan=True)
        torch.manual_seed(42)
        view_a = augment(torch.tensor(a[None]), cfg["contrastive"])
        torch.manual_seed(42)
        view_b = augment(torch.tensor(b[None]), cfg["contrastive"])
        assert torch.equal(view_a, view_b)
        checks.append(
            f"PASS: future mutation cannot alter {kind} inputs, RR features or contrastive views"
        )
    try:
        authorize_split(cfg, "test")
    except ValueError:
        checks.append("PASS: fold-10 evaluation is blocked during development")
    else:
        checks.append(
            "FINAL MODE: test access explicitly enabled; check frozen selection provenance"
        )
    summary = {
        "fingerprint": manifest["fingerprint"],
        "checks": checks,
        "passed": True,
        "splits": manifest["splits"],
    }
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "leakage_audit.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    args = parser.parse_args()
    print(audit(load_config(args.config, args.experiment)))
=======

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
>>>>>>> origin/work/manual-src-workflow
