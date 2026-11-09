"""Version-two preparation: patient sampling, causal input transforms and raw targets."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from config import FEATURES, digest, file_hash, protocol, read_json, write_json

EXPANDED = FEATURES + [
    name
    for variable in ("height", "weight", "bmi", "heart_rate", "rr_mean", "rr_std")
    for name in (variable, variable + "_missing")
]


def feature_indices(cfg, variant, names, manifest):
    if variant == "ecg_only":
        return []
    if variant in ("mrmr", "permutation"):
        artifact = read_json(cfg["features"][variant + "_path"])
        if artifact["fingerprint"] != manifest["fingerprint"] or artifact["method"] != variant:
            raise ValueError("Feature-selection provenance mismatch")
        selected = artifact["selected_features"]
    else:
        selected = (
            names[:4]
            if variant == "demographics"
            else (names if variant == "expanded" else names[:10])
        )
    return [names.index(name) for name in selected]


def select_patients(df, subset):
    mode = subset.get("mode", "custom")
    if mode not in ("pilot", "medium", "full", "custom"):
        raise ValueError("subset.mode must be pilot, medium, full or custom")
    defaults = {"pilot": (450, 100, 100), "medium": (1500, 300, 100)}
    selected = {}
    for j, split in enumerate(("train", "val", "test")):
        patients = np.sort(df.loc[df.split.eq(split), "patient_id"].unique())
        # Independent streams keep val/test membership stable as training size changes.
        rng = np.random.default_rng(np.random.SeedSequence([subset.get("seed", 42), j]))
        if mode == "full":
            count = len(patients)
        elif "fractions" in subset:
            fraction = subset["fractions"][split]
            if not 0 < fraction <= 1:
                raise ValueError("Patient fractions must be in (0,1]")
            count = max(1, int(np.floor(fraction * len(patients))))
        else:
            count = subset.get("patients", {}).get(split, defaults.get(mode, (0, 0, 0))[j])
        if not isinstance(count, int) or not 1 <= count <= len(patients):
            raise ValueError(f"Invalid patient count for {split}: {count}")
        selected[split] = np.sort(rng.permutation(patients)[:count]).tolist()
    keep = set(sum(selected.values(), []))
    return df.loc[df.patient_id.isin(keep)].reset_index(drop=True), selected


def input_filter(x, cfg):
    """Filter only observed samples, with fixed causal initial conditions per window."""
    setting = cfg["preprocessing"]
    kind = setting.get("filter", "none")
    if kind == "none":
        return np.asarray(x, dtype=np.float32).copy()
    if kind not in ("highpass", "bandpass"):
        raise ValueError("Filter must be none, highpass or bandpass")
    fs = cfg["data"]["sampling_rate"]
    cutoff = setting.get("low_hz", 0.5)
    if kind == "bandpass":
        cutoff = [cutoff, setting.get("high_hz", 40.0)]
    sos = signal.butter(setting.get("order", 4), cutoff, btype=kind, fs=fs, output="sos")
    # x is [leads,time]. Initial state depends only on each lead's first input sample.
    zi = signal.sosfilt_zi(sos)[:, None, :] * np.asarray(x)[:, 0][None, :, None]
    filtered, _ = signal.sosfilt(sos, x, axis=-1, zi=zi)
    return filtered.astype(np.float32)


def fit_scaling(x, method, train_mask=None):
    if train_mask is not None:
        centers, scales = [], []
        for j in range(12):
            values = x[train_mask, j, :]
            if method == "zscore":
                centers.append(values.mean(dtype=np.float64))
                scales.append(values.std(dtype=np.float64))
            elif method == "robust":
                q = np.quantile(values, [0.25, 0.5, 0.75])
                centers.append(q[1])
                scales.append(q[2] - q[0])
            else:
                raise ValueError("Scaling must be zscore or robust")
        scale = np.asarray(scales, dtype=np.float32)
        return np.asarray(centers, dtype=np.float32), np.where(scale > 1e-8, scale, 1.0)
    if method == "zscore":
        center = x.mean(axis=(0, 2), dtype=np.float64)
        scale = x.std(axis=(0, 2), dtype=np.float64)
    elif method == "robust":
        quantiles = np.array([np.quantile(x[:, j, :], [0.25, 0.5, 0.75]) for j in range(12)])
        center, scale = quantiles[:, 1], quantiles[:, 2] - quantiles[:, 0]
    else:
        raise ValueError("Scaling must be zscore or robust")
    return center.astype(np.float32), np.where(scale > 1e-8, scale, 1).astype(np.float32)


def window_starts(cfg, split):
    d = cfg["data"]
    if split != "train" or d.get("training_windows") == "single":
        return [d["start"]]
    return list(
        range(d["start"], 10 * d["sampling_rate"] - d["history"] - d["horizon"] + 1, d["stride"])
    )


def observed_features(x, fs):
    from morphology import detect_peaks

    peaks, failure = detect_peaks(x[1], fs)
    rr = np.diff(peaks) / fs
    return [
        60 / rr.mean() if len(rr) else np.nan,
        rr.mean() if len(rr) else np.nan,
        rr.std() if len(rr) >= 2 else np.nan,
    ], failure


def fit_extra(raw, train):
    """Impute/scale continuous features on training windows; flags remain unscaled."""
    result, stats = [], []
    for j in range(raw.shape[1]):
        a = raw[:, j]
        observed = a[train & np.isfinite(a)]
        median = float(np.median(observed)) if len(observed) else 0.0
        std = float(np.std(observed)) if len(observed) else 1.0
        std = std if std > 1e-8 else 1.0
        missing = ~np.isfinite(a)
        result.extend([(np.where(missing, median, a) - median) / std, missing])
        stats.append({"median": median, "std": std, "observed_training_windows": len(observed)})
    return np.column_stack(result).astype(np.float32), stats


def prepare(cfg):
    from preprocessing import metadata, read_wave, static_features

    started = time.perf_counter()
    d = cfg["data"]
    root, cache = Path(d["root"]), Path(d["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    requested = protocol(cfg)
    if (cache / "manifest.json").exists():
        previous = read_json(cache / "manifest.json")
        if previous["protocol"] != requested:
            raise ValueError("Cache exists for a different protocol; use a new cache directory")
        from preprocessing import validate_cache

        return validate_cache(cfg)
    df, selected = select_patients(metadata(root), cfg.get("subset", {"mode": "full"}))
    static, groups, age_stats = static_features(df, root)
    frames, statics, extras = [], [], []
    size = sum(len(window_starts(cfg, split)) for split in df.split)
    working = []

    def allocate(name, length):
        shape = (size, 12, length)
        if size > d.get("memmap_threshold", 5000):
            path = cache / f"_work_{name}.npy"
            working.append(path)
            return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        return np.empty(shape, dtype=np.float32)

    inputs = allocate("X", d["history"])
    targets = allocate("y", d["horizon"])
    raw_inputs = allocate("raw_X", d["history"])
    position = 0
    failures, detector_failures = [], 0
    for i, row in df.iterrows():
        try:
            wave = read_wave(row, cfg)
            for start in window_starts(cfg, row.split):
                cut = start + d["history"]
                observed = wave[start:cut].T.copy()
                target = wave[cut : cut + d["horizon"]].T.copy()
                inputs[position] = input_filter(observed, cfg)
                raw_inputs[position] = observed
                targets[position] = target
                position += 1
                statics.append(static[i])
                features, failed = observed_features(observed, d["sampling_rate"])
                detector_failures += int(failed)
                height = pd.to_numeric(row.get("height"), errors="coerce")
                weight = pd.to_numeric(row.get("weight"), errors="coerce")
                height = height if pd.notna(height) and 50 <= height <= 250 else np.nan
                weight = weight if pd.notna(weight) and 10 <= weight <= 400 else np.nan
                bmi = weight / (height / 100) ** 2 if np.isfinite(height * weight) else np.nan
                extras.append([height, weight, bmi, *features])
                info = groups.iloc[i].to_dict()
                info.update(window_start=int(start), window_id=f"{int(row.ecg_id)}:{start}")
                frames.append(info)
        except Exception as exc:
            failures.append({"ecg_id": int(row.ecg_id), "error": str(exc)})
    if failures:
        write_json(cache / "invalid_records.json", failures)
        raise ValueError("Invalid records; see invalid_records.json")
    meta = pd.DataFrame(frames)
    x, y, raw_x = inputs, targets, raw_inputs
    train = meta.split.eq("train").to_numpy()
    center, scale = fit_scaling(x, cfg["preprocessing"]["scaling"], train)
    for j in range(12):
        x[:, j, :] -= center[j]
        x[:, j, :] /= scale[j]
        y[:, j, :] -= center[j]
        y[:, j, :] /= scale[j]
    extra, extra_stats = fit_extra(np.asarray(extras, dtype=float), train)
    static = np.column_stack([np.asarray(statics), extra]).astype(np.float32)
    age_stats.update(features=EXPANDED, expanded_statistics=extra_stats)
    manifest = {
        "protocol": requested,
        "static_statistics": age_stats,
        "metadata_sha256": file_hash(root / "ptbxl_database.csv"),
        "scp_sha256": file_hash(root / "scp_statements.csv"),
        "mean_mV": center.tolist(),
        "std_mV": scale.tolist(),
        "scaling": cfg["preprocessing"]["scaling"],
        "selected_patients": selected,
        "targets": "raw unfiltered future mV, then affine scaling",
        "detector_failures_in_observed_features": detector_failures,
        "preprocessing_seconds": time.perf_counter() - started,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        mask = meta.split.eq(split).to_numpy()
        frame = meta.loc[mask].reset_index(drop=True)
        for name, array in (("X", x), ("y", y), ("static", static), ("raw_X", raw_x)):
            positions = np.flatnonzero(mask)
            destination = np.lib.format.open_memmap(
                cache / f"{name}_{split}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(positions), *array.shape[1:]),
            )
            for offset in range(0, len(positions), 128):
                destination[offset : offset + 128] = array[positions[offset : offset + 128]]
            destination.flush()
            del destination
        frame.to_csv(cache / f"metadata_{split}.csv", index=False)
        manifest["splits"][split] = {
            "patients": int(frame.patient_id.nunique()),
            "records": int(frame.ecg_id.nunique()),
            "windows": len(frame),
            "ids_sha256": digest(frame.ecg_id.tolist()),
            "window_ids_sha256": digest(frame.window_id.tolist()),
        }
    manifest["files"] = {
        p.name: file_hash(p)
        for pattern in ("*.npy", "metadata_*.csv")
        for p in cache.glob(pattern)
        if not p.name.startswith("_work_")
    }
    manifest["preprocessing_seconds"] = time.perf_counter() - started
    manifest["preprocessing_timing_scope"] = "read, features, scaling, array writing and hashing"
    manifest["fingerprint"] = digest(manifest)
    write_json(cache / "manifest.json", manifest)
    for array in (x, y, raw_x):
        if isinstance(array, np.memmap):
            array._mmap.close()
    for path in working:
        path.unlink()
    print(f"Prepared {cache}: {manifest['splits']}", flush=True)
    return manifest
