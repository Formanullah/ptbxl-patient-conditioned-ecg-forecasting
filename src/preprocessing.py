"""Audit ECGs, build train-fitted forecast windows, and construct DataLoaders."""

import argparse
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from filelock import FileLock
from scipy import signal
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASSES,
    FEATURES,
    LEADS,
    digest,
    file_hash,
    load_config,
    protocol,
    read_json,
    write_json,
)

EXPANDED = FEATURES + [
    name
    for variable in ("height", "weight", "bmi", "heart_rate", "rr_mean", "rr_std")
    for name in (variable, variable + "_missing")
]



def metadata(root):
    root = Path(root)
    df = pd.read_csv(root / "ptbxl_database.csv").sort_values("ecg_id").reset_index(drop=True)
    needed = {"ecg_id", "patient_id", "strat_fold", "age", "sex", "recording_date", "scp_codes"}
    if needed - set(df):
        raise ValueError(f"Missing metadata: {sorted(needed - set(df))}")
    if df.ecg_id.duplicated().any() or df[["patient_id", "strat_fold"]].isna().any().any():
        raise ValueError("Duplicate ECG IDs or missing patient/fold identifiers")
    if not df.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Invalid stratified fold")
    if df.groupby("patient_id").strat_fold.nunique().max() > 1:
        raise ValueError("Patient leakage across folds")
    df["split"] = np.where(df.strat_fold <= 8, "train", np.where(df.strat_fold == 9, "val", "test"))
    if set(df.split) != {"train", "val", "test"}:
        raise ValueError("Every split must be nonempty")
    return df


def static_features(df, root):
    """Current labels returned separately; never present in the static matrix."""
    df = df.copy()
    train = df.split.eq("train")
    age = pd.to_numeric(df.age, errors="coerce").where(lambda a: a.between(0, 120))
    observed = age[train].dropna()
    median = float(observed.median()) if len(observed) else 0.0
    std = float(observed.std(ddof=0)) if len(observed) else 1.0
    std = std if np.isfinite(std) and std > 1e-8 else 1.0
    sex = pd.to_numeric(df.sex, errors="coerce")
    valid_sex = sex.isin([0, 1])
    mapping = pd.read_csv(Path(root) / "scp_statements.csv", index_col=0)
    mapping = mapping.loc[mapping.diagnostic.eq(1), "diagnostic_class"].dropna().to_dict()
    labels = np.zeros((len(df), 5), dtype=np.float32)
    for i, codes in enumerate(df.scp_codes):
        parsed = ast.literal_eval(codes)
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid SCP dictionary for ECG {df.iloc[i].ecg_id}")
        # Likelihood 0 means unknown, not an absent statement (PhysioNet metadata).
        known = {mapping.get(code) for code in parsed}
        labels[i] = [label in known for label in CLASSES]
    prior = np.zeros_like(labels)
    missing = np.ones(len(df), dtype=np.float32)
    work = pd.DataFrame(
        {
            "position": np.arange(len(df)),
            "patient": df.patient_id,
            "date": pd.to_datetime(df.recording_date, errors="coerce"),
        }
    )
    for _, group in work.groupby("patient", sort=False):
        known = np.zeros(5, dtype=np.float32)
        seen = False
        for _, same_time in group.dropna(subset=["date"]).groupby("date", sort=True):
            positions = same_time.position.to_numpy(dtype=int)
            prior[positions] = known
            missing[positions] = float(not seen)
            known = np.maximum(known, labels[positions].max(axis=0))
            seen = True
    static = np.column_stack(
        [
            (age.fillna(median) - median) / std,
            age.isna(),
            sex.where(valid_sex, 0.5),
            ~valid_sex,
            prior,
            missing,
        ]
    ).astype(np.float32)
    groups = pd.DataFrame(
        {
            "ecg_id": df.ecg_id.astype(int),
            "patient_id": df.patient_id.astype(int),
            "split": df.split,
            "age": age,
            "sex": sex.where(valid_sex),
            "history_available": 1 - missing.astype(int),
        }
    )
    for j, label in enumerate(CLASSES):
        groups[label] = labels[:, j].astype(int)
    return (
        static,
        groups,
        {
            "age_median": median,
            "age_std": std,
            "training_observed_age": len(observed),
            "features": FEATURES,
        },
    )


def read_wave(row, cfg):
    fs = cfg["data"]["sampling_rate"]
    field = "filename_hr" if fs == 500 else "filename_lr"
    path = Path(cfg["data"]["root"]) / row[field]
    a, info = wfdb.rdsamp(str(path))
    names = [str(name).upper() for name in info["sig_name"]]
    if info["fs"] != fs or a.shape != (10 * fs, 12):
        raise ValueError(f"Unexpected rate/dimensions: {info['fs']}, {a.shape}")
    if names != [name.upper() for name in LEADS] or any(unit != "mV" for unit in info["units"]):
        raise ValueError("Unexpected lead order or units; expected twelve standard leads in mV")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite waveform values")
    return np.asarray(a, dtype=np.float32)


def audit(cfg):
    df = metadata(cfg["data"]["root"])
    cache = Path(cfg["data"]["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    failures = []
    for i, row in df.iterrows():
        try:
            read_wave(row, cfg)
        except Exception as exc:
            failures.append({"ecg_id": int(row.ecg_id), "error": str(exc)})
        if (i + 1) % 2000 == 0:
            print(f"AUDIT {i + 1}/{len(df)}; failures={len(failures)}", flush=True)
    static, groups, stats = static_features(df, cfg["data"]["root"])
    summary = {
        "protocol": protocol(cfg),
        "records": len(df),
        "patients": int(df.patient_id.nunique()),
        "history_records": int((static[:, -1] == 0).sum()),
        "failures": failures,
        "splits": {
            s: {
                "records": int((df.split == s).sum()),
                "patients": int(df.loc[df.split == s, "patient_id"].nunique()),
            }
            for s in ("train", "val", "test")
        },
        "static_statistics": stats,
    }
    write_json(cache / "audit.json", summary)
    groups.to_csv(cache / "metadata_audit.csv", index=False)
    if failures:
        raise ValueError(f"{len(failures)} invalid records; see {cache / 'audit.json'}")
    print(summary, flush=True)
    return summary


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


def preprocess(cfg):
    started = time.perf_counter()
    d = cfg["data"]
    root, cache = Path(d["root"]), Path(d["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    requested = protocol(cfg)
    if (cache / "manifest.json").exists():
        previous = read_json(cache / "manifest.json")
        if previous["protocol"] != requested:
            raise ValueError("Cache exists for a different protocol; use a new cache directory")
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


def validate_cache(cfg):
    cache = Path(cfg["data"]["cache"])
    manifest = read_json(cache / "manifest.json")
    if manifest["protocol"] != protocol(cfg):
        raise ValueError("Cache protocol mismatch; run preprocess for the requested geometry")
    unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if digest(unsigned) != manifest["fingerprint"]:
        raise ValueError("Cache manifest fingerprint is invalid")
    for filename, expected_hash in manifest["files"].items():
        if file_hash(cache / filename) != expected_hash:
            raise ValueError(f"Cached file changed: {filename}; rerun preprocess")
    root = Path(cfg["data"]["root"])
    for filename, key in (
        ("ptbxl_database.csv", "metadata_sha256"),
        ("scp_statements.csv", "scp_sha256"),
    ):
        if (root / filename).exists() and file_hash(root / filename) != manifest[key]:
            raise ValueError(f"Source metadata changed: {filename}; rerun preprocess")
    patients = []
    records, windows = [], []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(cache / f"metadata_{split}.csv")
        if digest(frame.ecg_id.tolist()) != manifest["splits"][split]["ids_sha256"]:
            raise ValueError(f"Cached record identifiers changed: {split}")
        patients.append(set(frame.patient_id))
        records.append(set(frame.ecg_id))
        if "window_id" in frame:
            if frame.window_id.duplicated().any():
                raise ValueError("Duplicate forecast window identifiers")
            if digest(frame.window_id.tolist()) != manifest["splits"][split]["window_ids_sha256"]:
                raise ValueError("Cached window identifiers changed")
            end = frame.window_start + cfg["data"]["history"] + cfg["data"]["horizon"]
            if (frame.window_start < 0).any() or (end > 10 * cfg["data"]["sampling_rate"]).any():
                raise ValueError("Window exceeds recording")
            windows.append(set(frame.window_id))
    if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Cached patient leakage")
    if any(records[i] & records[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Cached record leakage")
    if windows and any(windows[i] & windows[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Cached window leakage")
    return manifest


class ECGDataset(Dataset):
    def __init__(self, cfg, split, variant="patient_history", limit=None):
        cache = Path(cfg["data"]["cache"])
        self.x = np.load(cache / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(cache / f"y_{split}.npy", mmap_mode="r")
        self.static = np.load(cache / f"static_{split}.npy", mmap_mode="r")
        self.metadata = pd.read_csv(cache / f"metadata_{split}.csv")
        self.patients = self.metadata.patient_id.to_numpy(dtype=np.int64)
        manifest = read_json(cache / "manifest.json")
        self.feature_names = manifest.get("static_statistics", {}).get("features", FEATURES)
        self.indices = feature_indices(cfg, variant, self.feature_names, manifest)
        self.selected_features = [self.feature_names[i] for i in self.indices]
        self.static_dim = len(self.indices)
        self.layout = cfg["data"].get("layout", "channels_first")
        if len(self.x) != len(self.metadata) or self.static.shape != (
            len(self.x),
            len(self.feature_names),
        ):
            raise ValueError("Invalid cached array shapes")
        if self.x.shape[1:] != (12, cfg["data"]["history"]):
            raise ValueError("Cached input geometry mismatch")
        if self.y.shape != (len(self.x), 12, cfg["data"]["horizon"]):
            raise ValueError("Cached target geometry mismatch")
        self.n = min(len(self.x), limit) if limit is not None else len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, y = self.x[i].copy(), self.y[i].copy()
        if self.layout == "time_first":
            x, y = x.T.copy(), y.T.copy()
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "static": torch.from_numpy(self.static[i, self.indices].copy()),
            "patient_id": int(self.patients[i]),
            "index": i,
        }


def make_loaders(cfg, variant, generator=None, limit=None, splits=("train", "val", "test")):
    datasets = {s: ECGDataset(cfg, s, variant, limit) for s in splits}
    tr = cfg["training"]
    return {
        s: DataLoader(
            ds,
            batch_size=tr["batch_size"],
            shuffle=(s == "train"),
            generator=generator if s == "train" else None,
            num_workers=tr["num_workers"],
        )
        for s, ds in datasets.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--root", help="Raw PTB-XL folder")
    parser.add_argument(
        "--audit", action="store_true", help="Check raw ECGs without preparing arrays"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.experiment)
    if args.root:
        config["data"]["root"] = args.root

    with FileLock(".pipeline.lock", timeout=0):
        if args.audit:
            audit(config)
        else:
            preprocess(config)
