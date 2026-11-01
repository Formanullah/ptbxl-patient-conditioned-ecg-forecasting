"""Audited physical-unit WFDB loading and leakage-aware input construction."""

import argparse
import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from filelock import FileLock
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


def preprocess(cfg):
    """Read raw data, save only the requested window, and fit training inputs."""
    if cfg.get("subset"):
        return prepare_subset(cfg)
    d = cfg["data"]
    root, cache = Path(d["root"]), Path(d["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    # Invalidate the completion marker before replacing any cache files.
    (cache / "manifest.json").unlink(missing_ok=True)
    df = metadata(root)
    static, groups, age_stats = static_features(df, root)
    # Keep bounded RAM: ~471 MB at the default geometry, rather than full 10 s records.
    x = np.empty((len(df), 12, d["history"]), dtype=np.float32)
    y = np.empty((len(df), 12, d["horizon"]), dtype=np.float32)
    failures = []
    st, cut = d["start"], d["start"] + d["history"]
    for i, row in df.iterrows():
        try:
            wave = read_wave(row, cfg)
            x[i], y[i] = wave[st:cut].T, wave[cut : cut + d["horizon"]].T
        except Exception as exc:
            failures.append({"ecg_id": int(row.ecg_id), "error": str(exc)})
        if (i + 1) % 2000 == 0:
            print(f"PREPROCESS {i + 1}/{len(df)}", flush=True)
    if failures:
        write_json(cache / "invalid_records.json", failures)
        raise ValueError("Invalid waveforms; see invalid_records.json. No cache was committed.")
    observed = x[df.split.eq("train").to_numpy()]
    mean = observed.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = observed.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    del observed
    x -= mean[None, :, None]
    x /= std[None, :, None]
    y -= mean[None, :, None]
    y /= std[None, :, None]
    manifest = {
        "protocol": protocol(cfg),
        "metadata_sha256": file_hash(root / "ptbxl_database.csv"),
        "scp_sha256": file_hash(root / "scp_statements.csv"),
        "static_statistics": age_stats,
        "mean_mV": mean.tolist(),
        "std_mV": std.tolist(),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        mask = df.split.eq(split).to_numpy()
        for name, array in (("X", x), ("y", y), ("static", static)):
            np.save(cache / f"{name}_{split}.npy", array[mask])
        groups.loc[mask].to_csv(cache / f"metadata_{split}.csv", index=False)
        manifest["splits"][split] = {
            "records": int(mask.sum()),
            "ids_sha256": digest(df.loc[mask, "ecg_id"].tolist()),
        }
    manifest["files"] = {
        path.name: file_hash(path)
        for pattern in (
            "X_*.npy",
            "y_*.npy",
            "static_*.npy",
            "metadata_train.csv",
            "metadata_val.csv",
            "metadata_test.csv",
        )
        for path in cache.glob(pattern)
    }
    manifest["fingerprint"] = digest(manifest)
    # Written last: partially completed caches cannot masquerade as complete ones.
    write_json(cache / "manifest.json", manifest)
    print(f"Prepared cache {cache}; fingerprint={manifest['fingerprint']}", flush=True)
    return manifest


def prepare_subset(cfg):
    """Select whole patients, recover cached mV, and refit scalers on pilot inputs only."""
    source_cfg = copy.deepcopy(cfg)
    source_cfg.pop("subset")
    source_cfg["data"]["cache"] = cfg["subset"]["source_cache"]
    source = validate_cache(source_cfg)
    cache = Path(cfg["data"]["cache"])
    source_path = Path(source_cfg["data"]["cache"])
    if cache.resolve() == source_path.resolve():
        raise ValueError("Subset cache must differ from the full source cache")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").unlink(missing_ok=True)
    full_df = metadata(cfg["data"]["root"])
    rng = np.random.default_rng(cfg["subset"]["seed"])
    frames, xs, ys, selected_patients = [], [], [], {}
    source_mean = np.asarray(source["mean_mV"], dtype=np.float32)[None, :, None]
    source_std = np.asarray(source["std_mV"], dtype=np.float32)[None, :, None]
    for split in ("train", "val", "test"):
        meta = pd.read_csv(source_path / f"metadata_{split}.csv")
        patients = np.sort(meta.patient_id.unique())
        count = cfg["subset"]["patients"][split]
        if not isinstance(count, int) or count < 1 or count > len(patients):
            raise ValueError(f"Invalid subset patient count for {split}: {count}")
        selected = np.sort(rng.permutation(patients)[:count])
        selected_patients[split] = selected.astype(int).tolist()
        mask = meta.patient_id.isin(selected).to_numpy()
        ids = meta.loc[mask, "ecg_id"].tolist()
        frame = full_df.set_index("ecg_id").loc[ids].reset_index()
        frames.append(frame)
        for name, destination in (("X", xs), ("y", ys)):
            array = np.load(source_path / f"{name}_{split}.npy", mmap_mode="r")
            destination.append(np.asarray(array[mask]) * source_std + source_mean)
    df = pd.concat(frames, ignore_index=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    static, groups, stats = static_features(df, cfg["data"]["root"])
    train = df.split.eq("train").to_numpy()
    mean = x[train].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = x[train].std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    x = (x - mean[None, :, None]) / std[None, :, None]
    y = (y - mean[None, :, None]) / std[None, :, None]
    manifest = {
        "protocol": protocol(cfg),
        "metadata_sha256": source["metadata_sha256"],
        "scp_sha256": source["scp_sha256"],
        "static_statistics": stats,
        "mean_mV": mean.tolist(),
        "std_mV": std.tolist(),
        "splits": {},
        "selection": {
            "patients": selected_patients,
            "source_fingerprint": source["fingerprint"],
            "voltage_source": "inverse-transformed source cache; float32 roundoff possible",
        },
    }
    for split in ("train", "val", "test"):
        mask = df.split.eq(split).to_numpy()
        for name, array in (("X", x), ("y", y), ("static", static)):
            np.save(cache / f"{name}_{split}.npy", array[mask])
        groups.loc[mask].to_csv(cache / f"metadata_{split}.csv", index=False)
        manifest["splits"][split] = {
            "records": int(mask.sum()),
            "ids_sha256": digest(df.loc[mask, "ecg_id"].tolist()),
        }
    manifest["files"] = {
        path.name: file_hash(path)
        for pattern in ("*.npy", "metadata_*.csv")
        for path in cache.glob(pattern)
    }
    manifest["fingerprint"] = digest(manifest)
    write_json(cache / "manifest.json", manifest)
    print(f"Patient subset prepared: {manifest['splits']}", flush=True)
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
    for split in ("train", "val", "test"):
        frame = pd.read_csv(cache / f"metadata_{split}.csv")
        if digest(frame.ecg_id.tolist()) != manifest["splits"][split]["ids_sha256"]:
            raise ValueError(f"Cached record identifiers changed: {split}")
        patients.append(set(frame.patient_id))
    if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Cached patient leakage")
    return manifest


class ECGDataset(Dataset):
    def __init__(self, cfg, split, variant="patient_history", limit=None):
        cache = Path(cfg["data"]["cache"])
        self.x = np.load(cache / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(cache / f"y_{split}.npy", mmap_mode="r")
        self.static = np.load(cache / f"static_{split}.npy", mmap_mode="r")
        self.metadata = pd.read_csv(cache / f"metadata_{split}.csv")
        self.patients = self.metadata.patient_id.to_numpy(dtype=np.int64)
        self.static_dim = 0 if variant == "ecg_only" else (4 if variant == "demographics" else 10)
        if len(self.x) != len(self.metadata) or self.static.shape != (len(self.x), 10):
            raise ValueError("Invalid cached array shapes")
        if self.x.shape[1:] != (12, cfg["data"]["history"]):
            raise ValueError("Cached input geometry mismatch")
        if self.y.shape != (len(self.x), 12, cfg["data"]["horizon"]):
            raise ValueError("Cached target geometry mismatch")
        self.n = min(len(self.x), limit) if limit is not None else len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "x": torch.from_numpy(self.x[i].copy()),
            "y": torch.from_numpy(self.y[i].copy()),
            "static": torch.from_numpy(self.static[i, : self.static_dim].copy()),
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
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--root", help="Raw PTB-XL folder")
    parser.add_argument(
        "--audit", action="store_true", help="Check raw ECGs without preparing arrays"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.root:
        config["data"]["root"] = args.root

    with FileLock(".pipeline.lock", timeout=0):
        if args.audit:
            audit(config)
        else:
            preprocess(config)
