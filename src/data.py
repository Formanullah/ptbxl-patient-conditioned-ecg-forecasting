"""Validate PTB-XL, preserve official patient folds, and serve memory-mapped records."""

import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset

from config import (
    LEADS,
    SCHEMA,
    cohort_limits,
    digest,
    file_hash,
    protocol,
    read_json,
    study_dir,
    write_json,
)
from preprocessing import filtered_path, prepare_transforms

OFFICIAL_METADATA = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}


def metadata(root):
    frame = (
        pd.read_csv(Path(root) / "ptbxl_database.csv").sort_values("ecg_id").reset_index(drop=True)
    )
    required = {"ecg_id", "patient_id", "strat_fold", "filename_lr", "age", "sex"}
    if required - set(frame) or frame.ecg_id.duplicated().any():
        raise ValueError("Missing metadata columns or duplicate ECG IDs")
    if (
        frame[["patient_id", "strat_fold"]].isna().any().any()
        or not frame.strat_fold.isin(range(1, 11)).all()
    ):
        raise ValueError("Invalid patient/fold identifiers")
    if frame.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("Patient appears in different official folds")
    frame["split"] = np.where(
        frame.strat_fold <= 8, "train", np.where(frame.strat_fold == 9, "val", "test")
    )
    for value in frame.filename_lr:
        p = Path(str(value))
        if p.is_absolute() or ".." in p.parts or p.parts[0] != "records100":
            raise ValueError("filename_lr must be a relative records100 path")
    return frame


def select_patients(frame, cfg):
    selected = {}
    limits = cohort_limits(cfg)
    for index, split in enumerate(("train", "val")):
        ids = np.sort(frame.loc[frame.split.eq(split), "patient_id"].unique())
        count = limits[split]
        if count is not None:
            if not 1 <= count <= len(ids):
                raise ValueError(f"Requested {count} {split} patients but only {len(ids)} exist")
            rng = np.random.default_rng(np.random.SeedSequence([cfg["training"]["seed"], index]))
            ids = np.sort(rng.permutation(ids)[:count])
        selected[split] = ids.tolist()
    selected["test"] = sorted(frame.loc[frame.split.eq("test"), "patient_id"].unique().tolist())
    return selected


def source_checksums(root):
    path = Path(root) / "SHA256SUMS.txt"
    if not path.exists():
        raise ValueError("Missing official SHA256SUMS.txt; retain it with the v1.0.3 dataset")
    values = {}
    for line in path.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        values[name.strip().lstrip("*./")] = checksum
    return values


def verify_source(root, name, checksums):
    relative = Path(name).as_posix()
    path = Path(root) / relative
    if relative not in checksums or not path.is_file():
        raise ValueError(f"Missing source/checksum: {relative}")
    if file_hash(path) != checksums[relative]:
        raise ValueError(f"Source checksum mismatch: {relative}")


def read_wave(row, root):
    wave, info = wfdb.rdsamp(str(Path(root) / row.filename_lr))
    if wave.shape != (1000, 12) or info["fs"] != 100:
        raise ValueError(f"Unexpected waveform shape/rate for ECG {row.ecg_id}")
    if [n.upper() for n in info["sig_name"]] != [n.upper() for n in LEADS]:
        raise ValueError("Unexpected lead order")
    if info["units"] != ["mV"] * 12 or not np.isfinite(wave).all():
        raise ValueError("Expected finite physical mV waveforms")
    return wave.astype(np.float32)


def validate_cache(cache, cfg):
    cache = Path(cache)
    manifest = read_json(cache / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["protocol"] != protocol(cfg):
        raise ValueError("Cache protocol mismatch; legacy caches cannot be reused")
    unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if digest(unsigned) != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint changed")
    root = Path(cfg["data"]["root"])
    for filename, checksum in manifest["metadata_hashes"].items():
        if not (root / filename).exists() or file_hash(root / filename) != checksum:
            raise ValueError("Source metadata changed or is unavailable")
    for filename, checksum in manifest["files"].items():
        if not (cache / filename).exists() or file_hash(cache / filename) != checksum:
            raise ValueError(f"Cache file changed: {filename}")
    for filename, checksum in manifest.get("waveform_source_hashes", {}).items():
        if not (root / filename).exists() or file_hash(root / filename) != checksum:
            raise ValueError(f"Source waveform changed: {filename}")
    for split in ("train", "val"):
        waves = np.load(cache / f"waves_{split}.npy", mmap_mode="r")
        if waves.shape != (manifest["splits"][split]["records"], 1000, 12):
            raise ValueError("Cached waveform shape mismatch")
    return manifest


def prepare(cfg, include_test=False, lock=None):
    root = Path(cfg["data"]["root"])
    frame = metadata(root)
    checksums = source_checksums(root)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        verify_source(root, name, checksums)
    selected = select_patients(frame, cfg)
    metadata_hashes = {
        n: file_hash(root / n)
        for n in ("ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt")
    }
    if not cfg["data"].get("synthetic", False):
        if any(metadata_hashes[name] != expected for name, expected in OFFICIAL_METADATA.items()):
            raise ValueError("Metadata does not match the official PTB-XL v1.0.3 release")
        if len(frame) != 21799 or frame.patient_id.nunique() != 18869:
            raise ValueError("Unexpected v1.0.3 metadata population")
    identity = {
        "protocol": protocol(cfg),
        "patients": selected,
        "metadata_hashes": metadata_hashes,
        "synthetic": cfg["data"].get("synthetic", False),
    }
    cohort = digest(identity)
    cache = Path(cfg["paths"]["cache"]) / cohort[:16]
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    exists = manifest_path.exists()
    if exists:
        manifest = validate_cache(cache, cfg)
    else:
        manifest = {
            "schema": SCHEMA,
            "cohort": cohort,
            "protocol": protocol(cfg),
            "metadata_hashes": metadata_hashes,
            "selected_patients": selected,
            "synthetic": cfg["data"].get("synthetic", False),
            "splits": {},
            "files": {},
        }
    if include_test and (
        cfg["run_mode"] != "full"
        or not cfg["evaluation"]["final_test"]
        or lock is None
        or lock.get("cohort") != cohort
    ):
        raise ValueError(
            "Fold 10 requires full mode, explicit final_test and a matching frozen selection"
        )
    if include_test:
        from evaluate import verify_lock

        if verify_lock(cfg, cache) != lock:
            raise ValueError("The supplied selection is not the persisted frozen selection")
    splits = ("train", "val", "test") if include_test else ("train", "val")
    for split in splits:
        rows = frame.loc[
            frame.split.eq(split) & frame.patient_id.isin(selected[split])
        ].reset_index(drop=True)
        if split == "test":
            manifest_test = cache / "test_manifest.json"
            if manifest_test.exists():
                saved = read_json(manifest_test)
                if saved["cohort"] != cohort or saved["counts"]["records"] != len(rows):
                    raise ValueError("Final test cache belongs to another cohort")
                for name, checksum in saved["files"].items():
                    if file_hash(cache / name) != checksum:
                        raise ValueError("Final test cache changed")
                for name, checksum in saved["sources"].items():
                    verify_source(root, name, {name: checksum})
                continue
        elif exists:
            continue
        path = cache / f"waves_{split}.npy"
        waves = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=(len(rows), 1000, 12)
        )
        invalid, sources = [], {}
        for i, row in rows.iterrows():
            try:
                for suffix in (".hea", ".dat"):
                    name = str(row.filename_lr) + suffix
                    verify_source(root, name, checksums)
                    sources[name] = checksums[name]
                waves[i] = read_wave(row, root)
            except Exception as exc:
                invalid.append({"ecg_id": int(row.ecg_id), "reason": str(exc)})
        waves.flush()
        del waves
        if invalid:
            write_json(cache / f"invalid_{split}.json", invalid)
            raise ValueError(f"Invalid {split} records; see {cache}")
        rows.to_csv(cache / f"metadata_{split}.csv", index=False)
        files = {
            name: file_hash(cache / name)
            for name in (f"waves_{split}.npy", f"metadata_{split}.csv")
        }
        counts = {
            "records": len(rows),
            "patients": int(rows.patient_id.nunique()),
            "windows": len(rows),
        }
        if split == "test":
            write_json(
                cache / "test_manifest.json",
                {"cohort": cohort, "files": files, "sources": sources, "counts": counts},
            )
        else:
            manifest["files"].update(files)
            manifest["splits"][split] = counts
            manifest.setdefault("waveform_source_hashes", {}).update(sources)
    if not exists:
        manifest["dataset"] = {
            "records": len(frame),
            "patients": int(frame.patient_id.nunique()),
        }
        manifest["reserved_test"] = {
            "records": int(frame.split.eq("test").sum()),
            "patients": len(selected["test"]),
            "prepared": False,
        }
        manifest["fingerprint"] = digest(manifest)
        write_json(manifest_path, manifest)
    destination = study_dir(cfg) / cfg["run_mode"]
    previous = destination / "cohort.json"
    if previous.exists() and read_json(previous)["cohort"] != cohort:
        raise ValueError("This study/mode already has another cohort; use a new study_id")
    write_json(previous, manifest)
    selected_frame = frame.loc[frame.patient_id.isin(sum(selected.values(), []))].copy()
    columns = ["ecg_id", "patient_id", "strat_fold", "split", "filename_lr"]
    exported = selected_frame[columns].copy()
    exported["history_samples"] = cfg["data"]["history"]
    exported["future_samples"] = cfg["data"]["horizon"]
    exported["waveform_prepared"] = (
        ~exported.split.eq("test") | (cache / "test_manifest.json").exists()
    )
    exported.to_csv(destination / "cohort_manifest.csv", index=False)
    prepare_transforms(cache, cfg, "P0")
    return cache, manifest


class ECGDataset(Dataset):
    """Serve one complete-record forecast window without moving the cache onto GPU."""

    def __init__(self, cache, cfg, split, preprocessing, static):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if split == "test":
            from evaluate import verify_lock

            verify_lock(cfg, cache)
        self.waves = np.load(Path(cache) / f"waves_{split}.npy", mmap_mode="r")
        self.history, self.horizon = cfg["data"]["history"], cfg["data"]["horizon"]
        self.filtered = (
            np.load(filtered_path(cache, cfg, split), mmap_mode="r")
            if preprocessing == "P1"
            else None
        )
        scaler = read_json(Path(cache) / "scaler.json")
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.std = np.asarray(scaler["std"], dtype=np.float32)
        self.static = np.asarray(static, dtype=np.float32)
        self.metadata = pd.read_csv(Path(cache) / f"metadata_{split}.csv")
        if self.static.shape[0] != len(self.waves):
            raise ValueError("Static feature/record count mismatch")

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        raw = self.waves[i]
        x = self.filtered[i] if self.filtered is not None else raw[: self.history]
        y = raw[self.history : self.history + self.horizon]
        return {
            "x": torch.from_numpy(((x - self.mean) / self.std).copy()),
            "y": torch.from_numpy(((y - self.mean) / self.std).copy()),
            "static": torch.from_numpy(self.static[i].copy()),
            "patient_id": int(self.metadata.iloc[i].patient_id),
            "index": i,
        }


def loader(dataset, cfg, training=False, generator=None):
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=training,
        num_workers=cfg["training"]["num_workers"],
        generator=generator,
    )


def download(root, *, records=None):
    """Download only officially listed 100 Hz records and v1.0.3 metadata."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = "https://physionet.org/files/ptb-xl/1.0.3/"

    def fetch(name, expected=None):
        out = root / name
        if out.is_file() and (expected is None or file_hash(out) == expected):
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".partial")
        with (
            urllib.request.urlopen(base + name, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while block := response.read(1 << 16):
                stream.write(block)
        if expected is not None and file_hash(temporary) != expected:
            raise ValueError(f"Downloaded checksum mismatch: {name}")
        temporary.replace(out)

    for name in (
        "SHA256SUMS.txt",
        "ptbxl_database.csv",
        "scp_statements.csv",
        "LICENSE.txt",
        "ptbxl_v103_changelog.txt",
    ):
        fetch(name, OFFICIAL_METADATA.get(name))
    checksums = source_checksums(root)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        verify_source(root, name, checksums)
        if checksums[name] != OFFICIAL_METADATA[name]:
            raise ValueError("Download metadata does not match the official v1.0.3 release")
    # Use pinned file URLs; wfdb.dl_database can prepend a second version.
    listed = metadata(root).filename_lr.tolist()
    selected = set(listed if records is None else records)
    if not selected.issubset(listed):
        raise ValueError("Downloads must be listed in official filename_lr metadata")
    files = [record + ext for record in listed if record in selected for ext in (".hea", ".dat")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(lambda name: fetch(name, checksums[name]), files):
            pass
    return root
