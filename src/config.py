"""Configuration, atomic metadata writes, and scientific protocol identities."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

SCHEMA = 3
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FEATURE_SETS = ("F0", "F1", "F2", "expanded", "mrmr", "permutation")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_config(path="configs/config.yaml", mode=None):
    with Path(path).open(encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("schema") != SCHEMA:
        raise ValueError("Legacy configuration: use archive/legacy_2ad3e68/src and its run guide")
    cfg = copy.deepcopy(cfg)
    if mode:
        cfg["run_mode"] = mode
    validate(cfg)
    return cfg


def validate(cfg):
    d, tr = cfg["data"], cfg["training"]
    if cfg["run_mode"] not in ("pilot", "medium", "full"):
        raise ValueError("run_mode must be pilot, medium or full")
    if d["sampling_rate"] != 100 or (d["history"], d["horizon"]) not in ((700, 300), (500, 500)):
        raise ValueError("Use native 100 Hz and either 700/300 or the separate 500/500 ablation")
    if tr["epochs"] < 1 or tr["batch_size"] < 1 or tr["lr"] <= 0:
        raise ValueError("Invalid training budget, batch size or learning rate")
    if cfg["model"]["depth"] not in (1, 2):
        raise ValueError("Use one or two recurrent layers")
    p = cfg["preprocessing"]
    if not 0 < p["low_hz"] < p["high_hz"] < 50 or p["order"] < 1:
        raise ValueError("Invalid bandpass settings")


def protocol(cfg):
    return {
        "schema": SCHEMA,
        "dataset": "PTB-XL",
        "version": "1.0.3",
        "fs": 100,
        "history": cfg["data"]["history"],
        "horizon": cfg["data"]["horizon"],
        "start": 0,
        "windows_per_record": 1,
        "leads": LEADS,
        "target": "original unfiltered mV",
        "scaler": "raw training history per-lead zscore",
        "folds": {"train": list(range(1, 9)), "val": [9], "test": [10]},
    }


def cohort_limits(cfg):
    mode = cfg["run_mode"]
    return cfg["cohorts"][mode] if mode != "full" else {"train": None, "val": None}


def study_dir(cfg):
    return Path(cfg["paths"]["results"]) / cfg["study_id"]


def source_hashes():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}


def device_for(cfg):
    import torch

    requested = cfg["training"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a GPU runtime or device=cpu")
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    device = "cpu" if device == "auto" else device
    if cfg["run_mode"] == "full" and device != "cuda" and not cfg["training"]["allow_full_cpu"]:
        raise RuntimeError(
            "Full training requires CUDA; explicitly enable allow_full_cpu to override"
        )
    return torch.device(device)
