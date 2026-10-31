"""Shared configuration and portable protocol identifiers."""

import hashlib
import json
from pathlib import Path

import yaml

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FEATURES = (
    ["age_scaled", "age_missing", "sex_encoded", "sex_missing"]
    + ["prior_" + label for label in CLASSES]
    + ["history_missing"]
)
VARIANTS = ["ecg_only", "demographics", "patient_history", "patient_history_contrastive"]
SCHEMA = 1


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    d, tr = cfg["data"], cfg["training"]
    if d["sampling_rate"] not in (100, 500):
        raise ValueError("sampling_rate must be 100 or 500")
    if min(d["history"], d["horizon"]) <= 0 or d["start"] < 0:
        raise ValueError("Invalid input/target geometry")
    if d["start"] + d["history"] + d["horizon"] > 10 * d["sampling_rate"]:
        raise ValueError("Window exceeds the ten-second record")
    if min(tr["epochs"], tr["patience"], tr["batch_size"], tr["threads"]) < 1:
        raise ValueError("Training counts must be positive")
    if tr["lr"] <= 0 or cfg["contrastive"]["temperature"] <= 0:
        raise ValueError("Learning rate and temperature must be positive")
    return cfg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol(cfg):
    result = {
        "schema": SCHEMA,
        **{k: cfg["data"][k] for k in ("sampling_rate", "history", "horizon", "start")},
        "leads": LEADS,
        "train_folds": list(range(1, 9)),
        "val_fold": 9,
        "test_fold": 10,
        "normalization": "per_lead_training_inputs_only",
        "history_policy": "strictly_earlier_timestamp_union_scp_diagnostic_keys",
    }
    if cfg.get("subset"):
        result["subset"] = {
            "seed": cfg["subset"]["seed"],
            "patients": cfg["subset"]["patients"],
            "selection": "uniform_random_patients_with_all_their_records_within_official_folds",
        }
    return result


def write_json(path, value):
    """Write standards-compliant JSON atomically; undefined metrics use null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
