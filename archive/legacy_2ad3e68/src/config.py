"""Shared configuration and portable protocol identifiers."""

import copy
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
VARIANTS += ["expanded", "mrmr", "permutation"]
SCHEMA = 1


def merge_config(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge_config(result.get(key, {}), value)
            if isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def read_config_tree(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Circular configuration inheritance")
    seen.add(path)
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # Run snapshots are JSON (also valid YAML); resume their exact executed settings.
    if "config" in cfg and "architecture" in cfg:
        cfg = cfg["config"]
    if "extends" in cfg:
        base = read_config_tree(Path(path).parent / cfg.pop("extends"), seen)
        cfg = merge_config(base, cfg)
    return cfg


def load_config(path, experiment=None):
    cfg = read_config_tree(path)
    matrix = cfg.pop("experiments", None)
    if matrix is not None:
        if experiment not in matrix:
            raise ValueError(f"Choose --experiment from: {', '.join(matrix)}")
        cfg = merge_config(cfg, matrix[experiment])
        cfg.setdefault("experiment", {})["name"] = experiment
    d, tr = cfg["data"], cfg["training"]
    if "history_seconds" in d:
        for seconds, samples in (
            ("history_seconds", "history"),
            ("horizon_seconds", "horizon"),
            ("window_stride_seconds", "stride"),
        ):
            value = float(d[seconds]) * d["sampling_rate"]
            if value <= 0 or abs(value - round(value)) > 1e-6:
                raise ValueError(f"{seconds} must be positive and resolve to whole samples")
            d[samples] = int(round(value))
        d.setdefault("start", 0)
        d.setdefault("layout", "time_first")
        cfg.setdefault("preprocessing", {"filter": "none", "scaling": "zscore", "order": 4})
        cfg.setdefault("features", {})
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
    if "storage" in cfg:
        resolve_storage(cfg)
    return cfg


def resolve_storage(cfg):
    """Give each cohort/window protocol its own outputs and each transform its own cache.

    Existing explicit paths (including saved run snapshots) remain portable and exact.
    Changing model settings requires a new experiment name, not a new data cache.
    """
    settings = protocol(cfg)
    identity = {key: settings[key] for key in ("data", "subset")}
    data = cfg["data"]
    label = (
        f"{cfg['subset']['mode']}_{data['sampling_rate']}hz_"
        f"{data['history_seconds']:g}s_{data['horizon_seconds']:g}s_{digest(identity)[:8]}"
    )
    storage = cfg["storage"]
    transform = digest(cfg["preprocessing"])[:8]
    data.setdefault("cache", str(Path(storage["data"]) / label / transform))
    paths = cfg.setdefault("paths", {})
    for name in ("experiments", "outputs"):
        paths.setdefault(name, str(Path(storage[name]) / label))
    paths.setdefault("documents", storage["documents"])
    cfg["evaluation"].setdefault(
        "selection_path", str(Path(paths["experiments"]) / "selection.json")
    )
    for method in ("mrmr", "permutation"):
        cfg["features"].setdefault(
            method + "_path", str(Path(paths["outputs"]) / f"{method}.json")
        )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol(cfg):
    if "history_seconds" in cfg["data"]:
        return {
            "schema": 2,
            "data": {k: v for k, v in cfg["data"].items() if k not in ("root", "cache")},
            "subset": cfg.get("subset", {"mode": "full", "seed": 42}),
            "preprocessing": cfg["preprocessing"],
            "leads": LEADS,
            "targets": "unfiltered original mV; train-input scaling",
            "history_policy": "strictly earlier timestamps; equal/missing timestamps excluded",
            "train_folds": list(range(1, 9)),
            "val_fold": 9,
            "test_fold": 10,
            "heldout_windows": "one start-zero window per record",
        }
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


def runtime(cfg):
    import torch

    torch.set_num_threads(cfg["training"]["threads"])
    requested = cfg["training"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    return torch.device(device)


def run_directory(cfg, architecture, variant):
    return Path(cfg["paths"]["experiments"]) / cfg.get("experiment", {}).get(
        "name", f"{architecture}_{variant}"
    )
