"""Typed experiment configuration: YAML files + dot-list overrides validated by a schema.

Unknown keys and wrong types are errors, so a typo cannot silently fall back to a
default. The config hash identifies a run: it covers every resolved value except
labels and machine-specific settings (names, paths, logging, tracking), so the same
experiment gets the same hash on every machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path("configs/base.yaml")

# Keys excluded from the scientific identity of a run: labels and machine-specific settings.
NON_SCIENTIFIC = ("study_id", "run_name", "paths", "logging", "tracking")


@dataclass
class DataConfig:
    version: str = "1.0.3"
    sampling_rate: int = 100
    history: int = 700
    horizon: int = 290
    # Samples removed from the end of every record before windowing: the official 100 Hz
    # files carry a 500->100 Hz conversion artifact in their last 10 samples (D3.4).
    trim_end: int = 10
    # Drop records failing the pre-registered defect rules (D3.1; ptbxl_forecast.quality).
    exclude_defects: bool = True
    train_folds: list[int] = field(default_factory=lambda: list(range(1, 9)))
    val_folds: list[int] = field(default_factory=lambda: [9])
    test_folds: list[int] = field(default_factory=lambda: [10])
    allow_test: bool = False


@dataclass
class PreprocessingConfig:
    # Model-input transform (ptbxl_forecast.preprocessing.FILTERS); the target stays raw
    # mV. decompose_0.67 = causal baseline + residual channels (D5.11).
    filter: str = "decompose_0.67"
    scaler: str = "none"  # D5.5
    leads8: bool = False  # D5.7


@dataclass
class RuntimeConfig:
    device: str = "auto"
    threads: int = 4  # 0 = library default (machine-dependent results)
    num_workers: int = 0
    deterministic: bool = True


@dataclass
class PathsConfig:
    data: str = "PTB-XL"
    results: str = "results"
    cache: str = ".cache/data"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class TrackingConfig:
    mlflow: bool = False
    mlflow_uri: str = "sqlite:///results/mlflow.db"
    require_clean_git: bool = False


@dataclass
class Config:
    study_id: str = "default"
    run_name: str = "run"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def validate(cfg: DictConfig) -> None:
    d = cfg.data
    if d.history <= 0 or d.horizon <= 0 or d.trim_end < 0:
        raise ValueError("data.history and data.horizon must be positive, trim_end >= 0")
    if d.history + d.horizon > 10 * d.sampling_rate - d.trim_end:
        raise ValueError("history + horizon exceeds the 10 s record minus data.trim_end")
    groups = {"train": set(d.train_folds), "val": set(d.val_folds), "test": set(d.test_folds)}
    if not set().union(*groups.values()) <= set(range(1, 11)):
        raise ValueError("folds must be in 1..10")
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if groups[a] & groups[b]:
                raise ValueError(f"{a} and {b} folds overlap: {sorted(groups[a] & groups[b])}")
    device = str(cfg.runtime.device)
    if device not in {"auto", "cpu", "cuda", "mps"} and not device.startswith("cuda:"):
        raise ValueError(f"unknown runtime.device {device!r}")
    from .preprocessing import FILTERS, SCALERS

    p = cfg.preprocessing
    if p.filter not in FILTERS or p.scaler not in SCALERS:
        raise ValueError(f"unknown preprocessing filter {p.filter!r} or scaler {p.scaler!r}")


def load_config(
    files: Sequence[str | Path] = (DEFAULT_CONFIG,), overrides: Sequence[str] = ()
) -> DictConfig:
    """Merge schema defaults <- YAML files (in order) <- `a.b=value` overrides."""
    cfg = OmegaConf.structured(Config)
    for f in files:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    validate(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def config_hash(cfg: DictConfig, length: int = 12) -> str:
    """Stable hash of the scientifically relevant config (key order does not matter)."""
    content = {k: v for k, v in to_dict(cfg).items() if k not in NON_SCIENTIFIC}
    blob = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
