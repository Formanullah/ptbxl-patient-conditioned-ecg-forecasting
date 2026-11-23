"""Experiment tracking.

Plain files are the source of truth (one folder per run, readable without any
tracking server). MLflow, if enabled, mirrors the same parameters and metrics for
interactive browsing (``mlflow ui --backend-store-uri sqlite:///results/mlflow.db``).

    results/<study_id>/<run_name>_<config_hash>/
        config.yaml     resolved configuration
        manifest.json   provenance + status (running/completed/failed) + timing
        metrics.jsonl   one JSON object per log_metrics call
        summary.json    final metrics
        run.log         full log
        artifacts/      figures, predictions, checkpoints
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .config import config_hash, flatten, to_dict
from .logs import setup_logging
from .provenance import build_manifest
from .seeding import configure_torch, seed_everything

log = logging.getLogger(__name__)


class RunExistsError(RuntimeError):
    pass


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_dir(cfg: DictConfig) -> Path:
    return Path(cfg.paths.results) / cfg.study_id / f"{cfg.run_name}_{config_hash(cfg)}"


def run_status(cfg: DictConfig) -> str | None:
    manifest = run_dir(cfg) / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("status")


class Run:
    """Context manager for one training/evaluation run.

    Entering seeds all RNGs from ``cfg.seed``, applies ``cfg.runtime`` (threads,
    determinism, device -> ``run.device``) and records provenance. ``run.rng`` is a
    seeded NumPy Generator. A completed run with the same scientific config cannot be
    overwritten; an interrupted (running/failed) run reopens only with ``resume=True``.
    """

    def __init__(self, cfg: DictConfig, resume: bool = False):
        self.cfg = cfg
        self.hash = config_hash(cfg)
        self.dir = run_dir(cfg)
        self.resume = resume
        self.summary: dict[str, Any] = {}
        self._mlflow = None
        self._t0 = 0.0

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Run:
        status = run_status(self.cfg)
        if status == "completed":
            raise RunExistsError(f"{self.dir} already completed; change config or study_id")
        if status is not None and not self.resume:
            raise RunExistsError(f"{self.dir} exists with status {status!r}; use resume=True")

        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        setup_logging(self.cfg.logging.level, self.dir / "run.log")
        OmegaConf.save(self.cfg, self.dir / "config.yaml")
        # Seed and configure before recording provenance, so the manifest shows the
        # settings actually in effect.
        self.rng = seed_everything(self.cfg.seed, self.cfg.runtime.deterministic)
        try:
            self.device = configure_torch(self.cfg.runtime.threads, self.cfg.runtime.device)
        except ImportError:
            self.device = "cpu"
        self.manifest = build_manifest(to_dict(self.cfg), self.hash, Path(self.cfg.paths.data))
        if self.cfg.tracking.require_clean_git and self.manifest["git"].get("dirty", True):
            raise RuntimeError("tracking.require_clean_git is set but the work tree is dirty")
        self.manifest.update(
            status="running", run_dir=str(self.dir), resumed=status is not None, device=self.device
        )
        _write_json(self.dir / "manifest.json", self.manifest)
        if self.cfg.tracking.mlflow:
            self._start_mlflow()
        self._t0 = time.perf_counter()
        log.info("run %s started (%s)", self.dir.name, self.dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        failed = exc_type is not None
        self.manifest.update(
            status="failed" if failed else "completed",
            duration_s=round(time.perf_counter() - self._t0, 3),
            error=f"{exc_type.__name__}: {exc}" if failed else None,
        )
        _write_json(self.dir / "manifest.json", self.manifest)
        if self._mlflow is not None:
            for name in ("config.yaml", "manifest.json", "summary.json"):
                if (self.dir / name).exists():
                    self._mlflow.log_artifact(str(self.dir / name))
            self._mlflow.end_run(status="FAILED" if failed else "FINISHED")
        log.info("run %s %s", self.dir.name, self.manifest["status"])

    # ------------------------------------------------------------------ logging
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        record = {"step": step, "time": round(time.perf_counter() - self._t0, 3), **metrics}
        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self._mlflow is not None:
            finite = {k: float(v) for k, v in metrics.items() if math.isfinite(float(v))}
            self._mlflow.log_metrics(finite, step=step)

    def log_summary(self, **values: Any) -> None:
        self.summary.update(values)
        _write_json(self.dir / "summary.json", self.summary)
        if self._mlflow is not None:
            numeric = {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics({f"final.{k}": v for k, v in numeric.items()})

    def artifact(self, name: str) -> Path:
        return self.dir / "artifacts" / name

    # ------------------------------------------------------------------ mlflow mirror
    def _start_mlflow(self) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "tracking.mlflow=true needs MLflow: uv sync --extra cpu --extra tracking"
            ) from exc
        mlflow.set_tracking_uri(self.cfg.tracking.mlflow_uri)
        mlflow.set_experiment(self.cfg.study_id)
        mlflow.start_run(run_name=self.dir.name)
        params = {k: str(v)[:500] for k, v in flatten(to_dict(self.cfg)).items()}
        mlflow.log_params(params)
        git = self.manifest["git"]
        mlflow.set_tags(
            {
                "config_hash": self.hash,
                "git_commit": git.get("commit", "unknown"),
                "git_dirty": str(git.get("dirty", "unknown")),
                "run_dir": str(self.dir),
            }
        )
        self._mlflow = mlflow


def load_runs(results: str | Path, study_id: str) -> Any:
    """All runs of a study as a DataFrame: flattened config, status and summary metrics."""
    import pandas as pd

    rows = []
    for manifest_path in sorted((Path(results) / study_id).glob("*/manifest.json")):
        d = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = OmegaConf.to_container(OmegaConf.load(d / "config.yaml"))
        summary_path = d / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        )
        rows.append(
            {
                "run": d.name,
                "status": manifest.get("status"),
                "config_hash": manifest.get("config_hash"),
                "git_commit": manifest.get("git", {}).get("commit"),
                "duration_s": manifest.get("duration_s"),
                **{f"cfg.{k}": v for k, v in flatten(cfg).items()},
                **summary,
            }
        )
    return pd.DataFrame(rows)
