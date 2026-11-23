import json
from pathlib import Path

import pytest

from ptbxl_forecast.config import load_config
from ptbxl_forecast.provenance import dataset_fingerprint
from ptbxl_forecast.tracking import Run, RunExistsError, load_runs

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def make_cfg(tmp_path, *overrides):
    return load_config([BASE], [f"paths.results={tmp_path / 'results'}", *overrides])


def test_run_writes_complete_record(tmp_path):
    cfg = make_cfg(tmp_path)
    with Run(cfg) as run:
        for epoch in range(3):
            run.log_metrics({"train_loss": 1.0 / (epoch + 1)}, step=epoch)
        run.log_summary(val_mae=0.1)
    files = {p.name for p in run.dir.iterdir()}
    assert {"config.yaml", "manifest.json", "metrics.jsonl", "summary.json", "run.log"} <= files
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["config_hash"] == run.hash
    assert {"git", "environment", "dataset", "uv_lock_sha256"} <= manifest.keys()
    assert len((run.dir / "metrics.jsonl").read_text().splitlines()) == 3


def test_completed_run_cannot_be_overwritten(tmp_path):
    cfg = make_cfg(tmp_path)
    with Run(cfg):
        pass
    with pytest.raises(RunExistsError):
        with Run(cfg, resume=True):
            pass


def test_failed_run_is_recorded_and_needs_explicit_resume(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(ZeroDivisionError):
        with Run(cfg) as run:
            _ = 1 / 0
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "ZeroDivisionError" in manifest["error"]
    with pytest.raises(RunExistsError):
        with Run(cfg):
            pass
    with Run(cfg, resume=True):
        pass


def test_load_runs_collects_seeds(tmp_path):
    for seed in (42, 123, 999):
        with Run(make_cfg(tmp_path, f"seed={seed}")) as run:
            run.log_summary(val_mae=seed / 1000)
    df = load_runs(tmp_path / "results", "phase1_foundation")
    assert sorted(df["cfg.seed"]) == [42, 123, 999]
    assert (df.status == "completed").all() and df.config_hash.nunique() == 3


def test_run_applies_seed_and_runtime(tmp_path):
    torch = pytest.importorskip("torch")
    draws = []
    for name in ("a", "b"):  # different labels, same scientific config
        with Run(make_cfg(tmp_path, f"run_name={name}", "runtime.threads=2")) as run:
            draws.append((run.rng.random(), torch.rand(1).item()))
        manifest = json.loads((run.dir / "manifest.json").read_text())
        assert manifest["environment"]["torch"]["num_threads"] == 2
        assert manifest["environment"]["torch"]["deterministic_algorithms"] is True
    assert draws[0] == draws[1]


def test_mlflow_mirror(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    cfg = make_cfg(tmp_path, "tracking.mlflow=true", f"tracking.mlflow_uri={uri}")
    with Run(cfg) as run:
        run.log_metrics({"loss": 0.5}, step=0)
        run.log_summary(val_mae=0.1)
    mlflow.set_tracking_uri(uri)
    found = mlflow.search_runs(experiment_names=["phase1_foundation"])
    assert len(found) == 1
    assert found["tags.config_hash"].iloc[0] == run.hash
    assert found["metrics.final.val_mae"].iloc[0] == pytest.approx(0.1)


def test_dataset_fingerprint_uses_only_study_files(tmp_path):
    lines = ["a  ptbxl_database.csv", "b  records100/00000/00001_lr.dat", "c  records500/x.dat"]
    (tmp_path / "SHA256SUMS.txt").write_text("\n".join(lines))
    fp = dataset_fingerprint(tmp_path, 100)
    assert fp["available"] and fp["used_files"] == 2
    assert not dataset_fingerprint(tmp_path / "missing")["available"]
