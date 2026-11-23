"""Run provenance: code version, environment, lockfile and dataset fingerprints."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGES = (
    "ptbxl-forecast", "numpy", "scipy", "pandas", "wfdb", "omegaconf", "torch", "mlflow",
)  # fmt: skip


def project_root() -> Path:
    """Nearest ancestor of the working directory that contains pyproject.toml."""
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists():
            return p
    return here


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=True
        )
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = _git(root, "status", "--porcelain", "--untracked-files=normal") or ""
    diff = _git(root, "diff", "HEAD", "--binary") or ""
    return {
        "available": True,
        "commit": commit.strip(),
        "branch": (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
        "dirty": bool(status.strip()),
        "n_changed_paths": len(status.splitlines()),
        # Identifies uncommitted tracked changes; untracked files are only counted above.
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    for pkg in PACKAGES:
        try:
            info["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None,
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    except ImportError:
        info["torch"] = None
    return info


def dataset_fingerprint(data_root: Path, sampling_rate: int = 100) -> dict[str, Any]:
    """Fingerprint from the official checksum list, restricted to the files this study uses.

    Cheap (no signal files are read). Whether local files match the list is verified
    separately by ``scripts/download_ptbxl.py --verify-only``.
    """
    sums = data_root / "SHA256SUMS.txt"
    if not sums.exists():
        return {"available": False, "root": str(data_root)}
    used = sorted(
        line
        for line in sums.read_text().splitlines()
        if line.split(maxsplit=1)[-1].startswith(f"records{sampling_rate}/")
        or line.split(maxsplit=1)[-1] in ("ptbxl_database.csv", "scp_statements.csv")
    )
    return {
        "available": True,
        "sha256sums_sha256": sha256_file(sums),
        "used_files": len(used),
        "used_files_sha256": hashlib.sha256("\n".join(used).encode()).hexdigest(),
    }


def build_manifest(cfg_dict: dict[str, Any], cfg_hash: str, data_root: Path) -> dict[str, Any]:
    root = project_root()
    lock = root / "uv.lock"
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": cfg_hash,
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "git": git_info(root),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "environment": environment_info(),
        "dataset": dataset_fingerprint(data_root, cfg_dict["data"]["sampling_rate"]),
    }
