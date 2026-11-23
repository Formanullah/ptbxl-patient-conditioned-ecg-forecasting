"""Command-line entry point.

ptbxl-forecast env                                  environment and reproducibility check
ptbxl-forecast config [-c FILE ...] [key=value ...] print resolved config and its hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .config import DEFAULT_CONFIG, config_hash, load_config, to_dict
from .provenance import build_manifest


def _load(args: argparse.Namespace):
    return load_config(args.config or [DEFAULT_CONFIG], args.overrides)


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(OmegaConf.to_yaml(cfg), end="")
    print(f"# config_hash: {config_hash(cfg)}")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    cfg = _load(args)
    manifest = build_manifest(to_dict(cfg), config_hash(cfg), Path(cfg.paths.data))
    print(json.dumps(manifest, indent=2, default=str))
    problems = []
    if manifest["environment"]["torch"] is None:
        problems.append("torch missing: uv sync --extra cpu (or cu126 / cu130)")
    if manifest["uv_lock_sha256"] is None:
        problems.append("uv.lock not found")
    if not manifest["dataset"]["available"]:
        problems.append(f"dataset not found at {cfg.paths.data}: scripts/download_ptbxl.py")
    if manifest["git"].get("dirty"):
        problems.append("git work tree is dirty (allowed for development, not final runs)")
    for p in problems:
        print(f"WARNING: {p}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptbxl-forecast", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("env", cmd_env), ("config", cmd_config)):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", action="append", type=Path, help="YAML file(s), in order")
        p.add_argument("overrides", nargs="*", help="dot-list overrides, e.g. seed=123")
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OmegaConfBaseException, ValueError) as exc:
        print(f"config error: {exc}".splitlines()[0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
