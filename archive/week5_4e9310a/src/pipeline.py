"""Small CLI over the same functions used by the Colab notebook."""

import argparse
import json
from pathlib import Path

from filelock import FileLock

from config import load_config, study_dir
from data import download, prepare
from evaluate import baseline, evaluate_run, final_evaluation, freeze_selection
from fixtures import fixture_config
from leakage_audit import audit
from report import bundle, generate
from study import feature_extension, full, screen, smoke, tune
from train import train_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "download",
            "audit",
            "train",
            "evaluate",
            "freeze",
            "final-test",
            "smoke",
            "screen",
            "tune",
            "features",
            "full",
            "report",
            "export",
        ],
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["pilot", "medium", "full"])
    parser.add_argument("--model", choices=["lstm", "cnn_lstm"], default="lstm")
    parser.add_argument("--preprocessing", choices=["P0", "P1"], default="P0")
    parser.add_argument("--features", choices=["F0", "F1", "F2", "expanded"], default="F0")
    parser.add_argument("--run", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-final-test", action="store_true")
    parser.add_argument(
        "--fixture-dir", type=Path, help="Synthetic software checks only; never PTB-XL performance"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.mode)
    if args.fixture_dir:
        cfg = fixture_config(args.fixture_dir, cfg)
        if args.mode:
            cfg["run_mode"] = args.mode
    if args.enable_final_test:
        cfg["evaluation"]["final_test"] = True
    root = study_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".pipeline.lock"), timeout=0):
        if args.stage == "download":
            return str(download(cfg["data"]["root"]))
        if args.stage in ("smoke", "screen", "tune", "features", "full"):
            functions = {
                "smoke": smoke,
                "screen": screen,
                "tune": tune,
                "features": feature_extension,
                "full": full,
            }
            result = functions[args.stage](cfg, resume=args.resume)
            cfg["run_mode"] = (
                "pilot" if args.stage == "smoke" else ("full" if args.stage == "full" else "medium")
            )
            generate(cfg)
            print(json.dumps(result, indent=2))
            return result
        if args.stage == "report":
            return str(generate(cfg))
        if args.stage == "export":
            return str(bundle(cfg))
        cache, manifest = prepare(cfg)
        if args.stage == "prepare":
            return {"cache": str(cache), "splits": manifest["splits"]}
        if args.stage == "audit":
            return audit(cfg, cache)
        if args.stage == "train":
            return str(
                train_run(
                    cfg, cache, args.model, args.preprocessing, args.features, resume=args.resume
                )
            )
        if args.stage == "evaluate":
            if args.run:
                return [
                    evaluate_run(cfg, cache, root / cfg["run_mode"] / "checkpoints" / name)
                    for name in args.run
                ]
            return [baseline(cfg, cache, name) for name in ("persistence", "repeat_window")]
        if args.stage == "freeze":
            if not args.run:
                raise ValueError("freeze requires the four run names")
            return freeze_selection(
                cfg, cache, [root / "full" / "checkpoints" / name for name in args.run]
            )
        if args.stage == "final-test":
            return final_evaluation(cfg, cache)


if __name__ == "__main__":
    try:
        value = main()
        if value is not None:
            print(value if isinstance(value, str) else json.dumps(value, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"Pipeline stopped: {exc}") from exc
