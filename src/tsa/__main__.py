"""Run `python -m tsa --help` for the shared local/Colab command interface."""

import argparse

from filelock import FileLock, Timeout

from .config import VARIANTS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "audit",
            "preprocess",
            "benchmark",
            "smoke",
            "train",
            "evaluate",
            "baselines",
            "run-experiments",
            "compare",
            "report",
            "bundle",
        ],
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--model", choices=["lstm", "cnn_lstm"], default="lstm")
    parser.add_argument("--variant", choices=VARIANTS, default="patient_history")
    parser.add_argument("--root", help="Override the raw PTB-XL directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root
    if args.device:
        cfg["training"]["device"] = args.device
    writers = {
        "audit",
        "preprocess",
        "benchmark",
        "smoke",
        "train",
        "evaluate",
        "baselines",
        "run-experiments",
    }
    if args.command in writers:
        try:
            with FileLock(".pipeline.lock", timeout=0):
                return dispatch(args, cfg)
        except Timeout:
            raise SystemExit(
                "Another preparation/training job is active. Inspect it with "
                "tools/status.py using the matching configuration; do not start a duplicate job."
            ) from None
    return dispatch(args, cfg)


def dispatch(args, cfg):
    if args.command in ("audit", "preprocess"):
        from . import data

        getattr(data, args.command)(cfg)
    elif args.command in (
        "train",
        "evaluate",
        "smoke",
        "baselines",
        "benchmark",
        "run-experiments",
    ):
        from .training import baselines, benchmark, evaluate_saved, train

        if args.command == "evaluate":
            evaluate_saved(cfg, args.model, args.variant)
        elif args.command in ("train", "smoke"):
            # train resumes automatically and evaluates once; completed runs are reused.
            train(cfg, args.model, args.variant, smoke=args.command == "smoke")
        elif args.command == "baselines":
            baselines(cfg)
        elif args.command == "benchmark":
            benchmark(cfg)
        else:
            baselines(cfg)
            for architecture in ("lstm", "cnn_lstm"):
                for variant in VARIANTS:
                    train(cfg, architecture, variant)
            from .reporting import report

            report(cfg)
    elif args.command == "compare":
        from .evaluation import compare_runs

        print(compare_runs(cfg).to_string(index=False))
    else:
        from .reporting import bundle, report

        (bundle if args.command == "bundle" else report)(cfg)


if __name__ == "__main__":
    main()
