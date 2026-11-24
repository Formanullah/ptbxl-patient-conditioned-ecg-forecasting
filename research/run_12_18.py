"""Run the training queue of Phases 12, 16 and 18 in one worker pool, then the analyses.

python research/run_12_18.py           # train, then analyse
python research/run_12_18.py --train   # training only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase11_tuning as p11  # noqa: E402
import phase12_robustness as p12  # noqa: E402
import phase16_final as p16  # noqa: E402
import phase18_ablations as p18  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    specs = [*p16.specs(), *p12.specs(), *p18.specs()]
    seen, queue = set(), []
    for s in specs:
        if s["path"] not in seen:
            seen.add(s["path"])
            queue.append(s)
    minutes = p11.run_many(queue, log=lambda m: print(m, flush=True))
    print(f"training finished in {minutes:.0f} min", flush=True)
    if args.train:
        return
    for script in (
        "phase16_final.py",
        "phase12_robustness.py",
        "phase13_15_analysis.py",
        "phase18_ablations.py",
    ):
        print("===", script, flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / script)], check=True)


if __name__ == "__main__":
    main()
