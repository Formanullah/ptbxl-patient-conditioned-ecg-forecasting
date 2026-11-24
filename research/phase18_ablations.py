"""Phase 18: ablations of the final model on fold 9 (PREREGISTRATION.md). 6 seeds each,
``compare_runs`` against the final configuration's 6 fold-9 runs.

    python research/phase18_ablations.py   # after the training queue (run_12_18.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase9_models as p9  # noqa: E402
import phase10_objectives as p10  # noqa: E402
import phase11_tuning as p11  # noqa: E402

OUT = Path("reports/phase18_ablations")
CACHE = Path(".cache/data/phase18")
SEEDS6, SUITE = p9.SEEDS6, p9.SUITE


def ablations() -> dict:
    c = p11.final_config()
    return {
        "no gate (sn4 + correction)": {**c, "model": "additive"},
        "sn4 as input only (no skip)": {**c, "model": "input"},
        "raw input (no 0.67 Hz decomposition)": {**c, "filter": "none"},
    }


def specs() -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    return [p11.spec(c, 9, s, CACHE) for c in ablations().values() for s in SEEDS6]


def main() -> None:
    from ptbxl_forecast.stats import compare_runs

    OUT.mkdir(parents=True, exist_ok=True)
    c = p11.final_config()
    arms = {"final": [str(p11.run_path(c, 9, s)) for s in SEEDS6]}
    arms.update(
        {n: [str(p11.run_path(a, 9, s, CACHE)) for s in SEEDS6] for n, a in ablations().items()}
    )
    arms["no sn4 (L1 CNN-LSTM, Phase 9)"] = [
        str(p9.run_path("cnn_lstm", p10.LR, False, 9, s)) for s in SEEDS6
    ]
    meta, sig, fold_rows, pids, base = p9.setup_scoring()
    pid = pids[9]
    jobs = {(a, s): (9, [p]) for a in arms for s, p in zip(SEEDS6, arms[a], strict=True)}
    jobs.update({(a, "ens"): (9, arms[a]) for a in arms})
    sc = p9.score_many(jobs, CACHE / "scores.pkl")
    rows, cmps = [], []
    fin = [sc[("final", s)] for s in SEEDS6]
    for a in arms:
        runs = [sc[(a, s)] for s in SEEDS6]
        single = pd.DataFrame([p9.macro(r, pid) for r in runs])
        infos = [p9.info(p) for p in arms[a]]
        rows.append({"arm": a, "params": infos[0].get("params"), **{f"{m}_seed_mean": single[m].mean() for m in SUITE},
                     "share_le_50ms": single.share_timing_le_50ms.mean(),
                     **{f"ens_{m}": v for m, v in p9.macro(sc[(a, "ens")], pid).items()}})  # fmt: skip
        if a != "final":
            cmps.append({"ablation": a, "vs": "final", **p9.cmp_row(compare_runs(runs, fin, pid))})
    cmps.append(
        {
            "ablation": "sn4 alone",
            "vs": "final",
            **p9.cmp_row(compare_runs(fin, [base[("sn4", 9)]], pid)),
        }
    )
    t, comps = pd.DataFrame(rows), pd.DataFrame(cmps)
    beating = comps.iloc[:-1][comps.iloc[:-1].beats.astype(bool)].ablation.tolist()
    # the sn4-alone row is final - sn4; flip its sign so every row reads ablation - final
    last = comps.index[-1]
    for col in [c_ for c_ in comps if c_.startswith("d_")]:
        comps.loc[last, col] = -comps.loc[last, col]
    for m in SUITE:
        lo, hi = f"d_{m}_lo", f"d_{m}_hi"
        comps.loc[last, [lo, hi]] = comps.loc[last, [hi, lo]].to_numpy()
    comps.loc[last, "beats"] = None
    t.to_csv(OUT / "table_ablations.csv", index=False, float_format="%.6g")
    comps.to_csv(OUT / "table_ablation_deltas.csv", index=False, float_format="%.5g")
    d = comps.set_index("ablation")
    S = {
        "H18_1_no_gate_worse_mae": bool(d.loc["no gate (sn4 + correction)", "d_mae_lo"] > 0),
        "H18_2_input_only_worse_timing": bool(d.loc["sn4 as input only (no skip)", "d_beat_timing_ms_lo"] > 0),
        "H18_3_raw_input_mae_ns": bool(d.loc["raw input (no 0.67 Hz decomposition)", "d_mae_lo"] <= 0
                                       <= d.loc["raw input (no 0.67 Hz decomposition)", "d_mae_hi"]),
        "ablations_beating_final": beating,
    }  # fmt: skip
    json.dump(S, (OUT / "summary.json").open("w"), indent=2)
    pd.set_option("display.width", 250)
    print(t.round(5).to_string(), "\n", comps.round(2).to_string(), "\n", json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
