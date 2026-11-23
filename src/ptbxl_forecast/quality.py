"""Pre-registered record-defect rules (Phase 3, decision D3.1).

Deterministic and computed from the signal alone, so the same rules apply to every
fold, including the sealed test fold when it is opened. Rules are evaluated on the
full, untrimmed 10 s record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DROPOUT_MIN_RUN = 50  # samples (0.5 s) of identical consecutive values in any lead
PLATEAU_MIN_RUN = 5  # samples (50 ms) flat at the lead's own max or min ...
PLATEAU_MIN_ABS = 1.0  # ... where |value| >= 1 mV
LEAD_RESIDUAL_MAX = 0.01  # mV RMS for III=II-I, aVR=-(I+II)/2, aVL=I-II/2, aVF=II-I/2


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values in a 1-D boolean array."""
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return int((edges[1::2] - edges[::2]).max())


def limb_lead_residual(x: np.ndarray) -> float:
    """Largest RMS deviation (mV) from the Einthoven/Goldberger relations; x is (samples, 12)."""
    x = x.astype(np.float64)
    i, ii = x[:, 0], x[:, 1]
    derived = (ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2)
    return float(max(np.sqrt(((x[:, 2 + k] - d) ** 2).mean()) for k, d in enumerate(derived)))


def record_defects(x: np.ndarray) -> dict[str, bool]:
    x = np.asarray(x, dtype=np.float64)
    dropout = max(longest_run(np.diff(x[:, j]) == 0) + 1 for j in range(x.shape[1]))
    plateau = max(
        (longest_run(x[:, j] == v) for j in range(x.shape[1])
         for v in (x[:, j].max(), x[:, j].min()) if abs(v) >= PLATEAU_MIN_ABS),
        default=0,
    )  # fmt: skip
    return {
        "dropout": dropout >= DROPOUT_MIN_RUN,
        "plateau": plateau >= PLATEAU_MIN_RUN,
        "lead_relation": limb_lead_residual(x) > LEAD_RESIDUAL_MAX,
    }


def defect_table(signals: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """One row per record with a boolean column per rule and ``any``."""
    rows = [record_defects(x) for x in signals]
    table = pd.DataFrame(rows, index=index)
    table["any"] = table.any(axis=1)
    return table
