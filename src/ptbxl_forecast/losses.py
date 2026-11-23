"""Training objectives (Phase 10). Each maps (prediction, target) of shape
(batch, horizon, leads) to a scalar; L1 is the Phase 5-9 objective.

Auxiliary terms are weighted ``lam * scale``. ``scale`` is fixed once from training data
(``aux_scale``) so that ``lam = 1`` gives the auxiliary term the same size as the L1 term
for a flat forecast, which makes ``lam`` comparable across objectives.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

STFT_SIZES = (16, 32, 64)  # 160, 320, 640 ms windows at 100 Hz


def derivative_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between first differences along time (penalises wrong slopes, e.g. QRS edges)."""
    return F.l1_loss(torch.diff(pred, dim=1), torch.diff(target, dim=1))


def _stft_mag(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    flat = x.transpose(1, 2).reshape(-1, x.shape[1])  # (batch * leads, time)
    window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    spec = torch.stft(flat, n_fft, hop_length=n_fft // 4, window=window, center=True,
                      pad_mode="reflect", return_complex=True)  # fmt: skip
    return spec.abs()


def stft_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT magnitude L1 (insensitive to small time shifts, sensitive to
    missing or blurred waves)."""
    terms = [F.l1_loss(_stft_mag(pred, n), _stft_mag(target, n)) for n in STFT_SIZES]
    return torch.stack(terms).mean()


AUX: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "derivative": derivative_l1,
    "spectral": stft_l1,
}


def aux_scale(name: str, y: np.ndarray, x: np.ndarray) -> float:
    """L1 / aux ratio of the flat (history-median) forecast on training records."""
    flat = np.broadcast_to(np.median(x, axis=1, keepdims=True), y.shape)
    yt, ft = (
        torch.from_numpy(np.ascontiguousarray(y, np.float32)),
        torch.from_numpy(np.ascontiguousarray(flat, np.float32)),
    )
    with torch.no_grad():
        return float(F.l1_loss(ft, yt) / AUX[name](ft, yt))


def make_objective(name: str = "l1", lam: float = 0.0, scale: float = 1.0):
    """L1 (``name='l1'``) or L1 + lam * scale * AUX[name]."""
    if name == "l1":
        return F.l1_loss
    aux, w = AUX[name], lam * scale

    def objective(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target) + w * aux(pred, target)

    return objective
