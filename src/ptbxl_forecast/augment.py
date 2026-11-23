"""Training-time augmentations on whole raw records (history + target), batch (B, T, leads).

Applied only to training batches, before splitting into history and target. Effects
that should carry into the forecast (scaling, baseline wander, time warping, mixup)
act on the whole record; input-only corruptions (noise, masking, lead dropout) touch
only the history.
"""

from __future__ import annotations

import math

import torch


def gaussian_noise(rec, h, g, sigma=0.02):
    rec[:, :h] += sigma * torch.randn(rec[:, :h].shape, generator=g)
    return rec


def magnitude(rec, h, g, low=0.8, high=1.2):
    return rec * (low + (high - low) * torch.rand(rec.shape[0], 1, 1, generator=g))


def baseline_wander(rec, h, g, f_lo=0.05, f_hi=0.5, amp=0.3):
    b, t_len, leads = rec.shape
    t = torch.arange(t_len, dtype=rec.dtype) / 100
    f = f_lo + (f_hi - f_lo) * torch.rand(b, 1, 1, generator=g)
    a = amp * torch.rand(b, 1, 1, generator=g) * (0.5 + 0.5 * torch.rand(b, 1, leads, generator=g))
    phase = 2 * math.pi * torch.rand(b, 1, 1, generator=g)
    return rec + a * torch.sin(2 * math.pi * f * t[None, :, None] + phase)


def time_warp(rec, h, g, low=0.95):
    """Slow the whole record by a factor in [low, 1]: x_new(t) = x(s t). Only slowing is
    possible, because speeding up would need samples beyond the end of the record."""
    b, t_len, _ = rec.shape
    out = torch.empty_like(rec)
    for i in range(b):
        s = low + (1 - low) * float(torch.rand(1, generator=g))
        n_src = max(2, round(t_len * s))
        src = rec[i : i + 1, :n_src].transpose(1, 2)
        out[i] = torch.nn.functional.interpolate(
            src, size=t_len, mode="linear", align_corners=True
        )[0].T
    return out


def masking(rec, h, g, n_masks=2, width=10):
    for i in range(rec.shape[0]):
        for _ in range(n_masks):
            s = int(torch.randint(0, h - width, (1,), generator=g))
            rec[i, s : s + width] = 0.0
    return rec


def mixup(rec, h, g, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample((rec.shape[0], 1, 1))
    perm = torch.randperm(rec.shape[0], generator=g)
    return lam * rec + (1 - lam) * rec[perm]


def lead_dropout(rec, h, g):
    leads = torch.randint(0, rec.shape[2], (rec.shape[0],), generator=g)
    rec[torch.arange(rec.shape[0]), :h, leads] = 0.0
    return rec


AUGMENTATIONS = {
    "gaussian_noise": gaussian_noise,
    "magnitude": magnitude,
    "baseline_wander": baseline_wander,
    "time_warp": time_warp,
    "masking": masking,
    "mixup": mixup,
    "lead_dropout": lead_dropout,
}
