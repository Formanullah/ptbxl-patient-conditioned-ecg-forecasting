"""Preprocessing candidates (Phase 5). All operate on the observed history only.

Per-record filters take ``x`` of shape (n, T, leads) and return the same shape (or more
channels for ``decompose``). They never see the forecast target, so they cannot leak
the future (D4.3). "Zero-phase" filters run forwards and backwards over the
*history window only*.

Scalers are fitted on training histories only and record the IDs they were fitted on
(checked with ``leakage.assert_fitted_on``, D4.4). They transform inputs and targets,
and invert predictions back to raw mV before scoring.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage
from scipy import signal as sps

FS = 100
INDEPENDENT = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6


# ------------------------------------------------------------------ filters
def _sos(kind: str, cutoff, order: int):
    return sps.butter(order, cutoff, kind, fs=FS, output="sos")


def causal_iir(x: np.ndarray, kind: str, cutoff, order: int = 2) -> np.ndarray:
    """Causal Butterworth filter; initial state set as if the first sample had always held."""
    sos = _sos(kind, cutoff, order)
    zi = sps.sosfilt_zi(sos)[:, None, :, None] * x[None, :, 0:1, :]  # (sections, n, 2, leads)
    return sps.sosfilt(sos, x, axis=1, zi=zi)[0]


def zero_phase_history(x: np.ndarray, kind: str, cutoff, order: int = 2) -> np.ndarray:
    """Forward-backward Butterworth over the history window only (uses no future samples)."""
    return sps.sosfiltfilt(_sos(kind, cutoff, order), x, axis=1)


def causal_fir_lowpass(x: np.ndarray, cutoff: float = 40, numtaps: int = 31) -> np.ndarray:
    """Causal linear-phase FIR low-pass (constant delay of (numtaps - 1) / 2 samples)."""
    b = sps.firwin(numtaps, cutoff, fs=FS)
    zi = sps.lfilter_zi(b, 1.0)[None, :, None] * x[:, 0:1, :]
    return sps.lfilter(b, 1.0, x, axis=1, zi=zi)[0]


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Soft universal-threshold wavelet denoising per record and lead (history only)."""
    import pywt

    out = np.empty_like(x)
    n_t = x.shape[1]
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            coeffs = pywt.wavedec(x[i, :, j], wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thr = sigma * np.sqrt(2 * np.log(n_t))
            if thr > 0:  # a noiseless (e.g. flat) lead has nothing to threshold
                coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
            out[i, :, j] = pywt.waverec(coeffs, wavelet)[:n_t]
    return out


def median_baseline_removal(x: np.ndarray) -> np.ndarray:
    """Two-stage median filter (200 ms, then 600 ms) baseline estimate subtracted (history only)."""
    base = ndimage.median_filter(x, size=(1, 21, 1), mode="nearest")
    base = ndimage.median_filter(base, size=(1, 61, 1), mode="nearest")
    return x - base


def decompose(x: np.ndarray, cutoff: float = 0.67) -> np.ndarray:
    """Causal baseline component and ECG residual, stacked as 2 x leads channels."""
    base = causal_iir(x, "lowpass", cutoff)
    return np.concatenate([base, x - base], axis=2)


FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda x: x,
    "causal_hp_0.5": lambda x: causal_iir(x, "highpass", 0.5),
    "causal_bp_0.5_40": lambda x: causal_iir(x, "bandpass", (0.5, 40)),
    "zero_phase_bp_0.5_40_history": lambda x: zero_phase_history(x, "bandpass", (0.5, 40)),
    "causal_fir_lp_40": causal_fir_lowpass,
    "wavelet_db4": wavelet_denoise,
    "median_baseline": median_baseline_removal,
    "decompose_0.67": decompose,
}


# ------------------------------------------------------------------ scalers
class Scaler:
    """Affine scaling fitted on training histories. ``center``/``scale`` broadcast over (n, T, leads)."""

    per_record = False

    def fit(self, x: np.ndarray, ids) -> Scaler:
        self.fit_ids = list(ids)
        self._fit(x)
        return self

    def _fit(self, x):
        self.center, self.scale = 0.0, 1.0

    def _params(self, x):
        return self.center, self.scale

    def transform(self, x: np.ndarray) -> np.ndarray:
        c, s = self._params(x)
        return (x - c) / s

    def transform_target(self, y: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
        c, s = self._params(x_raw)
        return (y - c) / s

    def inverse_target(self, y_scaled: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
        c, s = self._params(x_raw)
        return y_scaled * s + c


class GlobalStandard(Scaler):
    def _fit(self, x):
        self.center, self.scale = float(x.mean()), float(x.std())


class LeadStandard(Scaler):
    def _fit(self, x):
        self.center, self.scale = x.mean(axis=(0, 1)), x.std(axis=(0, 1))


class LeadRobust(Scaler):
    def _fit(self, x):
        q25, q50, q75 = np.percentile(x[::4], [25, 50, 75], axis=(0, 1))
        self.center, self.scale = q50, q75 - q25


class LeadMinMax(Scaler):
    def _fit(self, x):
        lo, hi = x.min(axis=(0, 1)), x.max(axis=(0, 1))
        self.center, self.scale = (hi + lo) / 2, (hi - lo) / 2  # maps to [-1, 1]


class InstanceNorm(Scaler):
    """Per-record, per-lead z-score from the record's own history (needs no training data)."""

    per_record = True

    def _params(self, x):
        return x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-3


SCALERS: dict[str, type[Scaler]] = {
    "none": Scaler,
    "global_standard": GlobalStandard,
    "lead_standard": LeadStandard,
    "lead_robust": LeadRobust,
    "lead_minmax": LeadMinMax,
    "instance_history": InstanceNorm,
}


# ------------------------------------------------------------------ leads
def to_independent(x: np.ndarray) -> np.ndarray:
    return x[..., INDEPENDENT]


def from_independent(x8: np.ndarray) -> np.ndarray:
    """Rebuild 12 leads from I, II, V1-V6 with the Einthoven/Goldberger relations."""
    i, ii = x8[..., 0], x8[..., 1]
    derived = np.stack([ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2], axis=-1)
    return np.concatenate([x8[..., :2], derived, x8[..., 2:]], axis=-1)


# ------------------------------------------------------------------ pipeline
class Pipeline:
    """Lead selection -> filter -> scaling, fitted on training histories only.

    Leads are selected before filtering (every filter acts per lead), so 8-lead output
    combines with the decomposition (16 input channels). The input scaler is fitted on
    filtered training histories. The target scaler is fitted on raw training histories,
    because the target stays raw mV. ``outputs`` maps predictions back to raw 12-lead mV.
    """

    def __init__(self, filter_name: str = "none", scaler_name: str = "none", leads8: bool = False):
        self.filter_name, self.scaler_name, self.leads8 = filter_name, scaler_name, leads8
        self._filter = FILTERS[filter_name]

    @classmethod
    def from_config(cls, cfg) -> Pipeline:
        """The project's adopted pipeline (``cfg.preprocessing``)."""
        p = cfg.preprocessing
        return cls(p.filter, p.scaler, p.leads8)

    def _sel(self, x):
        return to_independent(x) if self.leads8 else x

    def fit(self, x_raw: np.ndarray, ids) -> Pipeline:
        cls = SCALERS[self.scaler_name]
        self.in_scaler = cls().fit(self._filter(self._sel(x_raw)), ids)
        self.out_scaler = cls().fit(self._sel(x_raw), ids)
        self.fit_ids = list(ids)
        return self

    def inputs(self, x_raw: np.ndarray) -> np.ndarray:
        return self.in_scaler.transform(self._filter(self._sel(x_raw)))

    def targets(self, y_raw: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
        return self.out_scaler.transform_target(self._sel(y_raw), self._sel(x_raw))

    def outputs(self, pred: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
        y = self.out_scaler.inverse_target(pred, self._sel(x_raw))
        return from_independent(y) if self.leads8 else y

    @property
    def in_channels(self) -> int:
        return (8 if self.leads8 else 12) * (2 if self.filter_name == "decompose_0.67" else 1)

    @property
    def out_leads(self) -> int:
        return 8 if self.leads8 else 12
