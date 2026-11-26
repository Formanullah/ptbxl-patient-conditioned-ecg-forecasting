"""Detector-based morphology evaluation; these are not clinical reference annotations."""

import numpy as np
import pandas as pd
from wfdb import processing


def detect_peaks(voltage, fs):
    voltage = np.asarray(voltage, dtype=float)
    if not np.isfinite(voltage).all():
        raise ValueError("Nonfinite detector input")
    if len(voltage) < fs or np.ptp(voltage) < 1e-8:
        return np.array([], dtype=int), False
    try:
        peaks = processing.xqrs_detect(voltage, fs=fs, verbose=False)
        return np.asarray(peaks, dtype=int), False
    except (ValueError, IndexError, FloatingPointError):
        return np.array([], dtype=int), True


def match_peaks(actual, predicted, tolerance):
    """Ordered one-to-one matching: maximize count, then minimize total timing error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    n, m = len(actual), len(predicted)
    states = [[(0, 0.0, []) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [states[i - 1][j], states[i][j - 1]]
            distance = abs(int(actual[i - 1]) - int(predicted[j - 1]))
            if distance <= tolerance:
                count, error, pairs = states[i - 1][j - 1]
                candidates.append(
                    (
                        count + 1,
                        error + distance,
                        pairs + [(int(actual[i - 1]), int(predicted[j - 1]))],
                    )
                )
            states[i][j] = max(candidates, key=lambda s: (s[0], -s[1]))
    return states[n][m][2]


def peak_counts(actual, predicted, tolerance):
    pairs = match_peaks(actual, predicted, tolerance)
    tp, fp, fn = len(pairs), len(predicted) - len(pairs), len(actual) - len(pairs)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }, pairs


def qrs_metrics(pred, true, observed, fs, tolerance_ms=100, lead=1):
    rows, timing, amplitudes, detections = [], [], [], []
    for i in range(len(pred)):
        offset = observed.shape[-1]
        a, failed_a = detect_peaks(np.r_[observed[i, lead], true[i, lead]], fs)
        b, failed_b = detect_peaks(np.r_[observed[i, lead], pred[i, lead]], fs)
        a, b = a[a >= offset] - offset, b[b >= offset] - offset
        detections.append((a, b))
        if failed_a or failed_b:
            rows.append({"index": i, "detector_failure": True, "tp": 0, "fp": 0, "fn": 0})
            continue
        counts, pairs = peak_counts(a, b, round(tolerance_ms * fs / 1000))
        differences = [abs(x - y) * 1000 / fs for x, y in pairs]
        errors = [abs(float(true[i, lead, x]) - float(pred[i, lead, y])) for x, y in pairs]
        timing.extend(differences)
        amplitudes.extend(errors)
        rows.append(
            {
                "index": i,
                "detector_failure": False,
                **counts,
                "true_peaks": len(a),
                "predicted_peaks": len(b),
                "timing_mae_ms": np.mean(differences) if differences else None,
                "amplitude_mae_mV": np.mean(errors) if errors else None,
            }
        )
    frame = pd.DataFrame(rows)
    tp, fp, fn = [int(frame[k].sum()) for k in ("tp", "fp", "fn")]
    summary = {
        "qrs_precision": tp / (tp + fp) if tp + fp else None,
        "qrs_recall": tp / (tp + fn) if tp + fn else None,
        "qrs_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "peak_timing_mae_ms": float(np.mean(timing)) if timing else None,
        "peak_amplitude_mae_mV": float(np.mean(amplitudes)) if amplitudes else None,
        "matched_peaks": tp,
        "missed_peaks": fn,
        "false_detections": fp,
        "detector_failures": int(frame.detector_failure.sum()),
        "peak_detector": "WFDB XQRS; lead II; common observed prefix; independent futures",
        "peak_tolerance_ms": tolerance_ms,
        "derivative_mae_mV_per_s": float(np.abs(np.diff(pred) - np.diff(true)).mean() * fs),
        "spectral_magnitude_mae": float(
            np.abs(np.abs(np.fft.rfft(pred, axis=-1)) - np.abs(np.fft.rfft(true, axis=-1))).mean()
        ),
    }
    return summary, frame, detections, timing
