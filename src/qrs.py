"""QRS peak detection and forecast-vs-truth timing metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from wfdb import processing


def detect_qrs_peaks(signal: np.ndarray, *, fs: int = 100, lead: int = 1) -> np.ndarray:
    """Detect QRS peaks in one physical-unit ECG window.

    The detector operates on one lead because QRS timing is shared across leads.
    Empty results are returned when the detector cannot identify a valid rhythm in
    the supplied window.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2 or not 0 <= lead < signal.shape[1]:
        raise ValueError("Expected signal with shape [time, leads] and a valid lead")
    channel = np.asarray(signal[:, lead], dtype=np.float64)
    if channel.size < fs or not np.isfinite(channel).all() or np.ptp(channel) < 1e-8:
        return np.empty(0, dtype=np.int64)
    try:
        return np.asarray(
            processing.xqrs_detect(channel, fs=fs, verbose=False),
            dtype=np.int64,
        )
    except (ValueError, IndexError, FloatingPointError):
        return np.empty(0, dtype=np.int64)


def match_peaks(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance_samples: int = 10,
) -> tuple[int, int, int, float | None]:
    """Match predicted peaks to truth peaks using a one-to-one greedy match."""
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative")
    predicted = np.sort(np.asarray(predicted, dtype=np.int64))
    truth = np.sort(np.asarray(truth, dtype=np.int64))
    used = np.zeros(len(truth), dtype=bool)
    errors: list[int] = []
    for peak in predicted:
        candidates = np.flatnonzero(~used & (np.abs(truth - peak) <= tolerance_samples))
        if candidates.size:
            candidate = candidates[np.argmin(np.abs(truth[candidates] - peak))]
            used[candidate] = True
            errors.append(abs(int(peak) - int(truth[candidate])))
    true_positives = len(errors)
    false_positives = len(predicted) - true_positives
    false_negatives = len(truth) - true_positives
    mean_error = float(np.mean(errors)) if errors else None
    return true_positives, false_positives, false_negatives, mean_error


def score_forecasts(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    fs: int = 100,
    lead: int = 1,
    tolerance_samples: int = 10,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Detect and score QRS peaks for batched physical-unit forecasts."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have the same shape [records, time, leads]")
    rows = []
    for index, (forecast, actual) in enumerate(zip(prediction, target, strict=True)):
        predicted_peaks = detect_qrs_peaks(forecast, fs=fs, lead=lead)
        truth_peaks = detect_qrs_peaks(actual, fs=fs, lead=lead)
        tp, fp, fn, mean_error = match_peaks(
            predicted_peaks, truth_peaks, tolerance_samples=tolerance_samples
        )
        rows.append(
            {
                "record_index": index,
                "predicted_qrs": len(predicted_peaks),
                "truth_qrs": len(truth_peaks),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "mean_timing_error_samples": mean_error,
            }
        )
    records = pd.DataFrame(rows)
    tp = int(records.true_positives.sum())
    fp = int(records.false_positives.sum())
    fn = int(records.false_negatives.sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    summary = {
        "records": len(records),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_timing_error_samples": (
            float(records.mean_timing_error_samples.dropna().mean())
            if records.mean_timing_error_samples.notna().any()
            else None
        ),
    }
    return records, summary
