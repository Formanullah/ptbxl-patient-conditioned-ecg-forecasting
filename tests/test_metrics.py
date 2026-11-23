import numpy as np
import pandas as pd
import pytest

from ptbxl_forecast.metrics import beat_f1, beat_timing_ms, dtw_mae, evaluate, pointwise
from ptbxl_forecast.stats import cluster_bootstrap, compare_runs, paired_diff_ci, patient_macro

nk = pytest.importorskip("neurokit2")


@pytest.fixture(scope="module")
def ecg():
    """Simulated 9.9 s ECG at 100 Hz, 12 leads as scaled copies of lead II: (1, 990, 12)."""
    x = nk.ecg_simulate(duration=10, sampling_rate=100, heart_rate=70, noise=0, random_state=1)[
        :990
    ]
    return (x[:, None] * np.linspace(0.5, 1.5, 12)[None, :])[None].astype(np.float64)


def test_perfect_and_flat_forecasts(ecg):
    hist, y = ecg[:, :700], ecg[:, 700:]
    perfect = evaluate(hist, y, y.copy()).iloc[0]
    assert perfect.mae == 0 and perfect.dtw_mae == 0 and perfect.pearson == pytest.approx(1)
    assert perfect.beat_f1 == 1 and perfect.template_mae == 0 and perfect.n_true_beats >= 3
    flat = evaluate(hist, y, np.full_like(y, np.median(hist))).iloc[0]
    assert (
        flat.pearson == 0 and flat.beat_f1 == 0 and flat.template_mae > 0.05
    )  # penalized, not NaN


def test_dtw_tolerates_small_shifts_but_pointwise_does_not(ecg):
    y = ecg[:, 700:]
    shifted = np.roll(ecg, 3, axis=1)[:, 700:]  # 30 ms late
    assert dtw_mae(y, shifted)[0] < 0.3 * pointwise(y, shifted).mae[0]


def test_beat_matching_tolerance():
    assert beat_f1(np.array([10, 100]), np.array([14, 100])) == 1.0  # 40 ms apart: match
    assert beat_f1(np.array([10, 100]), np.array([20, 100])) == 0.5  # 100 ms apart: miss
    assert beat_f1(np.array([], int), np.array([], int)) == 1.0
    true = np.array([50, 130, 210])
    assert beat_timing_ms(true, true) == 0
    assert beat_timing_ms(true + 2, true) == 20  # every beat 20 ms late
    assert beat_timing_ms(np.array([], int), true) == 300  # flat forecast: worst value
    assert 0 < beat_timing_ms(true[[0, 2]], true) < 300  # a missing beat is penalized


def test_patient_macro_weights_patients_equally():
    values, patients = [1.0, 1.0, 1.0, 4.0], [1, 1, 1, 2]
    assert patient_macro(values, patients) == 2.5  # record mean would be 1.75
    mean, lo, hi = cluster_bootstrap(values, patients, n_boot=200)
    assert mean == 2.5 and lo <= mean <= hi


def _runs(offsets, base, **shift):
    """One score table per training run: shared per-patient base + a run-level offset."""
    return [pd.DataFrame({m: base + o + shift.get(m, 0.0) for m in ("mae", "dtw_mae", "template_mae",
                          "beat_timing_ms")}) for o in offsets]  # fmt: skip


def test_compare_runs_accounts_for_run_to_run_variance():
    base = np.random.default_rng(0).normal(0.1, 0.02, 300)
    patients = np.arange(300)
    a = _runs(np.array([-1.0, 1.0, -0.5, 0.5, 2.0, 0.0]) * 1e-3, base)
    b = _runs(np.array([1.0, -1.0, 0.5, -0.5, -2.0, 0.3]) * 1e-3, base)
    mean = lambda runs: np.mean([r.mae for r in runs], axis=0)  # noqa: E731
    assert paired_diff_ci(mean(a), mean(b), patients)[1] > 0  # patients only: spurious "worse"
    c = compare_runs(a, b, patients, metrics=("mae",))["mae"]
    assert c["lo"] < 0 < c["hi"]


def test_compare_runs_detects_effects_and_keeps_gatekeeper_veto():
    base = np.random.default_rng(1).normal(0.1, 0.02, 300)
    offsets = np.array([-1.0, 1.0, -0.5, 0.5, 0.2, 0.0]) * 1e-4
    ref = _runs(offsets, base)
    assert compare_runs(_runs(offsets[::-1], base, mae=-0.005), ref, np.arange(300))[
        "beats_reference"
    ]
    worse_dtw = _runs(offsets[::-1], base, mae=-0.005, dtw_mae=0.005)
    assert not compare_runs(worse_dtw, ref, np.arange(300))["beats_reference"]
