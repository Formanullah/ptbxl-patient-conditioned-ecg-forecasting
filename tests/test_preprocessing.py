import numpy as np
import pandas as pd
import pytest

from ptbxl_forecast.leakage import causality_violation
from ptbxl_forecast.models.ridge import RidgeForecaster
from ptbxl_forecast.preprocessing import (
    FILTERS,
    SCALERS,
    Pipeline,
    causal_iir,
    from_independent,
    to_independent,
)
from ptbxl_forecast.stats import compare


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    rec = rng.normal(0, 0.2, (40, 990, 12)).cumsum(axis=1) * 0.05
    i, ii = rec[..., 0], rec[..., 1]
    rec[..., 2], rec[..., 3], rec[..., 4], rec[..., 5] = (
        ii - i,
        -(i + ii) / 2,
        i - ii / 2,
        ii - i / 2,
    )
    return rec


def test_causal_filters_are_causal_and_zero_phase_is_not(records):
    x = records[0, :700]
    hp = lambda r: causal_iir(r[None], "highpass", 0.5)[0]  # noqa: E731
    assert causality_violation(hp, x, 500) == 0
    assert causality_violation(lambda r: FILTERS["causal_fir_lp_40"](r[None])[0], x, 500) == 0
    # Applied to the history only this uses no future samples, but within the window it is two-sided.
    assert (
        causality_violation(lambda r: FILTERS["zero_phase_bp_0.5_40_history"](r[None])[0], x, 500)
        > 0
    )


@pytest.mark.parametrize("scaler", list(SCALERS))
@pytest.mark.parametrize("leads8", [False, True])
@pytest.mark.parametrize("filt", ["causal_bp_0.5_40", "decompose_0.67"])
def test_pipeline_round_trip_to_raw_mv(records, scaler, leads8, filt):
    x, y = records[:, :700], records[:, 700:]
    p = Pipeline(filt, scaler, leads8).fit(x[:30], ids=range(30))
    assert p.fit_ids == list(range(30))
    back = p.outputs(p.targets(y[30:], x[30:]), x[30:])
    np.testing.assert_allclose(back, y[30:], atol=1e-9)
    assert p.inputs(x[30:]).shape[-1] == p.in_channels


def test_independent_leads_round_trip(records):
    np.testing.assert_allclose(from_independent(to_independent(records)), records, atol=1e-12)


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 20))
    w = rng.normal(size=(20, 5))
    y = x @ w + 0.01 * rng.normal(size=(400, 5))
    model = RidgeForecaster(lambdas=np.logspace(-4, 2, 7)).fit(x, y, groups=np.arange(400))
    assert np.abs(model.predict(x) - y).mean() < 0.02 and model.lam <= 1


def test_ridge_extends_grid_when_optimum_is_at_edge():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 50)) * 100  # large-scale inputs need a large penalty
    y = rng.normal(size=(300, 3))  # pure noise: more shrinkage is always better
    model = RidgeForecaster(lambdas=np.logspace(-2, 1, 4)).fit(x, y, groups=np.arange(300))
    assert model.lam > 10 and len(model.lambdas_tried) > 4


def test_compare_requires_gatekeepers():
    rng = np.random.default_rng(2)
    patients = np.arange(300)
    ref = pd.DataFrame(
        {m: rng.uniform(1, 2, 300) for m in ("mae", "dtw_mae", "template_mae", "beat_timing_ms")}
    )
    better = ref.assign(mae=ref.mae - 0.1)
    assert compare(better, ref, patients)["beats_reference"]
    worse_shape = better.assign(
        template_mae=ref.template_mae + 0.1
    )  # lower MAE but worse morphology
    result = compare(worse_shape, ref, patients)
    assert result["mae_better"] and not result["beats_reference"]
