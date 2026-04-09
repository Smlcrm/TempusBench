"""SVR must train/predict when context+train length is shorter than configured lookback."""

import numpy as np
import pytest

from tempus_bench.models.svr.svr_model import SvrModel


SVR_PARAMS = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}
SVR_SETTINGS = {
    "model_type": "deterministic",
    "python_version": "3.12",
    "lookback_window": 32,
}


def test_resolve_effective_lookback_caps_to_available_history():
    assert SvrModel._resolve_effective_lookback(
        series_length=23,
        forecast_horizon=5,
        configured_lookback=32,
    ) == int(23 - 5)

    assert SvrModel._resolve_effective_lookback(
        series_length=16,
        forecast_horizon=4,
        configured_lookback=32,
    ) == int(16 - 4)


def test_resolve_effective_lookback_uses_config_when_series_is_long():
    assert SvrModel._resolve_effective_lookback(
        series_length=500,
        forecast_horizon=4,
        configured_lookback=32,
    ) == 32


def test_resolve_effective_lookback_rejects_too_short_series():
    with pytest.raises(ValueError, match="Not enough data for SVR"):
        SvrModel._resolve_effective_lookback(
            series_length=5,
            forecast_horizon=5,
            configured_lookback=32,
        )


def test_train_predict_gdp_like_window():
    """Mimics gdp_*_covariate: context_window=18, forecast_horizon=5, lookback 32."""
    rng = np.random.default_rng(0)
    ctx = 18
    fh = 5
    n_variates = 1
    y_context = rng.standard_normal((ctx, n_variates))
    y_target = rng.standard_normal((fh, n_variates))
    ts_ctx = np.arange(ctx, dtype=np.float64)
    ts_tgt = np.arange(ctx, ctx + fh, dtype=np.float64)

    model = SvrModel(params=SVR_PARAMS, settings=SVR_SETTINGS)
    model.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )
    pred = model.predict(
        y_context=np.concatenate([y_context, y_target], axis=0),
        timestamps_context=np.arange(ctx + fh, dtype=np.float64),
        timestamps_target=np.arange(ctx + fh, ctx + fh + fh, dtype=np.float64),
    )
    assert pred.shape == (fh, n_variates)


def test_train_predict_patient_sparse_like_window():
    """Mimics patient_sparse_univariate: context_window=12, forecast_horizon=4."""
    rng = np.random.default_rng(1)
    ctx = 12
    fh = 4
    n_variates = 1
    y_context = rng.standard_normal((ctx, n_variates))
    y_target = rng.standard_normal((fh, n_variates))
    ts_ctx = np.arange(ctx, dtype=np.float64)
    ts_tgt = np.arange(ctx, ctx + fh, dtype=np.float64)

    model = SvrModel(params=SVR_PARAMS, settings=SVR_SETTINGS)
    model.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )
    pred = model.predict(
        y_context=np.concatenate([y_context, y_target], axis=0),
        timestamps_context=np.arange(ctx + fh, dtype=np.float64),
        timestamps_target=np.arange(ctx + fh, ctx + fh + fh, dtype=np.float64),
    )
    assert pred.shape == (fh, n_variates)
