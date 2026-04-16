"""Regression: joint Θ path must not assume len(y_context) >= 2."""

from __future__ import annotations

import numpy as np
import pytest

from tempus_bench.models.theta.theta_model import ThetaModel


def test_theta_hyperparams_survives_pydantic_pep563_regression() -> None:
    """Worker conda env uses Pydantic 2.12+; Literal-based hyperparams failed model_validate there."""
    for method in ("least_squares", "correlation_optimal"):
        m = ThetaModel(
            params={"sp": 4, "theta_method": method},
            settings={"model_type": "deterministic"},
        )
        assert str(m.theta_method) == method


@pytest.mark.parametrize("theta_method", ["least_squares", "correlation_optimal"])
def test_univariate_single_step_context_joint_path(theta_method: str) -> None:
    context_len = 1
    horizon = 3
    y_context = np.array([[2.5]])
    y_target = np.zeros((horizon, 1))
    ts_ctx = np.arange(context_len, dtype=np.float64)
    ts_tgt = np.arange(context_len, context_len + horizon, dtype=np.float64)
    m = ThetaModel(
        params={"sp": 1, "theta_method": theta_method},
        settings={"model_type": "deterministic"},
    )
    m.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )
    pred = m.predict(
        y_context=y_context,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )
    assert pred.shape == (horizon, 1)
    assert np.all(np.isfinite(pred))


def test_univariate_single_step_with_covariates_stacked_columns() -> None:
    context_len = 1
    horizon = 2
    rng = np.random.default_rng(0)
    y_context = np.array([[1.0]])
    y_target = np.zeros((horizon, 1))
    x_context = rng.standard_normal((context_len, 2))
    ts_ctx = np.arange(context_len, dtype=np.float64)
    ts_tgt = np.arange(context_len, context_len + horizon, dtype=np.float64)
    m = ThetaModel(
        params={"sp": 1, "theta_method": "least_squares"},
        settings={"model_type": "deterministic"},
    )
    m.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
        x_context=x_context,
    )
    pred = m.predict(
        y_context=y_context,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
        x_context=x_context,
    )
    assert pred.shape == (horizon, 1)
    assert np.all(np.isfinite(pred))


def test_joint_path_constant_univariate_uses_channel_fallback_not_crash() -> None:
    """Joint univariate path: strictly constant history cannot fit ThetaForecaster; stay finite."""
    context_len = 30
    horizon = 4
    y_context = np.ones((context_len, 1)) * 7.0
    y_target = np.ones((horizon, 1)) * 7.0
    ts_ctx = np.arange(context_len, dtype=np.float64)
    ts_tgt = np.arange(context_len, context_len + horizon, dtype=np.float64)
    m = ThetaModel(
        params={"sp": 4, "theta_method": "least_squares"},
        settings={"model_type": "deterministic"},
    )
    m.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )
    pred = m.predict(
        y_context=y_context,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )
    assert pred.shape == (horizon, 1)
    assert np.all(np.isfinite(pred))
    assert any(m is None for m in m._models)
