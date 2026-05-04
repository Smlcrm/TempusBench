"""Seasonal naive: past covariates as extra channels (channel-independent); targets-only output."""

from __future__ import annotations

import numpy as np
import pytest

from tempus_bench.models.seasonal_naive.seasonal_naive_model import SeasonalNaiveModel


def _ts_context_target(
    *,
    context_len: int = 24,
    horizon: int = 4,
    num_targets: int = 2,
    num_covariates: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_context = rng.standard_normal((context_len, num_targets))
    y_target = rng.standard_normal((horizon, num_targets))
    x_context = rng.standard_normal((context_len, num_covariates))
    ts_ctx = np.arange(context_len, dtype=np.float64)
    ts_tgt = np.arange(horizon, dtype=np.float64)
    return y_context, y_target, x_context, ts_ctx, ts_tgt, rng


def test_seasonal_naive_without_covariates_matches_baseline() -> None:
    y_context, y_target, _, ts_ctx, ts_tgt, _ = _ts_context_target()
    model = SeasonalNaiveModel(params={"sp": 4}, settings={})
    model.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
        x_context=None,
        x_target=None,
    )
    pred = model.predict(
        y_context=y_context,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
        x_context=None,
        x_target=None,
    )
    assert pred.shape == (ts_tgt.shape[0], y_context.shape[1])


def test_seasonal_naive_covariates_past_only_channel_independent_targets() -> None:
    """Target forecasts do not depend on covariate values (extra channels only)."""
    y_context, y_target, x_context, ts_ctx, ts_tgt, rng = _ts_context_target()
    x_alt = rng.standard_normal(x_context.shape)

    common = dict(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
    )

    m0 = SeasonalNaiveModel(params={"sp": 4}, settings={})
    m0.train(**common, x_context=None, x_target=None)
    p0 = m0.predict(**common, x_context=None, x_target=None)

    m1 = SeasonalNaiveModel(params={"sp": 4}, settings={})
    m1.train(**common, x_context=x_context, x_target=None)
    p1 = m1.predict(**common, x_context=x_context, x_target=None)

    m2 = SeasonalNaiveModel(params={"sp": 4}, settings={})
    m2.train(**common, x_context=x_alt, x_target=None)
    p2 = m2.predict(**common, x_context=x_alt, x_target=None)

    np.testing.assert_allclose(p0, p1, rtol=0, atol=0)
    np.testing.assert_allclose(p0, p2, rtol=0, atol=0)


def test_seasonal_naive_rejects_future_covariates_without_past() -> None:
    y_context, y_target, _, ts_ctx, ts_tgt, _ = _ts_context_target()
    model = SeasonalNaiveModel(params={"sp": 4}, settings={})
    x_target_only = np.zeros((ts_tgt.shape[0], 2))
    with pytest.raises(ValueError, match="does not support future covariates"):
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_ctx,
            timestamps_target=ts_tgt,
            x_context=None,
            x_target=x_target_only,
        )


def test_seasonal_naive_rejects_past_and_future_covariates_together() -> None:
    y_context, y_target, x_context, ts_ctx, ts_tgt, _ = _ts_context_target()
    model = SeasonalNaiveModel(params={"sp": 4}, settings={})
    x_target = np.zeros((y_target.shape[0], x_context.shape[1]))
    with pytest.raises(ValueError, match="does not support both past and future covariates"):
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_ctx,
            timestamps_target=ts_tgt,
            x_context=x_context,
            x_target=x_target,
        )


def test_seasonal_naive_predict_channel_mismatch_raises() -> None:
    y_context, y_target, x_context, ts_ctx, ts_tgt, _ = _ts_context_target(num_covariates=1)
    model = SeasonalNaiveModel(params={"sp": 4}, settings={})
    model.train(
        y_context=y_context,
        y_target=y_target,
        timestamps_context=ts_ctx,
        timestamps_target=ts_tgt,
        x_context=x_context,
        x_target=None,
    )
    x_wrong = np.zeros((y_context.shape[0], 3))
    with pytest.raises(ValueError, match="Channel count mismatch"):
        model.predict(
            y_context=y_context,
            timestamps_context=ts_ctx,
            timestamps_target=ts_tgt,
            x_context=x_wrong,
            x_target=None,
        )
