"""Exponential smoothing: channel-independent covariates (extra univariate fits)."""

import numpy as np
import pytest

from tempus_bench.models.exponential_smoothing.exponential_smoothing_model import (
    ExponentialSmoothingModel,
)


def _simple_params() -> dict:
    return {
        "trend": "null",
        "seasonal": "null",
        "seasonal_periods": "null",
        "damped_trend": False,
    }


class TestExponentialSmoothingCovariates:
    def test_target_forecast_unchanged_when_covariates_present(self) -> None:
        """Fitting extra channels on x_context must not change target-column ETS."""
        rng = np.random.default_rng(0)
        n_ctx, n_h = 48, 8
        y_ctx = np.cumsum(rng.standard_normal((n_ctx, 1)), axis=0)
        y_tgt = rng.standard_normal((n_h, 1))
        x_ctx = rng.standard_normal((n_ctx, 2))
        x_tgt = rng.standard_normal((n_h, 2))
        ts_c = np.arange(n_ctx, dtype=np.float64)
        ts_t = np.arange(n_ctx, n_ctx + n_h, dtype=np.float64)

        m0 = ExponentialSmoothingModel(_simple_params(), {})
        m0.train(y_ctx, y_tgt, ts_c, ts_t, x_context=None, x_target=None)
        p0 = m0.predict(y_ctx, ts_c, ts_t, x_context=None, x_target=None)

        m1 = ExponentialSmoothingModel(_simple_params(), {})
        m1.train(y_ctx, y_tgt, ts_c, ts_t, x_context=x_ctx, x_target=x_tgt)
        p1 = m1.predict(y_ctx, ts_c, ts_t, x_context=x_ctx, x_target=x_tgt)

        assert p0.shape == (n_h, 1)
        assert p1.shape == (n_h, 1)
        np.testing.assert_allclose(p0, p1, rtol=1e-9, atol=1e-9)

    def test_multivariate_targets_plus_covariates_output_shape(self) -> None:
        n_ctx, n_h = 36, 6
        y_ctx = np.random.default_rng(1).standard_normal((n_ctx, 3))
        y_tgt = np.random.default_rng(2).standard_normal((n_h, 3))
        x_ctx = np.random.default_rng(3).standard_normal((n_ctx, 2))
        x_tgt = np.random.default_rng(4).standard_normal((n_h, 2))
        ts_c = np.arange(n_ctx, dtype=np.float64)
        ts_t = np.arange(n_ctx, n_ctx + n_h, dtype=np.float64)

        m = ExponentialSmoothingModel(_simple_params(), {})
        m.train(y_ctx, y_tgt, ts_c, ts_t, x_context=x_ctx, x_target=x_tgt)
        pred = m.predict(y_ctx, ts_c, ts_t, x_context=x_ctx, x_target=x_tgt)

        assert pred.shape == (n_h, 3)

    def test_future_covariates_without_past_rejected(self) -> None:
        n_ctx, n_h = 24, 4
        y_ctx = np.random.default_rng(5).standard_normal((n_ctx, 1))
        y_tgt = np.random.default_rng(6).standard_normal((n_h, 1))
        x_tgt = np.random.default_rng(7).standard_normal((n_h, 1))
        ts_c = np.arange(n_ctx, dtype=np.float64)
        ts_t = np.arange(n_ctx, n_ctx + n_h, dtype=np.float64)

        m = ExponentialSmoothingModel(_simple_params(), {})
        with pytest.raises(ValueError, match="future covariates"):
            m.train(y_ctx, y_tgt, ts_c, ts_t, x_context=None, x_target=x_tgt)

    def test_past_only_covariates_allowed(self) -> None:
        n_ctx, n_h = 30, 5
        y_ctx = np.random.default_rng(8).standard_normal((n_ctx, 1))
        y_tgt = np.random.default_rng(9).standard_normal((n_h, 1))
        x_ctx = np.random.default_rng(10).standard_normal((n_ctx, 2))
        ts_c = np.arange(n_ctx, dtype=np.float64)
        ts_t = np.arange(n_ctx, n_ctx + n_h, dtype=np.float64)

        m = ExponentialSmoothingModel(_simple_params(), {})
        m.train(y_ctx, y_tgt, ts_c, ts_t, x_context=x_ctx, x_target=None)
        pred = m.predict(y_ctx, ts_c, ts_t, x_context=x_ctx, x_target=None)
        assert pred.shape == (n_h, 1)
