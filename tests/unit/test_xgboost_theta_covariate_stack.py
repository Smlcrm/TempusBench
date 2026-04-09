"""Channel-independent covariate stacking for XGBoost and Theta models.

Covariates (x_context / x_target) are stacked as additional columns in
y_context / y_target.  Predictions must be sliced back to the original
target count so the output shape is always (horizon, num_original_targets).

Tests:
 * output shape is correct with covariates
 * output shape is correct without covariates (no regression)
 * _num_original_targets is recorded during train
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

_TF_STUBS = ("tensorflow", "tensorboard", "tensorboard.summary")
for mod in _TF_STUBS:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import pytest

from tempus_bench.models.xgboost.xgboost_model import XgboostModel
from tempus_bench.models.theta.theta_model import ThetaModel


CONTEXT_LEN = 80
HORIZON = 4
NUM_TARGETS = 2
NUM_COVARIATES = 3
SEED = 42


def _make_data(
    *,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    num_targets: int = NUM_TARGETS,
    num_covariates: int = NUM_COVARIATES,
    seed: int = SEED,
):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((context_len + horizon, num_targets))
    y_context = raw[:context_len]
    y_target = raw[context_len:]
    x_context = rng.standard_normal((context_len, num_covariates))
    x_target = rng.standard_normal((horizon, num_covariates))
    ts_ctx = np.arange(context_len, dtype=np.float64)
    ts_tgt = np.arange(context_len, context_len + horizon, dtype=np.float64)
    return y_context, y_target, x_context, x_target, ts_ctx, ts_tgt


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
class TestXgboostCovariates:
    PARAMS = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3}
    SETTINGS = {"model_type": "deterministic", "lookback_window": 10}

    def test_output_shape_with_covariates(self):
        y_ctx, y_tgt, x_ctx, x_tgt, ts_ctx, ts_tgt = _make_data()
        m = XgboostModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, x_target=x_tgt, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, x_context=x_ctx, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_output_shape_without_covariates(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt = _make_data()
        m = XgboostModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_num_original_targets_recorded(self):
        y_ctx, y_tgt, x_ctx, x_tgt, ts_ctx, ts_tgt = _make_data()
        m = XgboostModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, x_target=x_tgt, freq="D",
        )
        assert m._num_original_targets == NUM_TARGETS

    def test_more_covariates_still_returns_targets_only(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt = _make_data(num_covariates=7)
        rng = np.random.default_rng(99)
        x_ctx = rng.standard_normal((CONTEXT_LEN, 7))
        x_tgt = rng.standard_normal((HORIZON, 7))
        m = XgboostModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, x_target=x_tgt, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, x_context=x_ctx, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------
class TestThetaCovariates:
    PARAMS = {"sp": 4, "theta_method": "least_squares"}
    SETTINGS: dict = {"model_type": "deterministic"}

    def test_output_shape_with_covariates(self):
        y_ctx, y_tgt, x_ctx, _, ts_ctx, ts_tgt = _make_data()
        m = ThetaModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, x_context=x_ctx, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_output_shape_without_covariates(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt = _make_data()
        m = ThetaModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_num_original_targets_recorded(self):
        y_ctx, y_tgt, x_ctx, _, ts_ctx, ts_tgt = _make_data()
        m = ThetaModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, freq="D",
        )
        assert m._num_original_targets == NUM_TARGETS

    def test_more_covariates_still_returns_targets_only(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt = _make_data(num_covariates=7)
        rng = np.random.default_rng(99)
        x_ctx = rng.standard_normal((CONTEXT_LEN, 7))
        m = ThetaModel(params=self.PARAMS, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, x_context=x_ctx, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_correlation_optimal_with_covariates(self):
        y_ctx, y_tgt, x_ctx, _, ts_ctx, ts_tgt = _make_data()
        params = {"sp": 4, "theta_method": "correlation_optimal"}
        m = ThetaModel(params=params, settings=self.SETTINGS)
        m.train(
            y_context=y_ctx, y_target=y_tgt,
            timestamps_context=ts_ctx, timestamps_target=ts_tgt,
            x_context=x_ctx, freq="D",
        )
        pred = m.predict(
            y_context=y_ctx, timestamps_context=ts_ctx,
            timestamps_target=ts_tgt, x_context=x_ctx, freq="D",
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)
