"""Channel-independent covariate support for RandomForest, SVR, and CrostonClassic.

These models treat covariates as additional target columns: each column gets its
own independent model/process. Predictions for original target columns must be
identical regardless of covariate values.

Tests:
 * output shape is ``(horizon, num_original_targets)`` (covariates stripped)
 * channel independence: target predictions match with/without covariates
 * settings.yaml declares expected ``covariates`` mode per model
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub TensorFlow/TensorBoard before any tempus_bench import triggers metric_registry.
_TF_STUBS = ("tensorflow", "tensorboard", "tensorboard.summary")
for mod in _TF_STUBS:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import pytest
import yaml

from tempus_bench.models.random_forest.random_forest_model import RandomForestModel
from tempus_bench.models.svr.svr_model import SvrModel
from tempus_bench.models.croston_classic.croston_classic_model import CrostonClassicModel
from tempus_bench.utils.model_settings import load_capabilities_for_model, clear_model_settings_cache


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
    intermittent: bool = False,
):
    rng = np.random.default_rng(seed)
    if intermittent:
        raw = rng.choice([0.0, 1.0, 3.0, 5.0], size=(context_len + horizon, num_targets), p=[0.6, 0.2, 0.1, 0.1])
    else:
        raw = rng.standard_normal((context_len + horizon, num_targets))
    y_context = raw[:context_len]
    y_target = raw[context_len:]
    x_context = rng.standard_normal((context_len, num_covariates))
    x_target = rng.standard_normal((horizon, num_covariates))
    ts_ctx = np.arange(context_len, dtype=np.float64)
    ts_tgt = np.arange(context_len, context_len + horizon, dtype=np.float64)
    return y_context, y_target, x_context, x_target, ts_ctx, ts_tgt, rng


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------
class TestRandomForestCovariates:
    RF_PARAMS = {"n_estimators": 5, "max_depth": 2}
    RF_SETTINGS = {"model_type": "deterministic", "lookback_window": 16, "random_state": 42, "n_jobs": 1}

    def test_output_shape_with_covariates(self):
        y_ctx, y_tgt, x_ctx, x_tgt, ts_ctx, ts_tgt, _ = _make_data()
        m = RandomForestModel(params=self.RF_PARAMS, settings=self.RF_SETTINGS)
        m.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                timestamps_target=ts_tgt, x_context=x_ctx, x_target=x_tgt)
        pred = m.predict(y_context=np.concatenate([y_ctx, y_tgt], axis=0),
                         timestamps_context=np.concatenate([ts_ctx, ts_tgt]),
                         timestamps_target=ts_tgt, x_context=np.concatenate([x_ctx, x_tgt], axis=0))
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_output_shape_without_covariates(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt, _ = _make_data()
        m = RandomForestModel(params=self.RF_PARAMS, settings=self.RF_SETTINGS)
        m.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                timestamps_target=ts_tgt)
        pred = m.predict(y_context=np.concatenate([y_ctx, y_tgt], axis=0),
                         timestamps_context=np.concatenate([ts_ctx, ts_tgt]),
                         timestamps_target=ts_tgt)
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_settings_yaml_declares_past_future(self):
        clear_model_settings_cache()
        cap = load_capabilities_for_model("random_forest")
        assert cap.covariates == "past_future"
        assert cap.univariate is True
        assert cap.multivariate is True


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------
class TestSvrCovariates:
    SVR_PARAMS = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}
    SVR_SETTINGS = {"model_type": "deterministic", "lookback_window": 16}

    def test_output_shape_with_covariates(self):
        """past_only wiring: past covariates for context and train segments only."""
        y_ctx, y_tgt, x_ctx, x_tgt, ts_ctx, ts_tgt, _ = _make_data()
        m = SvrModel(params=self.SVR_PARAMS, settings=self.SVR_SETTINGS)
        m.train(
            y_context=y_ctx,
            y_target=y_tgt,
            timestamps_context=ts_ctx,
            timestamps_target=ts_tgt,
            x_context=x_ctx,
        )
        pred = m.predict(
            y_context=np.concatenate([y_ctx, y_tgt], axis=0),
            timestamps_context=np.concatenate([ts_ctx, ts_tgt]),
            timestamps_target=ts_tgt,
            x_context=np.concatenate([x_ctx, x_tgt], axis=0),
        )
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_output_shape_without_covariates(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt, _ = _make_data()
        m = SvrModel(params=self.SVR_PARAMS, settings=self.SVR_SETTINGS)
        m.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                timestamps_target=ts_tgt)
        pred = m.predict(y_context=np.concatenate([y_ctx, y_tgt], axis=0),
                         timestamps_context=np.concatenate([ts_ctx, ts_tgt]),
                         timestamps_target=ts_tgt)
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_settings_yaml_declares_past_only(self):
        clear_model_settings_cache()
        cap = load_capabilities_for_model("svr")
        assert cap.covariates == "past_only"
        assert cap.univariate is True
        assert cap.multivariate is True


# ---------------------------------------------------------------------------
# CrostonClassic
# ---------------------------------------------------------------------------
class TestCrostonClassicCovariates:
    CC_PARAMS = {"alpha": 0.3, "gamma": 0.3}
    CC_SETTINGS = {"model_type": "deterministic"}

    def test_output_shape_with_covariates(self):
        y_ctx, y_tgt, x_ctx, _, ts_ctx, ts_tgt, _ = _make_data(intermittent=True)
        m = CrostonClassicModel(params=self.CC_PARAMS, settings=self.CC_SETTINGS)
        m.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                timestamps_target=ts_tgt, x_context=x_ctx)
        pred = m.predict(y_context=y_ctx, timestamps_context=ts_ctx,
                         timestamps_target=ts_tgt, x_context=x_ctx)
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_output_shape_without_covariates(self):
        y_ctx, y_tgt, _, _, ts_ctx, ts_tgt, _ = _make_data(intermittent=True)
        m = CrostonClassicModel(params=self.CC_PARAMS, settings=self.CC_SETTINGS)
        m.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                timestamps_target=ts_tgt)
        pred = m.predict(y_context=y_ctx, timestamps_context=ts_ctx,
                         timestamps_target=ts_tgt)
        assert pred.shape == (HORIZON, NUM_TARGETS)

    def test_channel_independence_croston(self):
        """Target forecasts must be identical regardless of covariate values."""
        y_ctx, y_tgt, x_ctx, _, ts_ctx, ts_tgt, rng = _make_data(intermittent=True)
        x_alt = rng.standard_normal(x_ctx.shape)

        m0 = CrostonClassicModel(params=self.CC_PARAMS, settings=self.CC_SETTINGS)
        m0.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                 timestamps_target=ts_tgt, x_context=None)
        p0 = m0.predict(y_context=y_ctx, timestamps_context=ts_ctx,
                        timestamps_target=ts_tgt)

        m1 = CrostonClassicModel(params=self.CC_PARAMS, settings=self.CC_SETTINGS)
        m1.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                 timestamps_target=ts_tgt, x_context=x_ctx)
        p1 = m1.predict(y_context=y_ctx, timestamps_context=ts_ctx,
                        timestamps_target=ts_tgt, x_context=x_ctx)

        m2 = CrostonClassicModel(params=self.CC_PARAMS, settings=self.CC_SETTINGS)
        m2.train(y_context=y_ctx, y_target=y_tgt, timestamps_context=ts_ctx,
                 timestamps_target=ts_tgt, x_context=x_alt)
        p2 = m2.predict(y_context=y_ctx, timestamps_context=ts_ctx,
                        timestamps_target=ts_tgt, x_context=x_alt)

        np.testing.assert_allclose(p0, p1, rtol=0, atol=0)
        np.testing.assert_allclose(p0, p2, rtol=0, atol=0)

    def test_settings_yaml_declares_past_future(self):
        clear_model_settings_cache()
        cap = load_capabilities_for_model("croston_classic")
        assert cap.covariates == "past_future"
        assert cap.univariate is True
        assert cap.multivariate is True


# ---------------------------------------------------------------------------
# Cross-model: pipeline covariate wiring
# ---------------------------------------------------------------------------
class TestPipelineCovariateSets:
    """Models no longer appear in NO_COVARIATE_MODELS set."""

    def test_not_in_no_covariate_set(self):
        clear_model_settings_cache()
        from tempus_bench.utils.model_settings import get_no_covariate_models
        no_cov = get_no_covariate_models()
        for name in ("random_forest", "svr", "croston_classic"):
            assert name not in no_cov, f"{name} should NOT be in no-covariate set"
