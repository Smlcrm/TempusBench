"""End-to-end smoke test for the Chronarium-backed LAFN benchmark adapter.

Loads a real LAFN checkpoint from GCS via Chronarium, runs ``train`` (no-op)
and ``predict`` for univariate / multivariate inputs, and checks that the
returned ``(num_samples, forecast_horizon, num_targets)`` array matches
the TempusBench stochastic-model contract: shape ``(S, H, T)`` with
finite values.

Set ``RUN_LAFN_GCS_TEST=1`` to enable; skipped by default because the
fixture downloads ~hundreds of MB from GCS and runs a real JAX forward
pass that takes ~30s on CPU.
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LAFN_GCS_TEST", "") != "1",
    reason="Set RUN_LAFN_GCS_TEST=1 to download the LAFN checkpoint from GCS.",
)


def _lafn_module():
    chronarium_spec = importlib.util.find_spec("chronarium")
    if chronarium_spec is None:
        pytest.skip("chronarium package not importable in current env")
    polars_spec = importlib.util.find_spec("polars")
    if polars_spec is None:
        pytest.skip("polars package not importable in current env")
    from tempus_bench.models.lafn.lafn_model import LafnModel  # type: ignore
    return LafnModel


def _lafn_settings() -> dict:
    settings_path = (
        Path(__file__).resolve().parents[2]
        / "tempus_bench"
        / "models"
        / "lafn"
        / "settings.yaml"
    )
    with open(settings_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def lafn_model():
    LafnModel = _lafn_module()
    settings = _lafn_settings()
    return LafnModel(params={}, settings=settings)


def _day_ms_timestamps(start: int, count: int) -> np.ndarray:
    day_ms = 86_400_000
    return (np.arange(start, start + count, dtype=np.int64) * day_ms).astype(np.int64)


def test_lafn_predict_univariate_shapes(lafn_model) -> None:
    rng = np.random.default_rng(0)
    ctx_steps, horizon, num_samples = 256, 16, 4
    y_context = rng.standard_normal((ctx_steps, 1)).astype(np.float64)
    timestamps_context = _day_ms_timestamps(0, ctx_steps)
    timestamps_target = _day_ms_timestamps(ctx_steps, horizon)

    trained = lafn_model.train(
        y_context=y_context,
        y_target=rng.standard_normal((horizon, 1)),
        timestamps_context=timestamps_context,
        timestamps_target=timestamps_target,
        num_samples=num_samples,
    )
    assert trained is lafn_model
    assert lafn_model.is_fitted

    samples = lafn_model.predict(
        y_context=y_context,
        timestamps_context=timestamps_context,
        timestamps_target=timestamps_target,
        num_samples=num_samples,
    )
    assert samples.shape == (num_samples, horizon, 1), f"got {samples.shape}"
    assert np.all(np.isfinite(samples)), "samples contain non-finite values"


def test_lafn_predict_multivariate_shapes(lafn_model) -> None:
    rng = np.random.default_rng(1)
    ctx_steps, horizon, num_targets, num_samples = 128, 12, 3, 8
    y_context = rng.standard_normal((ctx_steps, num_targets)).astype(np.float64)
    timestamps_context = _day_ms_timestamps(0, ctx_steps)
    timestamps_target = _day_ms_timestamps(ctx_steps, horizon)

    samples = lafn_model.predict(
        y_context=y_context,
        timestamps_context=timestamps_context,
        timestamps_target=timestamps_target,
        num_samples=num_samples,
    )
    assert samples.shape == (num_samples, horizon, num_targets)
    assert np.all(np.isfinite(samples))


def test_lafn_predict_rejects_covariates(lafn_model) -> None:
    rng = np.random.default_rng(2)
    ctx_steps, horizon = 64, 8
    y_context = rng.standard_normal((ctx_steps, 1)).astype(np.float64)
    timestamps_context = _day_ms_timestamps(0, ctx_steps)
    timestamps_target = _day_ms_timestamps(ctx_steps, horizon)
    x_context = rng.standard_normal((ctx_steps, 2)).astype(np.float64)
    x_target = rng.standard_normal((horizon, 2)).astype(np.float64)

    with pytest.raises(ValueError, match="LAFN"):
        lafn_model.predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            x_context=x_context,
            x_target=x_target,
            num_samples=4,
        )


def test_lafn_predict_matches_stochastic_metric_contract(lafn_model) -> None:
    """Predictions can be passed through MetricRegistry as model_type=stochastic."""

    from tempus_bench.pipeline.metric_registry import MetricRegistry

    rng = np.random.default_rng(3)
    ctx_steps, horizon, num_samples = 128, 10, 6
    y_context = rng.standard_normal((ctx_steps, 1)).astype(np.float64)
    y_true = rng.standard_normal((horizon, 1)).astype(np.float64)
    timestamps_context = _day_ms_timestamps(0, ctx_steps)
    timestamps_target = _day_ms_timestamps(ctx_steps, horizon)

    samples = lafn_model.predict(
        y_context=y_context,
        timestamps_context=timestamps_context,
        timestamps_target=timestamps_target,
        num_samples=num_samples,
    )

    registry = MetricRegistry()
    metrics = registry.compute_metrics(
        y_true=y_true,
        y_pred=samples,
        model_type="stochastic",
        point_forecast_statistic="mean",
        num_quantiles=3,
    )
    assert "mae" in metrics
    mae = metrics["mae"]
    if isinstance(mae, dict):
        flat = [v for v in mae.values() if isinstance(v, (int, float))]
        assert flat and all(np.isfinite(v) for v in flat)
    else:
        assert np.isfinite(float(mae))


def test_lafn_predict_validate_forecast_sanity(lafn_model) -> None:
    """The pipeline's ``validate_forecast_sanity`` consumes 3D LAFN samples directly."""

    from tempus_bench.utils.utils import validate_forecast_sanity

    rng = np.random.default_rng(4)
    ctx_steps, horizon, num_samples = 64, 8, 4
    y_context = rng.standard_normal((ctx_steps, 1)).astype(np.float64)
    y_true = rng.standard_normal((horizon, 1)).astype(np.float64)
    timestamps_context = _day_ms_timestamps(0, ctx_steps)
    timestamps_target = _day_ms_timestamps(ctx_steps, horizon)

    samples = lafn_model.predict(
        y_context=y_context,
        timestamps_context=timestamps_context,
        timestamps_target=timestamps_target,
        num_samples=num_samples,
    )

    validation = validate_forecast_sanity(
        y_true=y_true,
        y_pred=samples,
        point_forecast_statistic="mean",
    )
    assert isinstance(validation, dict) and "ok" in validation
