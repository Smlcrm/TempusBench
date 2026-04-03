"""Per-model train→predict matrix (uni / multi / cov, shapes, cloud parity).

Requires ``RUN_MODEL_MATRIX_TEST=1``. Foundation models with ``hf_model_name`` need
``MODEL_WEIGHTS_PATH`` + synced snapshot (offline via autouse ``HF_HUB_OFFLINE=1``).

Examples::

  RUN_MODEL_MATRIX_TEST=1 MODEL_WEIGHTS_PATH=/path/to/weights \\
    pytest tests/integration/test_all_models_inference_matrix.py -m \"not slow\" -q

  MODEL_MATRIX_FAST=1  # smaller ctx/train/horizon

  MODEL_MATRIX_EXCLUDE=heavy_pkg,other  # comma-separated package ids

  MODEL_MATRIX_ONLY=chronos_tiny  # one model per pytest run (or comma-separated)

  ./tests/integration/run_model_matrix_one_by_one.sh  # loop all packages sequentially

Timestamps in matrix factories use **day-millisecond** spacing to avoid pandas OOB
with statsmodels on synthetic data. Full matrix (``not slow``) can still take many
minutes without ``MODEL_MATRIX_FAST=1``; foundation rows skip until weights exist.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from tests.integration.model_matrix_support import (
    CTX_STEPS_BATCH,
    FORECAST_HORIZON_BATCH,
    TRAIN_STEPS_BATCH,
    assert_hf_weight_snapshot_has_checkpoint,
    assert_internal_api_hints,
    assert_output_shape,
    covariate_mode,
    load_settings,
    make_covariate,
    make_multivariate,
    make_univariate,
    matrix_dimensions,
    model_type_from_settings,
    require_hf_weights_if_configured,
    run_train_predict_or_skip,
    skip_if_excluded,
    sorted_model_packages,
    supports_multivariate,
    supports_univariate,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_MATRIX_TEST", "") != "1",
        reason="Set RUN_MODEL_MATRIX_TEST=1 to run the full model matrix.",
    ),
    pytest.mark.model_matrix,
]


@pytest.fixture(autouse=True)
def _matrix_hf_offline(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "matrix_hf"
    for d in ("hf_home", "hf_hub", "transformers", "datasets", "tabpfn_cache"):
        (run / d).mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(run / "hf_home"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(run / "hf_hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(run / "transformers"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(run / "datasets"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(run / "tabpfn_cache"))


@pytest.fixture(autouse=True)
def _sundial_fast_diffusion(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if "sundial" not in request.node.nodeid.lower():
        return
    if os.environ.get("SUNDIAL_FAST_TEST", "") != "1":
        return
    import tempus_bench.models.sundial.sundial_model as sm

    orig_train = sm.SundialModel.train

    def train_then_trim_steps(self: Any, *args: Any, **kwargs: Any):
        result = orig_train(self, *args, **kwargs)
        m = getattr(self, "_model", None)
        if m is not None and hasattr(m, "config"):
            cfg = m.config
            if hasattr(cfg, "num_sampling_steps"):
                cfg.num_sampling_steps = 1
            if hasattr(cfg, "diffusion_batch_mul"):
                cfg.diffusion_batch_mul = 1
        return result

    monkeypatch.setattr(sm.SundialModel, "train", train_then_trim_steps)


@pytest.mark.parametrize("package", sorted_model_packages())
@pytest.mark.needs_weights
def test_matrix_univariate_train_predict_shapes(package: str) -> None:
    skip_if_excluded(package)
    settings = load_settings(package)
    if not supports_univariate(settings):
        pytest.skip(f"{package}: capabilities.univariate is false")
    require_hf_weights_if_configured(package, settings)
    assert_hf_weight_snapshot_has_checkpoint(package, settings)

    ctx, tr, hz = matrix_dimensions()
    rng = np.random.default_rng(abs(hash(package)) % (2**32))
    data = make_univariate(rng, ctx_steps=ctx, train_steps=tr, forecast_horizon=hz)
    out, nt, model, _settings = run_train_predict_or_skip(package, data)
    assert_internal_api_hints(model, package)
    mt = model_type_from_settings(_settings)
    if mt == "deterministic":
        assert_output_shape(
            out,
            model_type="deterministic",
            num_samples=1,
            horizon=hz,
            num_targets=nt,
            package=package,
        )
    else:
        assert_output_shape(
            out,
            model_type=mt,
            num_samples=4,
            horizon=hz,
            num_targets=nt,
            package=package,
        )


@pytest.mark.parametrize("package", sorted_model_packages())
@pytest.mark.needs_weights
def test_matrix_multivariate_train_predict_shapes(package: str) -> None:
    skip_if_excluded(package)
    settings = load_settings(package)
    if not supports_multivariate(settings):
        pytest.skip(f"{package}: capabilities.multivariate is false")
    require_hf_weights_if_configured(package, settings)

    ctx, tr, hz = matrix_dimensions()
    rng = np.random.default_rng(1 + abs(hash(package)) % (2**32))
    data = make_multivariate(rng, ctx_steps=ctx, train_steps=tr, forecast_horizon=hz)
    out, nt, model, _settings = run_train_predict_or_skip(package, data)
    assert_internal_api_hints(model, package)
    mt = model_type_from_settings(_settings)
    if mt == "deterministic":
        assert_output_shape(
            out,
            model_type="deterministic",
            num_samples=1,
            horizon=hz,
            num_targets=nt,
            package=package,
        )
    else:
        assert_output_shape(
            out,
            model_type=mt,
            num_samples=4,
            horizon=hz,
            num_targets=nt,
            package=package,
        )


@pytest.mark.parametrize("package", sorted_model_packages())
@pytest.mark.needs_weights
def test_matrix_covariate_train_predict_shapes(package: str) -> None:
    skip_if_excluded(package)
    settings = load_settings(package)
    mode = covariate_mode(settings)
    if mode == "none":
        pytest.skip(f"{package}: covariates none")
    if not supports_univariate(settings):
        pytest.skip(f"{package}: covariate smoke uses univariate target; univariate disabled")
    require_hf_weights_if_configured(package, settings)

    ctx, tr, hz = matrix_dimensions()
    rng = np.random.default_rng(2 + abs(hash(package)) % (2**32))
    past_only = mode == "past_only"
    data = make_covariate(
        rng,
        ctx_steps=ctx,
        train_steps=tr,
        forecast_horizon=hz,
        past_only=past_only,
    )
    out, nt, model, _settings = run_train_predict_or_skip(package, data)
    assert_internal_api_hints(model, package)
    mt = model_type_from_settings(_settings)
    if mt == "deterministic":
        assert_output_shape(
            out,
            model_type="deterministic",
            num_samples=1,
            horizon=hz,
            num_targets=nt,
            package=package,
        )
    else:
        assert_output_shape(
            out,
            model_type=mt,
            num_samples=4,
            horizon=hz,
            num_targets=nt,
            package=package,
        )


@pytest.mark.parametrize("package", sorted_model_packages())
@pytest.mark.needs_weights
def test_cloud_parity_128_55_55_horizon(package: str) -> None:
    """TT07: Same ctx/train/h as Batch-oriented failing-models suite."""
    skip_if_excluded(package)
    if os.environ.get("MODEL_MATRIX_FAST", "") == "1":
        pytest.skip("cloud parity uses full 128/55/55; unset MODEL_MATRIX_FAST")

    settings = load_settings(package)
    if not supports_univariate(settings):
        pytest.skip(f"{package}: univariate required for cloud parity case")
    require_hf_weights_if_configured(package, settings)

    rng = np.random.default_rng(99)
    data = make_univariate(
        rng,
        ctx_steps=CTX_STEPS_BATCH,
        train_steps=TRAIN_STEPS_BATCH,
        forecast_horizon=FORECAST_HORIZON_BATCH,
    )
    out, nt, _, _settings = run_train_predict_or_skip(package, data)
    mt = model_type_from_settings(_settings)
    h = FORECAST_HORIZON_BATCH
    if mt == "deterministic":
        assert out.shape[0] == h, f"{package}: cloud parity horizon {out.shape[0]} != {h}"
        assert out.shape[1] == nt
    else:
        assert out.shape[1] == h, f"{package}: cloud parity horizon {out.shape[1]} != {h}"
        assert out.shape[2] == nt
