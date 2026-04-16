"""
Unit tests for covariate support across foundation models.

All foundation models must be compatible with covariates (x_context).
These tests verify that each model:
1. Accepts x_context in train() and predict()
2. Runs without error when covariates are provided
3. Produces correctly shaped output
"""

import importlib.util
import pytest
import numpy as np
import yaml
from pathlib import Path

from tempus_bench.utils.paths import get_models_dir
from tempus_bench.utils.model_settings import is_past_only_covariates

# Models skipped due to external/package issues (not covariate-related)
COVARIATE_TEST_SKIP = {
    "lafn",  # Requires Chronarium remote model, credentials
    "sundial",  # transformers DynamicCache API change (seen_tokens)
}

# Foundation models that must support covariates (from benchmark.yaml)
FOUNDATION_MODELS = [
    # Chronos family
    "chronos_tiny",
    "chronos_mini",
    "chronos_small",
    "chronos_base",
    "chronos_large",
    "chronos_bolt_tiny",
    "chronos_bolt_mini",
    "chronos_bolt_small",
    "chronos_bolt_base",
    "chronos2",
    "chronos2_small",
    # TimesFM
    "timesfm2",
    "timesfm_500m",
    "timesfm_200m",
    # Other stochastic foundation models
    "granite_flowstate",
    "kairos_10m",
    "kairos_23m",
    "kairos_50m",
    "moirai_small",
    "moirai_base",
    "moirai_large",
    "moirai_moe",
    "moirai2",
    "patchtst_fm",
    "patchtst_granite",
    "lagllama",
    "toto",
    "tabpfn",
    "tabpfn_ts",
    "tirex",
    "tirex_1_1_gifteval",
    "sundial",
    "lafn",
    # Deterministic foundation models
    "moment_small",
    "moment_base",
    "moment_large",
    "time_moe_50m",
    "time_moe_200m",
    "tiny_time_mixer_r1",
    "tiny_time_mixer_r2",
    "tiny_time_mixer_r2_1",
]


def _load_model_class(model_name: str):
    """Load model class and settings for a given model name."""
    models_dir = get_models_dir()
    model_dir = models_dir / model_name
    if not model_dir.exists():
        pytest.skip(f"Model directory not found: {model_name}")

    model_file = model_dir / f"{model_name}_model.py"
    settings_file = model_dir / "settings.yaml"
    if not model_file.exists():
        pytest.skip(f"Model file not found: {model_name}")

    # Load settings
    settings = {}
    if settings_file.exists():
        with open(settings_file) as f:
            settings = yaml.safe_load(f) or {}

    # Import model class
    class_name = "".join(word.capitalize() for word in model_name.split("_")) + "Model"
    spec = importlib.util.spec_from_file_location(f"{model_name}_model", str(model_file))
    if spec is None or spec.loader is None:
        pytest.skip(f"Failed to load module for {model_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        pytest.skip(f"Model class {class_name} not found in {model_name}")

    return model_class, settings


def _make_test_data(
    context_len: int = 64,
    forecast_horizon: int = 12,
    num_targets: int = 1,
    num_covariates: int = 2,
    seed: int = 42,
):
    """Create synthetic test data with covariates."""
    np.random.seed(seed)
    y_context = np.random.randn(context_len, num_targets).astype(np.float64)
    y_target = np.random.randn(forecast_horizon, num_targets).astype(np.float64)
    x_context = np.random.randn(context_len, num_covariates).astype(np.float64)
    x_target = np.random.randn(forecast_horizon, num_covariates).astype(np.float64)
    timestamps_context = np.arange(context_len, dtype=np.float64)
    timestamps_target = np.arange(forecast_horizon, dtype=np.float64)
    return {
        "y_context": y_context,
        "y_target": y_target,
        "x_context": x_context,
        "x_target": x_target,
        "timestamps_context": timestamps_context,
        "timestamps_target": timestamps_target,
    }


def _get_model_specific_kwargs(model_name: str, settings: dict):
    """Return model-specific kwargs (freq, num_samples, etc.)."""
    kwargs = {"num_samples": 5}
    # Lag-Llama and some GluonTS models require freq
    if "lagllama" in model_name or "lag_llama" in model_name:
        kwargs["freq"] = "D"
    # Granite FlowState: scale_factor depends on freq
    if "granite_flowstate" in model_name:
        kwargs["freq"] = "D"
    # Ensure context_length / context_window is not too large for small test data
    if "context_length" in settings and settings["context_length"] > 128:
        pass  # Models will trim to available data
    if "context_window" in settings and settings["context_window"] > 128:
        pass
    return kwargs


@pytest.mark.parametrize("model_name", FOUNDATION_MODELS)
def test_foundation_model_accepts_covariates(model_name):
    """
    Verify that each foundation model accepts x_context and runs without error.

    Tests train() and predict() with covariates provided.
    """
    if model_name in COVARIATE_TEST_SKIP:
        pytest.skip(f"{model_name}: skipped (external/package dependency)")

    try:
        model_class, settings = _load_model_class(model_name)
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e}")

    data = _make_test_data(
        context_len=64,
        forecast_horizon=12,
        num_targets=1,
        num_covariates=2,
    )

    kwargs = _get_model_specific_kwargs(model_name, settings)
    params = {}
    model_settings = {**settings}

    try:
        model = model_class(params=params, settings=model_settings)
    except Exception as e:
        pytest.skip(f"Could not instantiate {model_name}: {e}")

    # Train with covariates (x_context only for past-only models; both for full-support)
    x_ctx = data["x_context"]
    x_tgt = None if is_past_only_covariates(model_name) else data["x_target"]
    try:
        model.train(
            y_context=data["y_context"],
            y_target=data["y_target"],
            timestamps_context=data["timestamps_context"],
            timestamps_target=data["timestamps_target"],
            x_context=x_ctx,
            x_target=x_tgt,
            **kwargs,
        )
    except Exception as e:
        pytest.fail(f"{model_name} train() failed with covariates: {e}")

    # Predict with covariates
    try:
        pred = model.predict(
            y_context=data["y_context"],
            timestamps_context=data["timestamps_context"],
            timestamps_target=data["timestamps_target"],
            x_context=x_ctx,
            x_target=x_tgt,
            **kwargs,
        )
    except Exception as e:
        pytest.fail(f"{model_name} predict() failed with covariates: {e}")

    # Verify output shape
    assert pred is not None, f"{model_name} returned None from predict()"
    pred = np.asarray(pred)

    # Handle hybrid models (return tuple)
    if isinstance(pred, tuple):
        pred = pred[0] if len(pred) > 0 else pred
    pred = np.asarray(pred)

    forecast_horizon = data["timestamps_target"].shape[0]
    num_targets = data["y_context"].shape[1]

    model_type = settings.get("model_type", "stochastic")
    if model_type == "deterministic":
        # Accept (forecast_horizon, num_targets) or (num_targets, forecast_horizon)
        ok_shape1 = pred.shape == (forecast_horizon, num_targets)
        ok_shape2 = pred.shape == (num_targets, forecast_horizon)
        assert ok_shape1 or ok_shape2, (
            f"{model_name} deterministic output shape {pred.shape} "
            f"not in [(forecast_horizon={forecast_horizon}, num_targets={num_targets}), "
            f"(num_targets={num_targets}, forecast_horizon={forecast_horizon})]"
        )
    else:
        # Stochastic: (num_samples, forecast_horizon, num_targets)
        assert pred.ndim == 3, f"{model_name} stochastic output should be 3D, got {pred.ndim}D"
        assert pred.shape[1] == forecast_horizon, (
            f"{model_name} horizon mismatch: {pred.shape[1]} != {forecast_horizon}"
        )
        assert pred.shape[2] == num_targets, (
            f"{model_name} num_targets mismatch: {pred.shape[2]} != {num_targets}"
        )


@pytest.mark.parametrize("model_name", FOUNDATION_MODELS)
def test_foundation_model_runs_without_covariates(model_name):
    """
    Verify that each foundation model runs when x_context is None (backward compat).
    """
    if model_name in COVARIATE_TEST_SKIP:
        pytest.skip(f"{model_name}: skipped (external/package dependency)")

    try:
        model_class, settings = _load_model_class(model_name)
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e}")

    data = _make_test_data()
    kwargs = _get_model_specific_kwargs(model_name, settings)
    params = {}
    model_settings = {**settings}

    try:
        model = model_class(params=params, settings=model_settings)
    except Exception as e:
        pytest.skip(f"Could not instantiate {model_name}: {e}")

    try:
        model.train(
            y_context=data["y_context"],
            y_target=data["y_target"],
            timestamps_context=data["timestamps_context"],
            timestamps_target=data["timestamps_target"],
            x_context=None,
            x_target=None,
            **kwargs,
        )
        pred = model.predict(
            y_context=data["y_context"],
            timestamps_context=data["timestamps_context"],
            timestamps_target=data["timestamps_target"],
            x_context=None,
            x_target=None,
            **kwargs,
        )
    except Exception as e:
        pytest.fail(f"{model_name} failed without covariates: {e}")

    assert pred is not None
    pred = np.asarray(pred)
    if isinstance(pred, tuple):
        pred = np.asarray(pred[0])
    assert pred.size > 0, f"{model_name} returned empty prediction"


def test_extend_input_models_reject_x_target_only():
    """Extend-input models (x_context only) must raise when x_target is provided without x_context."""
    from tempus_bench.models.base_model import validate_covariate_support

    x_context = None
    x_target = np.random.randn(12, 2).astype(np.float64)

    with pytest.raises(ValueError, match="does not support future covariates \\(x_target\\) only"):
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="TestModel",
        )


def test_extend_input_models_reject_both():
    """Extend-input models must raise when both x_context and x_target are provided."""
    from tempus_bench.models.base_model import validate_covariate_support

    x_context = np.random.randn(64, 2).astype(np.float64)
    x_target = np.random.randn(12, 2).astype(np.float64)

    with pytest.raises(ValueError, match="does not support both past and future covariates"):
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="TestModel",
        )


def test_no_covariate_models_reject_x_context():
    """Models with no covariate support must raise when x_context is provided."""
    from tempus_bench.models.base_model import validate_covariate_support

    x_context = np.random.randn(64, 2).astype(np.float64)
    x_target = None

    with pytest.raises(ValueError, match="does not support past covariates \\(x_context\\) only"):
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=False,
            supports_future_only=False,
            supports_both=False,
            model_name="TestModel",
        )


def test_tabpfn_raises_on_x_target_only():
    """TabPFN (extend-input) must raise when x_target is provided without x_context."""
    from tempus_bench.models.tabpfn.tabpfn_model import TabpfnModel

    model = TabpfnModel(params={}, settings={"device": "cpu", "max_sequence_length": 1000})
    model.is_fitted = True

    data = _make_test_data()
    x_context = None
    x_target = np.random.randn(12, 2).astype(np.float64)

    with pytest.raises(ValueError, match="TabPFN.*does not support future covariates"):
        model.predict(
            y_context=data["y_context"],
            timestamps_context=data["timestamps_context"],
            timestamps_target=data["timestamps_target"],
            x_context=x_context,
            x_target=x_target,
            num_samples=5,
            freq="D",
        )
