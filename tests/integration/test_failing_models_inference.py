"""Integration tests for models that fail in Google Batch: toto, tabpfn, sundial, moirai_moe.

Each model is tested against one representative task per task type (univariate,
multivariate, covariate) using the same context/horizon parameters that the
cloud benchmark pipeline sends (context_steps=128, train_steps=55, horizon=55).

Requirements:
  - ``RUN_LOCAL_WEIGHTS_TEST=1``
  - ``MODEL_WEIGHTS_PATH`` pointing at the GCS FUSE mirror (or local rsync)
    with subdirectories for each model's ``hf_model_name`` from settings.yaml.
  - ``HF_HUB_OFFLINE=1`` is enforced by the fixture to guarantee no downloads.

Run example:
  RUN_DIR=$(mktemp -d) && source deployment/run_local_weights_env.sh "$RUN_DIR" \\
    && export MODEL_WEIGHTS_PATH=/path/to/weights \\
    && RUN_LOCAL_WEIGHTS_TEST=1 pytest tests/integration/test_failing_models_inference.py -v

Sundial diffusion is very slow on CPU even with ``SUNDIAL_FAST_TEST=1`` (reduces
``num_sampling_steps`` / batch mul after ``train()``). For routine pre-deploy checks::

  pytest tests/integration/test_failing_models_inference.py -m "not slow" -v

Run Sundial integration explicitly (prefer GPU or a long timeout)::

  SUNDIAL_FAST_TEST=1 pytest tests/integration/test_failing_models_inference.py::TestSundial -v
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers (shared with test_local_weights_inference.py)
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    import tempus_bench.models as m
    return Path(m.__path__[0])


def _load_settings(package: str) -> dict[str, Any]:
    path = _models_dir() / package / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_candidate_ok(name: str) -> bool:
    if not name.endswith("Model") or name == "BaseModel":
        return False
    if name.endswith("BaseModel"):
        return False
    return True


def _find_model_class(package: str):
    mod = importlib.import_module(f"tempus_bench.models.{package}")
    for name in dir(mod):
        if not _model_candidate_ok(name):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == mod.__name__:
            return obj
    sub = importlib.import_module(f"tempus_bench.models.{package}.{package}_model")
    for name in dir(sub):
        if not _model_candidate_ok(name):
            continue
        obj = getattr(sub, name)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == sub.__name__:
            return obj
    raise ValueError(f"No *Model class in tempus_bench.models.{package}")


def _weights_root() -> str:
    return os.environ.get("MODEL_WEIGHTS_PATH", "").strip()


def _require_local_weights(hf_id: str) -> Path:
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")
    local = Path(root) / hf_id
    if not local.is_dir() or not any(local.iterdir()):
        pytest.skip(f"Missing non-empty weights dir for {hf_id!r}: {local}")
    return local


def _day_ns_timestamps(start: int, count: int) -> np.ndarray:
    """Nanosecond-precision timestamps (what the pipeline passes to models)."""
    day_ns = 86_400_000_000_000
    return (np.arange(start, start + count, dtype=np.int64) * day_ns).astype(np.int64)


def _day_ms_timestamps(start: int, count: int) -> np.ndarray:
    day_ms = 86_400_000
    return (np.arange(start, start + count, dtype=np.int64) * day_ms).astype(np.int64)


# ---------------------------------------------------------------------------
# Benchmark-realistic parameters (from Batch logs: context=128, horizon=55)
# ---------------------------------------------------------------------------

CTX_STEPS = 128
TRAIN_STEPS = 55
FORECAST_HORIZON = 55
NUM_SAMPLES = 20
NUM_VARIATES_MULTI = 3
NUM_COVARIATES = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_WEIGHTS_TEST", "") != "1",
    reason="Set RUN_LOCAL_WEIGHTS_TEST=1 to run bucket-local inference tests.",
)


@pytest.fixture(autouse=True)
def _isolated_hf_hub_caches(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force offline mode; any download attempt is a test failure."""
    run = tmp_path / "isolated_run"
    for d in ("hf_home", "hf_hub", "transformers", "datasets", "tabpfn_cache"):
        (run / d).mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(run / "hf_home"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(run / "hf_hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(run / "transformers"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(run / "datasets"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(run / "tabpfn_cache"))


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------

def _make_univariate(rng: np.random.Generator):
    """Univariate: 1 target, no covariates."""
    y_ctx = rng.standard_normal((CTX_STEPS, 1)).astype(np.float64)
    y_train = rng.standard_normal((TRAIN_STEPS, 1)).astype(np.float64)
    ts_ctx = _day_ns_timestamps(0, CTX_STEPS)
    ts_train = _day_ns_timestamps(CTX_STEPS, TRAIN_STEPS)
    ts_target = _day_ns_timestamps(CTX_STEPS + TRAIN_STEPS, FORECAST_HORIZON)
    return dict(
        y_ctx=y_ctx, y_train=y_train,
        ts_ctx=ts_ctx, ts_train=ts_train, ts_target=ts_target,
        x_ctx=None, x_train=None, x_target=None,
        num_targets=1, task_type="univariate",
    )


def _make_multivariate(rng: np.random.Generator):
    """Multivariate: multiple targets, no covariates."""
    y_ctx = rng.standard_normal((CTX_STEPS, NUM_VARIATES_MULTI)).astype(np.float64)
    y_train = rng.standard_normal((TRAIN_STEPS, NUM_VARIATES_MULTI)).astype(np.float64)
    ts_ctx = _day_ns_timestamps(0, CTX_STEPS)
    ts_train = _day_ns_timestamps(CTX_STEPS, TRAIN_STEPS)
    ts_target = _day_ns_timestamps(CTX_STEPS + TRAIN_STEPS, FORECAST_HORIZON)
    return dict(
        y_ctx=y_ctx, y_train=y_train,
        ts_ctx=ts_ctx, ts_train=ts_train, ts_target=ts_target,
        x_ctx=None, x_train=None, x_target=None,
        num_targets=NUM_VARIATES_MULTI, task_type="multivariate",
    )


def _make_covariate(rng: np.random.Generator, *, past_only: bool = False):
    """Covariate: 1 target + exogenous covariates (past or past+future)."""
    y_ctx = rng.standard_normal((CTX_STEPS, 1)).astype(np.float64)
    y_train = rng.standard_normal((TRAIN_STEPS, 1)).astype(np.float64)
    ts_ctx = _day_ns_timestamps(0, CTX_STEPS)
    ts_train = _day_ns_timestamps(CTX_STEPS, TRAIN_STEPS)
    ts_target = _day_ns_timestamps(CTX_STEPS + TRAIN_STEPS, FORECAST_HORIZON)
    x_ctx = rng.standard_normal((CTX_STEPS, NUM_COVARIATES)).astype(np.float64)
    x_train = rng.standard_normal((TRAIN_STEPS, NUM_COVARIATES)).astype(np.float64)
    x_target = None if past_only else rng.standard_normal((FORECAST_HORIZON, NUM_COVARIATES)).astype(np.float64)
    return dict(
        y_ctx=y_ctx, y_train=y_train,
        ts_ctx=ts_ctx, ts_train=ts_train, ts_target=ts_target,
        x_ctx=x_ctx, x_train=x_train, x_target=x_target,
        num_targets=1, task_type="covariate",
    )


# ---------------------------------------------------------------------------
# Generic train-then-predict driver
# ---------------------------------------------------------------------------

def _run_model(
    package: str,
    data: dict,
    *,
    num_samples: int = NUM_SAMPLES,
    extra_predict_kwargs: dict | None = None,
):
    """Instantiate model, train, predict, return output + metadata for assertions."""
    settings = _load_settings(package)
    hf_id = settings.get("hf_model_name", "")
    if hf_id:
        _require_local_weights(hf_id)

    Model = _find_model_class(package)
    model = Model({}, settings)

    train_kwargs: dict[str, Any] = dict(num_samples=num_samples)
    if data["x_ctx"] is not None:
        train_kwargs["x_context"] = data["x_ctx"]
    if data["x_train"] is not None:
        train_kwargs["x_target"] = data["x_train"]

    model.train(
        data["y_ctx"],
        data["y_train"],
        data["ts_ctx"],
        data["ts_train"],
        **train_kwargs,
    )
    assert model.is_fitted, f"{package} model is not fitted after train()"

    y_full = np.concatenate([data["y_ctx"], data["y_train"]], axis=0)
    ts_full = np.concatenate([data["ts_ctx"], data["ts_train"]], axis=0)

    predict_kwargs: dict[str, Any] = dict(num_samples=num_samples)
    if data["x_ctx"] is not None:
        x_full = np.concatenate([data["x_ctx"], data["x_train"]], axis=0)
        predict_kwargs["x_context"] = x_full
    if data["x_target"] is not None:
        predict_kwargs["x_target"] = data["x_target"]
    if extra_predict_kwargs:
        predict_kwargs.update(extra_predict_kwargs)

    out = model.predict(
        y_full,
        ts_full,
        data["ts_target"],
        **predict_kwargs,
    )
    return out, data["num_targets"], settings


def _assert_stochastic_shape(out: np.ndarray, num_targets: int, num_samples: int, horizon: int, model_name: str):
    """Stochastic model output: (num_samples, forecast_horizon, num_targets)."""
    assert out.ndim == 3, (
        f"{model_name}: expected 3D output (num_samples, horizon, targets), got shape {out.shape}"
    )
    assert out.shape[0] == num_samples, (
        f"{model_name}: expected {num_samples} samples, got {out.shape[0]}"
    )
    assert out.shape[1] == horizon, (
        f"{model_name}: expected forecast_horizon={horizon}, got {out.shape[1]}"
    )
    assert out.shape[2] == num_targets, (
        f"{model_name}: expected {num_targets} targets, got {out.shape[2]}"
    )
    assert np.all(np.isfinite(out)), f"{model_name}: output contains NaN/Inf"


def _assert_deterministic_shape(out: np.ndarray, num_targets: int, horizon: int, model_name: str):
    """Deterministic model output: (forecast_horizon, num_targets)."""
    assert out.ndim == 2, (
        f"{model_name}: expected 2D output (horizon, targets), got shape {out.shape}"
    )
    assert out.shape[0] == horizon, (
        f"{model_name}: expected forecast_horizon={horizon}, got {out.shape[0]}"
    )
    assert out.shape[1] == num_targets, (
        f"{model_name}: expected {num_targets} targets, got {out.shape[1]}"
    )
    assert np.all(np.isfinite(out)), f"{model_name}: output contains NaN/Inf"


# ===========================================================================
# TOTO tests
# ===========================================================================

class TestToto:
    """Toto (Datadog): stochastic foundation model.

    Known failure: circular import in vendored toto/ subpackage — ``from toto.data.util.dataset``
    resolves to the TempusBench wrapper package instead of the Datadog library.
    """
    PACKAGE = "toto"
    EXTRA = dict(freq="d")

    def test_import_succeeds(self):
        """Verify the toto model module can be imported without circular import errors."""
        _find_model_class(self.PACKAGE)

    def test_univariate(self):
        rng = np.random.default_rng(100)
        data = _make_univariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4, extra_predict_kwargs=self.EXTRA)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "toto")

    def test_multivariate(self):
        rng = np.random.default_rng(101)
        data = _make_multivariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4, extra_predict_kwargs=self.EXTRA)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "toto")

    def test_covariate(self):
        """Toto supports past+future covariates."""
        rng = np.random.default_rng(102)
        data = _make_covariate(rng, past_only=False)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4, extra_predict_kwargs=self.EXTRA)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "toto")


# ===========================================================================
# TabPFN tests
# ===========================================================================

class TestTabpfn:
    """TabPFN: deterministic zero-shot tabular model.

    Known failure: weights not mirrored to FUSE bucket; falls back to HF download
    which requires gated access to Prior-Labs/tabpfn_2_6.
    """
    PACKAGE = "tabpfn"

    def test_weights_exist_locally(self):
        """Verify TabPFN .ckpt file is present under MODEL_WEIGHTS_PATH."""
        settings = _load_settings(self.PACKAGE)
        hf_id = settings["hf_model_name"]
        local = _require_local_weights(hf_id)
        ckpts = list(local.glob("*.ckpt")) + list(local.glob("*.pt"))
        assert len(ckpts) > 0, (
            f"No .ckpt/.pt files under {local}; "
            "deploy TabPFN weights to the FUSE bucket"
        )

    def test_univariate(self):
        rng = np.random.default_rng(200)
        data = _make_univariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data)
        _assert_deterministic_shape(out, nt, FORECAST_HORIZON, "tabpfn")

    def test_multivariate(self):
        rng = np.random.default_rng(201)
        data = _make_multivariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data)
        _assert_deterministic_shape(out, nt, FORECAST_HORIZON, "tabpfn")

    def test_covariate_past_only(self):
        """TabPFN supports past covariates only."""
        rng = np.random.default_rng(202)
        data = _make_covariate(rng, past_only=True)
        out, nt, _ = _run_model(self.PACKAGE, data)
        _assert_deterministic_shape(out, nt, FORECAST_HORIZON, "tabpfn")


# ===========================================================================
# Sundial tests
# ===========================================================================

@pytest.mark.slow
class TestSundial:
    """Sundial (THU): diffusion-based stochastic model.

    Known failure: ``RuntimeError: shape '[-1, 12]' is invalid for input of size 11``
    in ``modeling_sundial.py`` — position_ids / attention_mask shape mismatch when
    ``lookback_length`` and ``forecast_horizon`` yield a non-patch-aligned token count.

    Marked ``slow``: exclude from default pre-deploy runs with ``-m "not slow"``.
    """
    PACKAGE = "sundial"

    @pytest.fixture(autouse=True)
    def _sundial_fast_diffusion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Optional CPU-friendly runs when ``SUNDIAL_FAST_TEST=1``."""
        if os.environ.get("SUNDIAL_FAST_TEST", "") != "1":
            return None
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
        return None

    def test_univariate(self):
        rng = np.random.default_rng(300)
        data = _make_univariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "sundial")

    def test_multivariate(self):
        """Sundial processes each variate independently; output shape must match."""
        rng = np.random.default_rng(301)
        data = _make_multivariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "sundial")

    def test_covariate_past_only(self):
        """Sundial supports past covariates only."""
        rng = np.random.default_rng(302)
        data = _make_covariate(rng, past_only=True)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "sundial")

    def test_patch_alignment_matches_attention_mask(self):
        """Reproduce the exact failure: lookback=256, horizon=55, patch_size=16.

        Sundial tokenizes in fixed-size patches of 16. The attention_mask in
        generate() must have token_len that matches what the model's forward()
        computes for ``seq_length`` from the actual patched input. A mismatch
        causes ``position_ids.view(-1, seq_length)`` to fail.
        """
        settings = _load_settings(self.PACKAGE)
        lookback = int(settings.get("lookback_length", 256))
        patch_size = 16
        token_len = max(1, int((lookback + patch_size - 1) // patch_size))

        rng = np.random.default_rng(303)
        y_ctx = rng.standard_normal((lookback, 1)).astype(np.float64)
        y_train = rng.standard_normal((TRAIN_STEPS, 1)).astype(np.float64)
        ts_ctx = _day_ns_timestamps(0, lookback)
        ts_train = _day_ns_timestamps(lookback, TRAIN_STEPS)
        ts_target = _day_ns_timestamps(lookback + TRAIN_STEPS, FORECAST_HORIZON)

        hf_id = settings.get("hf_model_name", "")
        if hf_id:
            _require_local_weights(hf_id)

        Model = _find_model_class(self.PACKAGE)
        model = Model({}, settings)
        model.train(y_ctx, y_train, ts_ctx, ts_train, num_samples=4)

        y_full = np.concatenate([y_ctx, y_train], axis=0)
        ts_full = np.concatenate([ts_ctx, ts_train], axis=0)
        out = model.predict(y_full, ts_full, ts_target, num_samples=4)

        assert out.ndim == 3, f"sundial: expected 3D output, got shape {out.shape}"
        assert out.shape[1] == FORECAST_HORIZON, (
            f"sundial: attention mask token_len={token_len} produced wrong horizon; "
            f"expected {FORECAST_HORIZON}, got {out.shape[1]}"
        )


# ===========================================================================
# Moirai-MoE tests
# ===========================================================================

class TestMoiraiMoe:
    """Moirai-MoE (Salesforce): stochastic mixture-of-experts model.

    Known failure: ``ValueError: moirai_moe: forecast length 16 shorter than
    required horizon 55`` — MoiraiMoEForecast's internal prediction_length caps
    at patch_size, so longer horizons get truncated.
    """
    PACKAGE = "moirai_moe"

    def test_univariate(self):
        rng = np.random.default_rng(400)
        data = _make_univariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "moirai_moe")

    def test_multivariate(self):
        rng = np.random.default_rng(401)
        data = _make_multivariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "moirai_moe")

    def test_covariate_past_future(self):
        """Moirai-MoE supports both past and future covariates."""
        rng = np.random.default_rng(402)
        data = _make_covariate(rng, past_only=False)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "moirai_moe")

    def test_horizon_55_not_truncated(self):
        """Exact reproduction of the Batch failure: horizon=55 but model returns 16.

        The MoiraiMoEForecast is initialized with ``prediction_length=pdt`` in
        train(), where ``pdt = max(train_span, val_h)``. For benchmark tasks this
        is 55. But the actual uni2ts forward pass returns fewer steps (16) when
        ``patch_size = pdt + ctx`` is too large for the internal patching scheme.
        """
        settings = _load_settings(self.PACKAGE)
        hf_id = settings.get("hf_model_name", "")
        if hf_id:
            _require_local_weights(hf_id)

        Model = _find_model_class(self.PACKAGE)
        model = Model({}, settings)

        rng = np.random.default_rng(403)
        y_ctx = rng.standard_normal((CTX_STEPS, 1)).astype(np.float64)
        y_train = rng.standard_normal((TRAIN_STEPS, 1)).astype(np.float64)
        ts_ctx = _day_ns_timestamps(0, CTX_STEPS)
        ts_train = _day_ns_timestamps(CTX_STEPS, TRAIN_STEPS)
        ts_target = _day_ns_timestamps(CTX_STEPS + TRAIN_STEPS, FORECAST_HORIZON)

        model.train(
            y_ctx, y_train, ts_ctx, ts_train,
            num_samples=4, validate_horizon=FORECAST_HORIZON,
        )

        y_full = np.concatenate([y_ctx, y_train], axis=0)
        ts_full = np.concatenate([ts_ctx, ts_train], axis=0)

        out = model.predict(y_full, ts_full, ts_target, num_samples=4)

        assert out.ndim == 3, f"moirai_moe: expected 3D, got {out.shape}"
        assert out.shape[1] == FORECAST_HORIZON, (
            f"moirai_moe: forecast length {out.shape[1]} != required horizon {FORECAST_HORIZON}; "
            "model needs multi-step / iterative forecasting for horizons > internal patch limit"
        )


# ===========================================================================
# Moirai 1.1 Large (MoiraiBaseModel — same predict context alignment as Moirai-MoE)
# ===========================================================================

class TestMoiraiLarge:
    """Moirai 1.1 Large: inherits MoiraiBaseModel (uni2ts MoiraiForecast, not MoE)."""

    PACKAGE = "moirai_large"

    def test_multivariate(self):
        rng = np.random.default_rng(410)
        data = _make_multivariate(rng)
        out, nt, _ = _run_model(self.PACKAGE, data, num_samples=4)
        _assert_stochastic_shape(out, nt, 4, FORECAST_HORIZON, "moirai_large")

    def test_full_history_predict_matches_context_length(self):
        """Batch passes y of len context+train; predict must slice to train's context_length."""
        settings = _load_settings(self.PACKAGE)
        hf_id = settings.get("hf_model_name", "")
        if hf_id:
            _require_local_weights(hf_id)

        Model = _find_model_class(self.PACKAGE)
        model = Model({}, settings)

        rng = np.random.default_rng(411)
        y_ctx = rng.standard_normal((CTX_STEPS, NUM_VARIATES_MULTI)).astype(np.float64)
        y_train = rng.standard_normal((TRAIN_STEPS, NUM_VARIATES_MULTI)).astype(np.float64)
        ts_ctx = _day_ns_timestamps(0, CTX_STEPS)
        ts_train = _day_ns_timestamps(CTX_STEPS, TRAIN_STEPS)
        ts_target = _day_ns_timestamps(CTX_STEPS + TRAIN_STEPS, FORECAST_HORIZON)

        model.train(
            y_ctx, y_train, ts_ctx, ts_train,
            num_samples=4,
        )

        y_full = np.concatenate([y_ctx, y_train], axis=0)
        ts_full = np.concatenate([ts_ctx, ts_train], axis=0)
        out = model.predict(y_full, ts_full, ts_target, num_samples=4)

        assert out.ndim == 3, f"moirai_large: expected 3D, got {out.shape}"
        assert out.shape[1] == FORECAST_HORIZON, (
            f"moirai_large: forecast length {out.shape[1]} != horizon {FORECAST_HORIZON}"
        )
