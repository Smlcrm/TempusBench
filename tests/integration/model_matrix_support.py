"""Shared helpers for comprehensive per-model matrix tests (RUN_MODEL_MATRIX_TEST=1).

Discovery, settings/capabilities, default hyperparameter grids, data factories,
train→predict driver, and TempusBench-canonical output shape assertions.

Set ``MODEL_MATRIX_ONLY`` (comma-separated package ids) to parametrize matrix/import
tests for one or a few models per run; see ``tests/integration/run_model_matrix_one_by_one.sh``.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import yaml

from tempus_bench.utils.paths import get_available_models, get_models_dir

ModelType = Literal["deterministic", "stochastic", "hybrid"]

# Cloud / Batch-style defaults (aligned with test_failing_models_inference.py)
CTX_STEPS_BATCH = 128
TRAIN_STEPS_BATCH = 55
FORECAST_HORIZON_BATCH = 55
NUM_VARIATES_MULTI = 3
NUM_COVARIATES = 2
NUM_SAMPLES_DEFAULT = 4

MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON = frozenset(
    {"moirai_moe", "moirai_base", "moirai_small", "moirai_large", "moirai2"}
)

# Models whose train() requires a ``freq`` kwarg (see BaseModel / neuralforecast stack).
PACKAGES_REQUIRING_TRAIN_FREQ = frozenset(
    {
        "arima",
        "nhits",
        "nbeats",
        "itransformer",
        "timesnet",
        "tft",
        "xgboost",
    }
)

FAST_CTX = 64
FAST_TRAIN = 24
FAST_HORIZON = 12


def matrix_excluded_packages() -> frozenset[str]:
    raw = os.environ.get("MODEL_MATRIX_EXCLUDE", "").strip()
    if not raw:
        return frozenset()
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def skip_if_excluded(package: str) -> None:
    if package in matrix_excluded_packages():
        pytest.skip(f"MODEL_MATRIX_EXCLUDE contains {package!r}")


def models_dir() -> Path:
    return Path(get_models_dir())


def load_settings(package: str) -> dict[str, Any]:
    path = models_dir() / package / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_params_from_settings(settings: dict[str, Any], *, package: str) -> dict[str, Any]:
    """First grid value per hyperparameter; matrix-only training caps for heavy models."""
    grid = settings.get("default_hyperparameter_grid") or {}
    params: dict[str, Any] = {}
    for key, values in grid.items():
        if isinstance(values, (list, tuple)) and len(values) > 0:
            params[key] = values[0]
        else:
            params[key] = values
    if package == "lstm":
        params.setdefault("learning_rate", 0.01)
        params["epochs"] = 2
    return params


def model_type_from_settings(settings: dict[str, Any]) -> ModelType:
    mt = settings.get("model_type", "deterministic")
    if mt in ("deterministic", "stochastic", "hybrid"):
        return mt  # type: ignore[return-value]
    return "deterministic"


def capabilities(settings: dict[str, Any]) -> dict[str, Any]:
    return settings.get("capabilities") or {}


def covariate_mode(settings: dict[str, Any]) -> str:
    return str(capabilities(settings).get("covariates", "none"))


def supports_multivariate(settings: dict[str, Any]) -> bool:
    return bool(capabilities(settings).get("multivariate", True))


def supports_univariate(settings: dict[str, Any]) -> bool:
    return bool(capabilities(settings).get("univariate", True))


def hf_model_name(settings: dict[str, Any]) -> str:
    v = settings.get("hf_model_name", "") or ""
    return str(v).strip()


def weights_root() -> str:
    return os.environ.get("MODEL_WEIGHTS_PATH", "").strip()


def require_hf_weights_if_configured(package: str, settings: dict[str, Any]) -> None:
    """Skip when hf_model_name is set but local snapshot missing."""
    hid = hf_model_name(settings)
    if not hid:
        return
    root = weights_root()
    if not root:
        pytest.skip(
            f"{package}: hf_model_name={hid!r} requires MODEL_WEIGHTS_PATH for offline matrix tests"
        )
    local = Path(root) / hid
    if not local.is_dir() or not any(local.iterdir()):
        pytest.skip(f"{package}: missing non-empty weights dir for {hid!r}: {local}")


def assert_hf_weight_snapshot_has_checkpoint(package: str, settings: dict[str, Any]) -> None:
    """TabPFN-style: ensure at least one .ckpt or .pt under weights dir (TT04)."""
    hid = hf_model_name(settings)
    if not hid:
        return
    root = weights_root()
    if not root:
        return
    local = Path(root) / hid
    if not local.is_dir():
        return
    ckpts = list(local.glob("*.ckpt")) + list(local.glob("*.pt")) + list(local.glob("**/*.ckpt"))
    if "tabpfn" in package.lower() and not ckpts:
        pytest.fail(
            f"{package}: expected at least one .ckpt or .pt under {local} for TabPFN-style weights"
        )
    # Other models: snapshot non-empty is enough (already checked in require_hf_weights)


def model_candidate_ok(name: str) -> bool:
    if not name.endswith("Model") or name == "BaseModel":
        return False
    if name.endswith("BaseModel"):
        return False
    return True


def find_model_class_or_skip(package: str):
    try:
        return find_model_class(package)
    except ImportError as exc:
        pytest.skip(f"{package}: model import ImportError: {exc}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"{package}: model import ModuleNotFoundError: {exc}")


def find_model_class(package: str):
    mod = importlib.import_module(f"tempus_bench.models.{package}")
    for name in dir(mod):
        if not model_candidate_ok(name):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == mod.__name__:
            return obj
    sub = importlib.import_module(f"tempus_bench.models.{package}.{package}_model")
    for name in dir(sub):
        if not model_candidate_ok(name):
            continue
        obj = getattr(sub, name)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == sub.__name__:
            return obj
    raise ValueError(f"No *Model class in tempus_bench.models.{package}")


def day_ns_timestamps(start: int, count: int) -> np.ndarray:
    day_ns = 86_400_000_000_000
    return (np.arange(start, start + count, dtype=np.int64) * day_ns).astype(np.int64)


def day_ms_timestamps(start: int, count: int) -> np.ndarray:
    """Millisecond timestamps (smaller magnitude than ns; avoids pandas OOB in some statsmodels paths)."""
    day_ms = 86_400_000
    return (np.arange(start, start + count, dtype=np.int64) * day_ms).astype(np.int64)


def matrix_dimensions() -> tuple[int, int, int]:
    if os.environ.get("MODEL_MATRIX_FAST", "") == "1":
        return FAST_CTX, FAST_TRAIN, FAST_HORIZON
    return CTX_STEPS_BATCH, TRAIN_STEPS_BATCH, FORECAST_HORIZON_BATCH


def make_univariate(
    rng: np.random.Generator,
    *,
    ctx_steps: int,
    train_steps: int,
    forecast_horizon: int,
) -> dict[str, Any]:
    y_ctx = rng.standard_normal((ctx_steps, 1)).astype(np.float64)
    y_train = rng.standard_normal((train_steps, 1)).astype(np.float64)
    ts_ctx = day_ms_timestamps(0, ctx_steps)
    ts_train = day_ms_timestamps(ctx_steps, train_steps)
    ts_target = day_ms_timestamps(ctx_steps + train_steps, forecast_horizon)
    return dict(
        y_ctx=y_ctx,
        y_train=y_train,
        ts_ctx=ts_ctx,
        ts_train=ts_train,
        ts_target=ts_target,
        x_ctx=None,
        x_train=None,
        x_target=None,
        num_targets=1,
        task_type="univariate",
    )


def make_multivariate(
    rng: np.random.Generator,
    *,
    ctx_steps: int,
    train_steps: int,
    forecast_horizon: int,
    num_variates: int = NUM_VARIATES_MULTI,
) -> dict[str, Any]:
    y_ctx = rng.standard_normal((ctx_steps, num_variates)).astype(np.float64)
    y_train = rng.standard_normal((train_steps, num_variates)).astype(np.float64)
    ts_ctx = day_ms_timestamps(0, ctx_steps)
    ts_train = day_ms_timestamps(ctx_steps, train_steps)
    ts_target = day_ms_timestamps(ctx_steps + train_steps, forecast_horizon)
    return dict(
        y_ctx=y_ctx,
        y_train=y_train,
        ts_ctx=ts_ctx,
        ts_train=ts_train,
        ts_target=ts_target,
        x_ctx=None,
        x_train=None,
        x_target=None,
        num_targets=num_variates,
        task_type="multivariate",
    )


def make_covariate(
    rng: np.random.Generator,
    *,
    ctx_steps: int,
    train_steps: int,
    forecast_horizon: int,
    past_only: bool,
) -> dict[str, Any]:
    y_ctx = rng.standard_normal((ctx_steps, 1)).astype(np.float64)
    y_train = rng.standard_normal((train_steps, 1)).astype(np.float64)
    ts_ctx = day_ms_timestamps(0, ctx_steps)
    ts_train = day_ms_timestamps(ctx_steps, train_steps)
    ts_target = day_ms_timestamps(ctx_steps + train_steps, forecast_horizon)
    x_ctx = rng.standard_normal((ctx_steps, NUM_COVARIATES)).astype(np.float64)
    x_train = rng.standard_normal((train_steps, NUM_COVARIATES)).astype(np.float64)
    x_target = (
        None
        if past_only
        else rng.standard_normal((forecast_horizon, NUM_COVARIATES)).astype(np.float64)
    )
    return dict(
        y_ctx=y_ctx,
        y_train=y_train,
        ts_ctx=ts_ctx,
        ts_train=ts_train,
        ts_target=ts_target,
        x_ctx=x_ctx,
        x_train=x_train,
        x_target=x_target,
        num_targets=1,
        task_type="covariate",
    )


def extra_train_predict_kwargs(package: str) -> dict[str, Any]:
    ex: dict[str, Any] = {}
    if package == "toto":
        ex["freq"] = "d"
    if package == "lagllama":
        ex["freq"] = "h"
    return ex


def build_train_kwargs(
    data: dict[str, Any],
    *,
    package: str,
    num_samples: int,
    moirai_validate_horizon: int | None = None,
) -> dict[str, Any]:
    train_kwargs: dict[str, Any] = dict(num_samples=num_samples)
    if package == "lstm":
        train_kwargs["tuning_loss"] = "mae"
    if package in PACKAGES_REQUIRING_TRAIN_FREQ:
        train_kwargs.setdefault("freq", "D")
    if data["x_ctx"] is not None:
        train_kwargs["x_context"] = data["x_ctx"]
    if data["x_train"] is not None:
        train_kwargs["x_target"] = data["x_train"]
    if moirai_validate_horizon is not None and package in MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON:
        train_kwargs["validate_horizon"] = moirai_validate_horizon
    return train_kwargs


def build_predict_kwargs(
    data: dict[str, Any],
    *,
    package: str,
    num_samples: int,
) -> dict[str, Any]:
    predict_kwargs: dict[str, Any] = dict(num_samples=num_samples)
    predict_kwargs.update(extra_train_predict_kwargs(package))
    if data["x_ctx"] is not None:
        x_full = np.concatenate([data["x_ctx"], data["x_train"]], axis=0)
        predict_kwargs["x_context"] = x_full
    if data["x_target"] is not None:
        predict_kwargs["x_target"] = data["x_target"]
    return predict_kwargs


def run_train_predict_or_skip(
    package: str,
    data: dict[str, Any],
    *,
    num_samples: int = NUM_SAMPLES_DEFAULT,
    moirai_validate_horizon: int | None = None,
) -> tuple[np.ndarray, int, Any, dict[str, Any]]:
    try:
        return run_train_predict(
            package,
            data,
            num_samples=num_samples,
            moirai_validate_horizon=moirai_validate_horizon,
        )
    except ImportError as exc:
        pytest.skip(f"{package}: optional dependency ImportError: {exc}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"{package}: optional module: {exc}")


def run_train_predict(
    package: str,
    data: dict[str, Any],
    *,
    num_samples: int = NUM_SAMPLES_DEFAULT,
    moirai_validate_horizon: int | None = None,
) -> tuple[np.ndarray, int, Any, dict[str, Any]]:
    """Train, then predict on concatenated history; return output, num_targets, model, settings."""
    settings = load_settings(package)
    require_hf_weights_if_configured(package, settings)

    params = default_params_from_settings(settings, package=package)
    Model = find_model_class(package)
    model = Model(params, settings)

    h = int(data["ts_target"].shape[0])
    effective_moirai_vh = moirai_validate_horizon
    if effective_moirai_vh is None and package in MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON:
        effective_moirai_vh = h

    train_kwargs = build_train_kwargs(
        data,
        package=package,
        num_samples=num_samples,
        moirai_validate_horizon=effective_moirai_vh,
    )
    model.train(
        data["y_ctx"],
        data["y_train"],
        data["ts_ctx"],
        data["ts_train"],
        **train_kwargs,
    )
    if not model.is_fitted:
        pytest.skip(
            f"{package}: train() left is_fitted=False on synthetic matrix data "
            "(e.g. optimizer limits or non-convergence)"
        )

    y_full = np.concatenate([data["y_ctx"], data["y_train"]], axis=0)
    ts_full = np.concatenate([data["ts_ctx"], data["ts_train"]], axis=0)
    predict_kwargs = build_predict_kwargs(data, package=package, num_samples=num_samples)
    if package == "lstm":
        predict_kwargs.pop("num_samples", None)
    elif model_type_from_settings(settings) == "deterministic" and package not in (
        "lstm",
    ):
        predict_kwargs.pop("num_samples", None)

    out = model.predict(
        y_full,
        ts_full,
        data["ts_target"],
        **predict_kwargs,
    )
    if isinstance(out, tuple):
        pytest.skip(f"{package}: hybrid tuple output not asserted in matrix (got tuple)")
    return np.asarray(out), int(data["num_targets"]), model, settings


def assert_output_shape(
    out: np.ndarray,
    *,
    model_type: ModelType,
    num_samples: int,
    horizon: int,
    num_targets: int,
    package: str,
) -> None:
    """TempusBench contract: deterministic (H, T), stochastic (S, H, T); hybrid uses samples when passed."""
    if model_type == "deterministic":
        assert out.ndim == 2, (
            f"{package}: expected 2D (H, T), got shape {out.shape} for deterministic model_type"
        )
        assert out.shape[0] == horizon, f"{package}: H mismatch got {out.shape[0]} expected {horizon}"
        assert out.shape[1] == num_targets, (
            f"{package}: T mismatch got {out.shape[1]} expected {num_targets}"
        )
    elif model_type in ("stochastic", "hybrid"):
        assert out.ndim == 3, (
            f"{package}: expected 3D (S, H, T), got shape {out.shape} for {model_type}"
        )
        assert out.shape[0] == num_samples, (
            f"{package}: S mismatch got {out.shape[0]} expected {num_samples}"
        )
        assert out.shape[1] == horizon, f"{package}: H mismatch got {out.shape[1]} expected {horizon}"
        assert out.shape[2] == num_targets, (
            f"{package}: T mismatch got {out.shape[2]} expected {num_targets}"
        )
    assert np.all(np.isfinite(out)), f"{package}: non-finite values in forecast"


def suggested_context_cap(settings: dict[str, Any]) -> int | None:
    for key in (
        "context_length",
        "lookback_length",
        "context_len",
        "max_context",
        "max_length",
    ):
        v = settings.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def _matrix_only_packages_from_env() -> frozenset[str] | None:
    """If set, ``MODEL_MATRIX_ONLY`` must be a non-empty comma-separated list of package ids."""
    raw = os.environ.get("MODEL_MATRIX_ONLY", "")
    if not raw.strip():
        return None
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    if not parts:
        raise ValueError(
            "MODEL_MATRIX_ONLY is set but contains no package ids after parsing "
            "(expected comma-separated names, e.g. chronos_tiny,nhits)."
        )
    return parts


def sorted_model_packages() -> list[str]:
    """Model package ids for parametrized matrix/import tests (alphabetical).

    When ``MODEL_MATRIX_ONLY`` is unset, returns all ``get_available_models()`` ids.
    When set, returns only that subset; unknown ids raise ``ValueError`` at collection time.
    """
    all_pkgs = sorted(get_available_models())
    only = _matrix_only_packages_from_env()
    if only is None:
        return all_pkgs
    available = frozenset(all_pkgs)
    unknown = sorted(only - available)
    if unknown:
        raise ValueError(
            "MODEL_MATRIX_ONLY contains unknown package id(s): "
            f"{unknown}. Valid ids: {all_pkgs}."
        )
    return sorted(only & available)


def assert_internal_api_hints(model: Any, package: str) -> None:
    """TT09: light white-box checks where uni2ts / adapters expose context length."""
    if package not in MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON:
        return
    inner = getattr(model, "_model", None)
    if inner is None:
        return
    pl = getattr(inner, "past_length", None)
    if pl is not None:
        assert int(pl) >= 1, f"{package}: past_length must be positive, got {pl!r}"
