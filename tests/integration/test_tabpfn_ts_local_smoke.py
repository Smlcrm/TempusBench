"""Local smoke: tabpfn_ts train→predict for univariate, multivariate, past+future covariate.

Mirrors cloud-failing shapes (regression: ``TabPFNTSPipeline`` ctor must match pinned
``tabpfn-time-series``). Uses **nanosecond** timestamps — ``tabpfn_ts_model`` passes
``unit="ns"`` to pandas.

Requires:
  - ``pip install -r tempus_bench/models/tabpfn_ts/requirements.txt`` (or conda env)
  - Network on first run (TabPFN / checkpoint fetch) unless weights are already cached

Run::

  RUN_TABPFN_TS_LOCAL_TEST=1 pytest tests/integration/test_tabpfn_ts_local_smoke.py -v --tb=short

Optional smaller windows::

  TABPFN_TS_FAST=1 pytest ...  # ctx=32 train=12 horizon=6
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tempus_bench.utils.paths import get_models_dir
from tempus_bench.utils.utils import compute_point_forecast


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


def _load_settings(package: str) -> dict[str, Any]:
    path = Path(get_models_dir()) / package / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_type_from_settings(settings: dict[str, Any]) -> str:
    mt = settings.get("model_type", "deterministic")
    if mt in ("deterministic", "stochastic", "hybrid"):
        return str(mt)
    return "deterministic"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TABPFN_TS_LOCAL_TEST", "") != "1",
    reason="Set RUN_TABPFN_TS_LOCAL_TEST=1 (optional deps + may download model weights).",
)

PACKAGE = "tabpfn_ts"
NUM_COVARIATES = 2
NUM_VARIATES_MULTI = 3
NUM_SAMPLES = 4


def _dims() -> tuple[int, int, int]:
    if os.environ.get("TABPFN_TS_FAST", "").strip() == "1":
        return 32, 12, 6
    return 48, 16, 8


def _day_ns_timestamps(start: int, count: int) -> np.ndarray:
    day_ns = 86_400_000_000_000
    return (np.arange(start, start + count, dtype=np.int64) * day_ns).astype(np.int64)


def _make_univariate(rng: np.random.Generator) -> dict[str, Any]:
    ctx, tr, hz = _dims()
    y_ctx = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_train = rng.standard_normal((tr, 1)).astype(np.float64)
    ts_ctx = _day_ns_timestamps(0, ctx)
    ts_train = _day_ns_timestamps(ctx, tr)
    ts_target = _day_ns_timestamps(ctx + tr, hz)
    return {
        "y_ctx": y_ctx,
        "y_train": y_train,
        "ts_ctx": ts_ctx,
        "ts_train": ts_train,
        "ts_target": ts_target,
        "x_ctx": None,
        "x_train": None,
        "x_target": None,
        "num_targets": 1,
    }


def _make_multivariate(rng: np.random.Generator) -> dict[str, Any]:
    ctx, tr, hz = _dims()
    y_ctx = rng.standard_normal((ctx, NUM_VARIATES_MULTI)).astype(np.float64)
    y_train = rng.standard_normal((tr, NUM_VARIATES_MULTI)).astype(np.float64)
    ts_ctx = _day_ns_timestamps(0, ctx)
    ts_train = _day_ns_timestamps(ctx, tr)
    ts_target = _day_ns_timestamps(ctx + tr, hz)
    return {
        "y_ctx": y_ctx,
        "y_train": y_train,
        "ts_ctx": ts_ctx,
        "ts_train": ts_train,
        "ts_target": ts_target,
        "x_ctx": None,
        "x_train": None,
        "x_target": None,
        "num_targets": NUM_VARIATES_MULTI,
    }


def _make_covariate_past_future(rng: np.random.Generator) -> dict[str, Any]:
    ctx, tr, hz = _dims()
    y_ctx = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_train = rng.standard_normal((tr, 1)).astype(np.float64)
    ts_ctx = _day_ns_timestamps(0, ctx)
    ts_train = _day_ns_timestamps(ctx, tr)
    ts_target = _day_ns_timestamps(ctx + tr, hz)
    x_ctx = rng.standard_normal((ctx, NUM_COVARIATES)).astype(np.float64)
    x_train = rng.standard_normal((tr, NUM_COVARIATES)).astype(np.float64)
    x_target = rng.standard_normal((hz, NUM_COVARIATES)).astype(np.float64)
    return {
        "y_ctx": y_ctx,
        "y_train": y_train,
        "ts_ctx": ts_ctx,
        "ts_train": ts_train,
        "ts_target": ts_target,
        "x_ctx": x_ctx,
        "x_train": x_train,
        "x_target": x_target,
        "num_targets": 1,
    }


def _run_train_predict(data: dict[str, Any]) -> np.ndarray:
    pytest.importorskip("tabpfn_time_series")
    settings = _load_settings(PACKAGE)
    Model = _find_model_class(PACKAGE)
    model = Model({}, settings)

    train_kw: dict[str, Any] = {"num_samples": NUM_SAMPLES}
    if data["x_ctx"] is not None:
        train_kw["x_context"] = data["x_ctx"]
    if data["x_train"] is not None:
        train_kw["x_target"] = data["x_train"]

    model.train(
        data["y_ctx"],
        data["y_train"],
        data["ts_ctx"],
        data["ts_train"],
        **train_kw,
    )
    assert model.is_fitted

    y_full = np.concatenate([data["y_ctx"], data["y_train"]], axis=0)
    ts_full = np.concatenate([data["ts_ctx"], data["ts_train"]], axis=0)
    pred_kw: dict[str, Any] = {"num_samples": NUM_SAMPLES}
    if data["x_ctx"] is not None:
        pred_kw["x_context"] = np.concatenate([data["x_ctx"], data["x_train"]], axis=0)
    if data["x_target"] is not None:
        pred_kw["x_target"] = data["x_target"]

    out = model.predict(y_full, ts_full, data["ts_target"], **pred_kw)
    return np.asarray(out)


def _assert_stochastic(out: np.ndarray, *, horizon: int, num_targets: int) -> None:
    assert out.ndim == 3, f"expected (S,H,T), got {out.shape}"
    assert out.shape[0] == NUM_SAMPLES
    assert out.shape[1] == horizon
    assert out.shape[2] == num_targets
    assert np.all(np.isfinite(out)), "non-finite predictions"


def test_tabpfn_ts_univariate_smoke() -> None:
    """Analogous to univariate tasks (e.g. federal_funds_weeks_univariate)."""
    rng = np.random.default_rng(701)
    data = _make_univariate(rng)
    _, _, hz = _dims()
    out = _run_train_predict(data)
    _assert_stochastic(out, horizon=hz, num_targets=1)
    settings = _load_settings(PACKAGE)
    assert _model_type_from_settings(settings) == "stochastic"
    pt = compute_point_forecast(out, settings.get("point_forecast_statistic", "mean"))
    assert pt.shape == (hz, 1)


def test_tabpfn_ts_multivariate_smoke() -> None:
    """Analogous to multivariate tasks (e.g. gold_india_continuous_multivariate)."""
    rng = np.random.default_rng(702)
    data = _make_multivariate(rng)
    _, _, hz = _dims()
    out = _run_train_predict(data)
    _assert_stochastic(out, horizon=hz, num_targets=NUM_VARIATES_MULTI)


def test_tabpfn_ts_covariate_past_future_smoke() -> None:
    """Analogous to covariate tasks with future exog (e.g. building_manufacturing_covariate)."""
    rng = np.random.default_rng(703)
    data = _make_covariate_past_future(rng)
    _, _, hz = _dims()
    out = _run_train_predict(data)
    _assert_stochastic(out, horizon=hz, num_targets=1)
