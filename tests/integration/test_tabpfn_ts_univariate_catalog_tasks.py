"""tabpfn_ts on real catalog univariate tasks (first rolling window).

Covers tasks that failed in cloud before the ``TabPFNTSPipeline`` ctor fix:
synthetic additive2 / cyclic / multiplicative / nonstationary, and web traffic.

Uses ``DataLoader`` + first window (same layout as ``model_executor``).
TensorBoard is disabled and TensorFlow thread env is capped **before** pipeline
imports to avoid slow startup on macOS.

Run::

  RUN_TABPFN_TS_CATALOG_TEST=1 TABPFN_TS_FAST=1 pytest tests/integration/test_tabpfn_ts_univariate_catalog_tasks.py -v --tb=short

``TABPFN_TS_FAST=1`` caps context to 96 and horizon to 12 (still real CSV + yaml).
"""

from __future__ import annotations

import importlib
import os

# Before tempus_bench (TensorFlow); mirrors cloud worker defaults where applicable.
os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import (
    MAX_EVALUATION_WINDOWS,
    EvaluationConfig,
    TaskConfig,
)
from tempus_bench.utils.log_manager import LogManager
from tempus_bench.utils.paths import find_task_directories, get_models_dir
from tempus_bench.utils.task_yaml_loader import build_task_configs
from tempus_bench.utils.utils import compute_point_forecast

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TABPFN_TS_CATALOG_TEST", "") != "1",
    reason="Set RUN_TABPFN_TS_CATALOG_TEST=1 (tabpfn-time-series; may download weights).",
)

PACKAGE = "tabpfn_ts"

CATALOG_UNIVARIATE_TASKS: tuple[str, ...] = (
    "univariate_climate_daily_mean_humidity_delhi",
    "univariate_web_hourly_hourly_web_requests",
    "univariate_nature_minutes_soil_moisture",
)

NUM_SAMPLES = 4


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


def _task_config_for_folder(task_key: str) -> TaskConfig:
    paths = find_task_directories("*")
    if task_key not in paths:
        raise KeyError(f"Unknown task folder {task_key!r}")
    p = Path(paths[task_key])
    return build_task_configs(task_key, p)[0]


def _context_train_validate(task_config: TaskConfig) -> tuple[int, int, int]:
    ctx = int(task_config.context_window)
    fh = int(task_config.forecast_horizon)
    if os.environ.get("TABPFN_TS_FAST", "").strip() == "1":
        ctx = min(ctx, 96)
        fh = min(fh, 12)
    return ctx, fh, fh


@pytest.fixture(scope="module", autouse=True)
def _log_manager_module():
    LogManager.log_manager = None
    with tempfile.TemporaryDirectory() as d:
        lm = LogManager(
            logs_path=d,
            console_logging=False,
            file_logging=False,
            tensorboard_logging=False,
        )
        yield lm
        try:
            lm.close()
        except Exception:
            pass
        LogManager.log_manager = None


def _run_first_window_tabpfn_ts(task_key: str) -> None:
    pytest.importorskip("tabpfn_time_series")
    tc = _task_config_for_folder(task_key)
    ctx_steps, train_steps, val_steps = _context_train_validate(tc)
    eval_cfg = EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=MAX_EVALUATION_WINDOWS,
        max_num_variates=100,
        num_samples=NUM_SAMPLES,
        num_quantiles=10,
        point_forecast_statistic="mean",
    )
    loader = DataLoader(tc, eval_cfg)
    ds = loader.dataset
    timestamps = np.asarray(ds.timestamps)
    target = np.asarray(ds.target, dtype=np.float64)
    assert timestamps.ndim == 1 and target.ndim == 2

    steps = [
        ("context", ctx_steps),
        ("train", train_steps),
        ("validate", val_steps),
    ]
    dataset_splits = next(
        ds.generate_dataset_split(
            steps=steps,
            stride=val_steps,
            max_windows=1,
        )
    )

    cstart = dataset_splits["context"].start
    cend = dataset_splits["context"].end
    tstart = dataset_splits["train"].start
    tend = dataset_splits["train"].end
    vstart = dataset_splits["validate"].start
    vend = dataset_splits["validate"].end

    settings = _load_settings(PACKAGE)
    Model = _find_model_class(PACKAGE)
    model = Model({}, settings)

    model.train(
        target[cstart:cend],
        target[tstart:tend],
        timestamps[cstart:cend],
        timestamps[tstart:tend],
        num_samples=NUM_SAMPLES,
    )
    assert model.is_fitted

    y_full = target[cstart:tend]
    ts_full = timestamps[cstart:tend]
    ts_target = timestamps[vstart:vend]

    out = model.predict(y_full, ts_full, ts_target, num_samples=NUM_SAMPLES)
    out = np.asarray(out)
    hz = vend - vstart
    assert out.ndim == 3, f"{task_key}: expected (S,H,T), got {out.shape}"
    assert out.shape[0] == NUM_SAMPLES
    assert out.shape[1] == hz
    assert out.shape[2] == target.shape[1]
    assert np.all(np.isfinite(out)), f"{task_key}: non-finite predictions"

    pt = compute_point_forecast(out, settings.get("point_forecast_statistic", "mean"))
    assert pt.shape == (hz, target.shape[1])


@pytest.mark.parametrize("task_key", CATALOG_UNIVARIATE_TASKS)
def test_tabpfn_ts_first_window_catalog_univariate(task_key: str) -> None:
    _run_first_window_tabpfn_ts(task_key)
