"""
End-to-end checks: every catalog task loads through DataLoader + Preprocessor,
produces finite arrays, and yields as many rolling windows as ``evaluation.max_windows`` (capped at 5).

Run from repo root:
    pytest tests/integration/test_all_tasks_loader_preprocessor.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import (
    MAX_EVALUATION_WINDOWS,
    DatasetConfig,
    EvaluationConfig,
    TaskConfig,
)
from tempus_bench.utils.log_manager import LogManager
from tempus_bench.utils.paths import find_task_directories


@pytest.fixture(autouse=True)
def _log_manager():
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


def _task_configs() -> list[tuple[str, TaskConfig]]:
    out: list[tuple[str, TaskConfig]] = []
    for name, path in sorted(find_task_directories("*").items()):
        p = Path(path)
        with open(p / "task.yaml", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
        for task_data in docs:
            if not task_data or "task" not in task_data:
                continue
            raw = dict(task_data["task"])
            raw.pop("task_name", None)
            dataset = DatasetConfig(**raw.pop("dataset"))
            tc = TaskConfig(task_name=name, task_path=str(p), **raw, dataset=dataset)
            out.append((name, tc))
    return out


_TASK_CASES = _task_configs()


@pytest.mark.parametrize(
    "task_name,task_config",
    _TASK_CASES,
    ids=[n for n, _ in _TASK_CASES],
)
def test_task_loads_clean_finite_and_has_windows(task_name: str, task_config: TaskConfig):
    eval_cfg = EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=MAX_EVALUATION_WINDOWS,
        max_num_variates=100,
        num_samples=10,
        num_quantiles=10,
        point_forecast_statistic="mean",
    )
    loader = DataLoader(task_config, eval_cfg)
    ds = loader.dataset
    target = np.asarray(ds.target, dtype=float)
    assert target.ndim == 2
    assert np.isfinite(target).all(), f"{task_name}: non-finite target"
    assert not np.isnan(target).any(), f"{task_name}: NaN in target"
    assert len(ds.timestamps) == target.shape[0]

    if ds.covariate is not None:
        cov = np.asarray(ds.covariate, dtype=float)
        assert cov.shape[0] == target.shape[0]
        assert np.isfinite(cov).all(), f"{task_name}: non-finite covariate"

    steps = [
        ("context", task_config.context_window),
        ("train", task_config.forecast_horizon),
        ("validate", task_config.forecast_horizon),
    ]
    window_size = sum(s[1] for s in steps)
    assert len(ds.target) >= window_size, (
        f"{task_name}: series length {len(ds.target)} < window {window_size} "
        f"(context={task_config.context_window}, fh={task_config.forecast_horizon})"
    )
    windows = list(
        ds.generate_dataset_split(
            steps=steps,
            stride=task_config.forecast_horizon,
            max_windows=eval_cfg.max_windows,
        )
    )
    assert len(windows) >= 1, f"{task_name}: no rolling windows produced"
    assert len(windows) >= eval_cfg.max_windows, (
        f"{task_name}: expected at least {eval_cfg.max_windows} rolling windows, got {len(windows)} "
        f"(series length {len(ds.target)}, context={task_config.context_window}, fh={task_config.forecast_horizon})"
    )
