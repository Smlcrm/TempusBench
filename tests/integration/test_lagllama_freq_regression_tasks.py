"""
Regression: Lag-Llama ``_predict`` used ``pd.date_range(..., freq=...)`` with metadata
``time_freq`` values like ``ME`` / ``YE`` that older pandas in ``benchmark.lagllama`` rejects.

This integration test loads real catalog tasks (no torch / conda) and asserts the same
freq normalization + coercion + ``date_range`` path used by ``lagllama_model`` succeeds.

Tasks (per product request):
  - Employees Healthcare (univariate)
  - Power Consumption Years (univariate)
  - NYC COVID Healthcare (multivariate)
  - NYC COVID Healthcare (covariate)

Run from TempusBench repo root::

    pytest tests/integration/test_lagllama_freq_regression_tasks.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import DatasetConfig, EvaluationConfig, TaskConfig
from tempus_bench.utils.lagllama_freq import coerce_freq_for_pandas_date_range
from tempus_bench.utils.lagllama_freq import normalize_freq_for_lagllama
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


# Catalog paths aligned with cloud ``benchmark-metadata.json`` task ids.
LAGLLAMA_FREQ_REGRESSION_TASKS: tuple[str, ...] = (
    "univariate/employees_healthcare_univariate",
    "univariate/power_consumption_years_univariate",
    "multivariate/nyc_covid_healthcare_multivariate",
    "covariate/nyc_covid_healthcare_covariate",
)


def _load_task_config(task_rel_path: str) -> TaskConfig:
    found = find_task_directories(task_rel_path)
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one task dir for {task_rel_path!r}, got {list(found.keys())}"
        )
    _name, path = next(iter(found.items()))
    p = Path(path)
    with open(p / "task.yaml", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    task_data = next(d for d in docs if d and "task" in d)
    raw = dict(task_data["task"])
    raw.pop("task_name", None)
    dataset = DatasetConfig(**raw.pop("dataset"))
    return TaskConfig(task_name=_name, task_path=str(p), **raw, dataset=dataset)


@pytest.mark.parametrize("task_rel_path", LAGLLAMA_FREQ_REGRESSION_TASKS)
def test_lagllama_freq_pipeline_matches_model_executor_and_date_range(
    task_rel_path: str,
) -> None:
    """Mirror ``model_executor`` lagllama branch + ``lagllama_model._predict`` freq usage."""
    task_config = _load_task_config(task_rel_path)
    eval_cfg = EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=2,
        max_num_variates=100,
        num_samples=5,
        num_quantiles=3,
        point_forecast_statistic="mean",
    )
    loader = DataLoader(task_config, eval_cfg)
    ds = loader.dataset
    meta = ds.metadata
    assert isinstance(meta, dict)
    raw_freq = meta.get("time_freq")
    assert raw_freq is not None and str(raw_freq).strip(), (
        f"{task_rel_path}: missing time_freq in dataset metadata"
    )

    # Same as model_executor NEEDS_LEGACY_FREQ branch for lagllama.
    freq = normalize_freq_for_lagllama(str(raw_freq))
    freq_pd = coerce_freq_for_pandas_date_range(freq)

    ts = np.asarray(ds.timestamps)
    timestamps_ctx = ts[: task_config.context_window]
    start_time = pd.Timestamp(timestamps_ctx[0])
    periods = int(timestamps_ctx.shape[0])

    idx = pd.date_range(start=start_time, periods=periods, freq=freq_pd)
    assert len(idx) == periods, f"{task_rel_path}: date_range length mismatch"

    # Multivariate / covariate: ensure at least one rolling window exists (sanity).
    steps = [
        ("context", task_config.context_window),
        ("train", task_config.forecast_horizon),
        ("validate", task_config.forecast_horizon),
    ]
    windows = list(
        ds.generate_dataset_split(
            steps=steps,
            stride=task_config.forecast_horizon,
            max_windows=1,
        )
    )
    assert len(windows) >= 1, f"{task_rel_path}: no rolling windows"
