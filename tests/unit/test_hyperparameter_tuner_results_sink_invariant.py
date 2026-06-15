"""Invariant: when job_id + results_callback are set, every evaluated window must call the sink."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tempus_bench.pipeline.hyperparameter_tuner import HyperparameterTuner
from tempus_bench.utils.configs import (
    EvaluationConfig,
    EvaluationSetting,
    JobConfig,
    ModelConfig,
    TaskConfig,
)
from tempus_bench.utils.log_manager import LogManager
from tempus_bench.utils.model_settings import load_model_settings_yaml
from tempus_bench.utils.task_yaml_loader import load_task_config_from_task_dir

_CATALOG_UNIVARIATE = "univariate_climate_daily_mean_humidity_delhi"


def _catalog_task_config() -> TaskConfig:
    repo = Path(__file__).resolve().parents[2]
    task_dir = (
        repo / "tempus_bench" / "tasks" / "univariate" / _CATALOG_UNIVARIATE
    )
    return load_task_config_from_task_dir(task_dir)


@pytest.fixture
def _init_log_manager():
    LogManager.reset_singleton()
    with tempfile.TemporaryDirectory() as d:
        lm = LogManager(
            logs_path=d,
            console_logging=False,
            file_logging=False,
            tensorboard_logging=False,
        )
        yield
        try:
            lm.close()
        except Exception:
            pass
        LogManager.reset_singleton()


def _minimal_job_config(*, run_path: str) -> JobConfig:
    task_config = _catalog_task_config()
    evaluation_config = EvaluationConfig(
        task_path=f"univariate/{task_config.task_name}",
        tuning_loss="mae",
        max_windows=2,
        num_samples=5,
        num_quantiles=3,
        point_forecast_statistic="mean",
    )
    evaluation_setting = EvaluationSetting(
        file_logging=False,
        console_logging=False,
        tensorboard_logging=False,
        conda_env_prefix="benchmark",
        reinstall_conda=False,
        verbose=False,
    )
    model_config = ModelConfig("tiny_time_mixer_r2_1")
    model_setting = dict(load_model_settings_yaml("tiny_time_mixer_r2_1"))
    return JobConfig(
        evaluation_config=evaluation_config,
        evaluation_setting=evaluation_setting,
        model_config=model_config,
        model_setting=model_setting,
        task_config=task_config,
        run_path=run_path,
        task_datasets_dir=None,
    )


def _one_window_eval_outputs() -> dict:
    return {
        "mae": float(np.float64(0.42)),
        "y_true": np.array([1.0, 2.0]),
        "y_pred": np.array([[1.1, 2.1], [1.2, 2.2]]),
        "timestamps_pred": np.array(
            ["2020-01-01T00:00:00", "2020-01-02T00:00:00"], dtype="datetime64[ns]"
        ),
    }


def test_results_callback_invoked_once_per_window(_init_log_manager) -> None:
    with tempfile.TemporaryDirectory() as run_root:
        job_config = _minimal_job_config(run_path=run_root)

        class _FakeModelExecutor:
            def __init__(self, *_a, **_k) -> None:
                pass

            def execute_model(self, **_kwargs):
                return [_one_window_eval_outputs(), _one_window_eval_outputs()]

        def _tqdm_identity(iterable, **_kwargs):
            return iterable

        sink = MagicMock()

        with (
            patch(
                "tempus_bench.pipeline.hyperparameter_tuner.ModelExecutor",
                _FakeModelExecutor,
            ),
            patch("tempus_bench.pipeline.hyperparameter_tuner.tqdm", _tqdm_identity),
            patch(
                "tempus_bench.pipeline.hyperparameter_tuner.Visualizer",
                return_value=MagicMock(),
            ),
        ):
            tuner = HyperparameterTuner(
                job_config=job_config,
                job_id="sink-test-job",
                results_callback=sink,
            )
            tuner.optimize_hyperparameters(
                context_steps=job_config.task_config.context_window,
                train_steps=job_config.task_config.forecast_horizon,
                validate_steps=job_config.task_config.forecast_horizon,
            )

        assert sink.call_count == 2


def test_mismatch_results_sink_invocations_raises(_init_log_manager) -> None:
    """Simulate a bug where later windows skip the sink while still completing the loop."""

    with tempfile.TemporaryDirectory() as run_root:
        job_config = _minimal_job_config(run_path=run_root)

        class _FakeModelExecutor:
            def __init__(self, *_a, **_k) -> None:
                pass

            def execute_model(self, **_kwargs):
                return [_one_window_eval_outputs(), _one_window_eval_outputs()]

        def _tqdm_identity(iterable, **_kwargs):
            return iterable

        tuner_holder: dict = {}

        def _strip_callback_after_first(*_a, **_k) -> None:
            tuner_holder["t"].results_callback = None

        with (
            patch(
                "tempus_bench.pipeline.hyperparameter_tuner.ModelExecutor",
                _FakeModelExecutor,
            ),
            patch("tempus_bench.pipeline.hyperparameter_tuner.tqdm", _tqdm_identity),
            patch(
                "tempus_bench.pipeline.hyperparameter_tuner.Visualizer",
                return_value=MagicMock(),
            ),
        ):
            tuner = HyperparameterTuner(
                job_config=job_config,
                job_id="sink-test-job",
                results_callback=_strip_callback_after_first,
            )
            tuner_holder["t"] = tuner
            with pytest.raises(RuntimeError, match="results_callback mismatch"):
                tuner.optimize_hyperparameters(
                    context_steps=job_config.task_config.context_window,
                    train_steps=job_config.task_config.forecast_horizon,
                    validate_steps=job_config.task_config.forecast_horizon,
                )
