"""When ModelExecutor yields no windows, HyperparameterTuner logs and returns empty evals (run can still complete)."""

from __future__ import annotations

import os
import tempfile

# Before any tempus_bench import: avoid TensorFlow init in LogManager (macOS mutex hangs).
os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tempus_bench.pipeline.hyperparameter_tuner import HyperparameterTuner
from tempus_bench.utils.configs import (
    DatasetConfig,
    EvaluationConfig,
    EvaluationSetting,
    JobConfig,
    ModelConfig,
    TaskConfig,
)
from tempus_bench.utils.log_manager import LogManager
from tempus_bench.utils.model_settings import load_model_settings_yaml


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


def _minimal_job_config() -> JobConfig:
    evaluation_config = EvaluationConfig(
        task_path="univariate/chickenpox_dense_univariate",
        tuning_loss="mae",
        max_windows=1,
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
    repo = Path(__file__).resolve().parents[2]
    task_config = TaskConfig(
        task_name="chickenpox_dense_univariate",
        task_path=str(
            repo / "tempus_bench" / "tasks" / "univariate" / "chickenpox_dense_univariate"
        ),
        forecast_horizon=12,
        context_window=64,
        dataset=DatasetConfig(file_name="chickenpox_dense_univariate.csv"),
    )
    run_path = str(Path(__file__).resolve().parent / "_hp_tuner_test_run_path")
    return JobConfig(
        evaluation_config=evaluation_config,
        evaluation_setting=evaluation_setting,
        model_config=model_config,
        model_setting=model_setting,
        task_config=task_config,
        run_path=run_path,
        task_datasets_dir=None,
    )


def test_optimize_hyperparameters_returns_empty_when_execute_model_returns_empty(
    _init_log_manager,
) -> None:
    job_config = _minimal_job_config()

    class _FakeModelExecutor:
        def __init__(self, *_a, **_k) -> None:
            pass

        def execute_model(self, **_kwargs):
            return []

    def _tqdm_identity(iterable, **_kwargs):
        return iterable

    mock_log = MagicMock()
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
        patch.object(LogManager, "get_logger", return_value=mock_log),
    ):
        tuner = HyperparameterTuner(
            job_config=job_config,
            job_id="test-job-id",
            results_callback=None,
        )
        evals, hp = tuner.optimize_hyperparameters(
            context_steps=job_config.task_config.context_window,
            train_steps=job_config.task_config.forecast_horizon,
            validate_steps=job_config.task_config.forecast_horizon,
        )
    assert evals == {} and hp == {}
    mock_log.error.assert_called()
    err_msg = str(mock_log.error.call_args[0][1])
    assert "No successful evaluation windows" in err_msg
    assert "Hint:" in err_msg


def test_optimize_hyperparameters_logs_last_trial_error_when_all_trials_fail(
    _init_log_manager,
) -> None:
    job_config = _minimal_job_config()

    class _FailingExecutor:
        def __init__(self, *_a, **_k) -> None:
            pass

        def execute_model(self, **_kwargs):
            raise RuntimeError(
                "Failed to run in conda env (benchmark.tiny_time_mixer_r2_1).\n"
                "Exit code: 1\nStandard Output:\nTraceback (most recent call last):\n"
                "  File \"<fake>\", line 1, in <module>\nValueError: context too long\n"
            )

    def _tqdm_identity(iterable, **_kwargs):
        return iterable

    mock_log = MagicMock()
    with (
        patch(
            "tempus_bench.pipeline.hyperparameter_tuner.ModelExecutor",
            _FailingExecutor,
        ),
        patch("tempus_bench.pipeline.hyperparameter_tuner.tqdm", _tqdm_identity),
        patch(
            "tempus_bench.pipeline.hyperparameter_tuner.Visualizer",
            return_value=MagicMock(),
        ),
        patch.object(LogManager, "get_logger", return_value=mock_log),
    ):
        tuner = HyperparameterTuner(
            job_config=job_config,
            job_id="test-job-id",
            results_callback=None,
        )
        evals, hp = tuner.optimize_hyperparameters(
            context_steps=job_config.task_config.context_window,
            train_steps=job_config.task_config.forecast_horizon,
            validate_steps=job_config.task_config.forecast_horizon,
        )
    assert evals == {} and hp == {}
    err_msg = str(mock_log.error.call_args[0][1])
    assert "Last trial error:" in err_msg
    assert "context too long" in err_msg


def test_tiny_time_mixer_r2_1_settings_use_granite_r2_1_checkpoint() -> None:
    """R2.1 catalog model must load the r2.1 Hugging Face repo, not r2."""
    raw = load_model_settings_yaml("tiny_time_mixer_r2_1")
    assert raw.get("hf_model_name") == "ibm-granite/granite-timeseries-ttm-r2.1"
