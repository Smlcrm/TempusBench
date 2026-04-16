import os
import sys
import tempfile

import numpy as np
import pytest

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

# Ensure local workspace package is imported (not a globally installed version)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from tempus_bench.pipeline.preprocessor import Preprocessor
from tempus_bench.utils.configs import DatasetConfig, EvaluationConfig, TaskConfig
from tempus_bench.utils.log_manager import LogManager


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


def _dummy_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="norm_test",
        task_path="/tmp",
        forecast_horizon=1,
        context_window=1,
        dataset=DatasetConfig(file_name="dummy.csv"),
    )


def _evaluation_config_for_norm_tests() -> EvaluationConfig:
    return EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=5,
        max_num_variates=10,
        num_samples=100,
        num_quantiles=10,
        point_forecast_statistic="mean",
    )


def test_preprocessor_normalizes_univariate_when_enabled():
    pre = Preprocessor(_dummy_task_config(), _evaluation_config_for_norm_tests())

    target_raw = str([1.0, 2.0, 3.0, 4.0, 5.0])
    time_start = "2020-01-01"
    freq = "D"
    normalize = True
    handle_missing = "interpolate"

    _, _, _, target, _scaler = pre.clean(
        time_start=time_start,
        freq=freq,
        target_raw=target_raw,
        normalize=normalize,
        handle_missing=handle_missing,
    )

    assert target.shape == (5, 1)

    col = target[:, 0]
    mean = float(np.mean(col))
    std = float(np.std(col, ddof=0))

    assert abs(mean) < 1e-7
    assert abs(std - 1.0) < 1e-6


def test_preprocessor_normalizes_multivariate_when_enabled():
    pre = Preprocessor(_dummy_task_config(), _evaluation_config_for_norm_tests())

    raw_features = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
    target_raw = str(raw_features)
    time_start = "2020-01-01"
    freq = "D"
    normalize = True
    handle_missing = "interpolate"

    _, _, _, target, scaler = pre.clean(
        time_start=time_start,
        freq=freq,
        target_raw=target_raw,
        normalize=normalize,
        handle_missing=handle_missing,
    )

    assert target.shape == (5, 2)

    for j in range(target.shape[1]):
        col = target[:, j]
        mean = float(np.mean(col))
        std = float(np.std(col, ddof=0))
        assert abs(mean) < 1e-7
        assert abs(std - 1.0) < 1e-6


def test_preprocessor_no_normalization_when_disabled():
    pre = Preprocessor(_dummy_task_config(), _evaluation_config_for_norm_tests())

    target_raw = str([2.0, 4.0, 6.0, 8.0])
    time_start = "2020-01-01"
    freq = "D"
    normalize = False
    handle_missing = "interpolate"

    _, _, _, target, scaler = pre.clean(
        time_start=time_start,
        freq=freq,
        target_raw=target_raw,
        normalize=normalize,
        handle_missing=handle_missing,
    )

    assert target.shape == (4, 1)

    col = target[:, 0]
    mean = float(np.mean(col))
    std = float(np.std(col, ddof=0))

    assert abs(mean) > 1e-3
    assert abs(std - 1.0) > 1e-3


