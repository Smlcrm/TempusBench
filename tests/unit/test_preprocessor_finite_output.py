"""Preprocessor must emit only finite floats (JSON + sklearn safe; no NaN/Inf tokens in UI JSON)."""

import os
import sys
import tempfile

import types

import numpy as np
import pytest

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from tempus_bench.pipeline.preprocessor import Preprocessor
from tests.helpers.task_config_fixtures import dummy_univariate_task_config
from tempus_bench.utils.configs import EvaluationConfig, TaskConfig
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


def _task_config() -> TaskConfig:
    return dummy_univariate_task_config(task_name="finite_test")


def _eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=4,
        max_num_variates=None,
        num_samples=1,
        num_quantiles=1,
        point_forecast_statistic="mean",
    )


def test_clean_output_all_finite_after_interpolate_with_interior_nan():
    pre = Preprocessor(_task_config(), _eval_config())
    target_raw = str([1.0, float("nan"), 3.0, 4.0])
    _, _, _, target, _ = pre.clean(
        time_start="2020-01-01",
        freq="D",
        target_raw=target_raw,
        normalize=False,
        handle_missing="interpolate",
    )
    assert np.isfinite(target).all()


def test_clean_maps_inf_to_finite_via_interpolate():
    pre = Preprocessor(_task_config(), _eval_config())
    target_raw = str([1.0, float("inf"), 4.0, 7.0])
    _, _, _, target, _ = pre.clean(
        time_start="2020-01-01",
        freq="D",
        target_raw=target_raw,
        normalize=False,
        handle_missing="interpolate",
    )
    assert np.isfinite(target).all()


def test_mean_with_all_nan_column_becomes_finite():
    pre = Preprocessor(_task_config(), _eval_config())
    # Two variates × three steps: second series all NaN → mean imputation skips it; coerce fixes.
    target_raw = str(
        [
            [1.0, 2.0, 3.0],
            [float("nan"), float("nan"), float("nan")],
        ]
    )
    _, _, _, target, _ = pre.clean(
        time_start="2020-01-01",
        freq="D",
        target_raw=target_raw,
        normalize=False,
        handle_missing="mean",
    )
    assert target.shape == (3, 2)
    assert np.isfinite(target).all()


def test_coerce_finite_per_column_raises_when_column_cannot_be_repaired():
    pre = Preprocessor(_task_config(), _eval_config())

    def _return_all_nan(_self, col_data: np.ndarray) -> np.ndarray:
        return np.full_like(col_data, np.nan, dtype=np.float64)

    pre._interpolate_column = types.MethodType(_return_all_nan, pre)  # type: ignore[method-assign]

    dirty = np.array([[1.0], [float("nan")]], dtype=np.float64)
    with pytest.raises(ValueError, match="cannot eliminate non-finite"):
        pre._coerce_finite_per_column(dirty, "interpolate")
