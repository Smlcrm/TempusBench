"""Preprocessor maps GluonTS-style unknown start/frequency to a synthetic 1 Hz grid."""

import os

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from tempus_bench.pipeline.preprocessor import Preprocessor
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


def test_clean_maps_unknown_start_and_freq():
    tc = TaskConfig(
        task_name="synthetic_nonstationary_univariate",
        task_path="synthetic/synthetic_nonstationary_univariate",
        dataset_category="synthetic",
        dataset_name="synthetic_nonstationary_univariate",
        forecast_horizon=24,
        context_window=50,
        handle_missing="interpolate",
        normalization_method="standard",
        file_name="synthetic_nonstationary_univariate.csv",
        task_mode="univariate",
        target_variable_names=["target"],
        covariate_variable_names=[],
    )
    ev = EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=2,
        max_num_variates=10,
        num_samples=10,
        num_quantiles=10,
        point_forecast_statistic="mean",
    )
    prep = Preprocessor(tc, ev)
    num_steps = 3000
    target_raw = "[" + ",".join(["0.0"] * num_steps) + "]"

    ts, t0, fq, tgt, _scaler = prep.clean(
        "unknown",
        "unknown",
        target_raw,
        normalize=True,
        handle_missing="interpolate",
    )

    assert fq == "s"
    assert t0 == "2000-01-01 00:00:00"
    assert len(ts) == tgt.shape[0] == num_steps
    assert tgt.shape[1] == 1
