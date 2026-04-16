"""Preprocessor maps GluonTS-style unknown start/frequency to a synthetic 1 Hz grid."""

import os

os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

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
    task_yaml = (
        Path(PROJECT_ROOT)
        / "tempus_bench"
        / "tasks"
        / "univariate"
        / "synthetic_nonstationary_univariate"
        / "task.yaml"
    )
    doc = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    t = doc["task"]
    tc = TaskConfig(**t, task_path=str(task_yaml.parent.resolve()))
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
