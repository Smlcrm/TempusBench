"""DataLoader building datasets from generators instead of CSV files."""

import tempfile

import numpy as np
import pytest

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import EvaluationConfig, TaskConfig
from tempus_bench.utils.log_manager import LogManager


@pytest.fixture(autouse=True)
def _log_manager():
    """Preprocessor logs through the LogManager singleton."""
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


def _task(name="random_walk", mode="univariate", targets=("y",), covariates=()):
    return TaskConfig(
        task_name=f"Test {name}",
        task_path="/nonexistent",
        context_window=64,
        forecast_horizon=16,
        handle_missing="interpolate",
        normalization_method="none",
        task_mode=mode,
        task_catalog="synthetic",
        dataset_category="trend",
        dataset_name=name,
        target_type="continuous_real",
        series_length=256,
        target_variable_names=list(targets),
        covariate_variable_names=list(covariates),
    )


def _evaluation():
    return EvaluationConfig(task_path="Synthetic Tasks/Trend")


def test_generates_dataset_without_touching_disk():
    loader = DataLoader(_task(), _evaluation(), base_seed=11)
    dataset = loader.dataset

    assert len(dataset.target) == 256
    assert len(dataset.target[0]) == 1
    assert dataset.covariate is None
    assert dataset.metadata["num_targets"] == 1
    assert dataset.metadata["target_variable_units"] == ["unitless"]
    assert dataset.metadata["time_freq"] == "h"
    assert dataset.metadata["generator_name"] == "random_walk"


def test_same_base_seed_reproduces_the_series():
    a = DataLoader(_task(), _evaluation(), base_seed=11).dataset.target
    b = DataLoader(_task(), _evaluation(), base_seed=11).dataset.target
    c = DataLoader(_task(), _evaluation(), base_seed=12).dataset.target
    assert np.array_equal(np.array(a), np.array(b))
    assert not np.array_equal(np.array(a), np.array(c))


def test_metadata_records_the_derived_generator_seed():
    from tempus_bench import generators

    dataset = DataLoader(_task(), _evaluation(), base_seed=5).dataset
    assert dataset.metadata["base_seed"] == 5
    assert dataset.metadata["generator_seed"] == generators.resolve_seed(5, "random_walk")


def test_covariate_generator_splits_target_and_covariate_columns():
    task = _task(
        name="covariate_nonlinear", mode="covariate", targets=("y",), covariates=("x",)
    )
    dataset = DataLoader(task, _evaluation(), base_seed=3).dataset

    assert dataset.metadata["num_targets"] == 1
    assert dataset.metadata["num_covariates"] == 1
    assert len(dataset.covariate) == 256


def test_column_count_mismatch_is_reported():
    # mv_var returns 2 columns; declaring 1 target and no covariates is wrong.
    task = _task(name="mv_var", mode="multivariate", targets=("y1",))
    with pytest.raises(ValueError, match="returned .* column"):
        DataLoader(task, _evaluation(), base_seed=0)


def test_multivariate_generator_maps_all_columns_to_targets():
    task = _task(name="mv_var", mode="multivariate", targets=("y1", "y2"))
    dataset = DataLoader(task, _evaluation(), base_seed=0).dataset
    assert dataset.metadata["num_targets"] == 2
    assert dataset.metadata["num_covariates"] == 0
