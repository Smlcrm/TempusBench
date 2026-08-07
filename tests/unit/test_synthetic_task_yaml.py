"""Building TaskConfig objects from generator catalog documents."""

import pytest

from tempus_bench.utils.task_yaml_loader import build_task_config_from_raw


def _raw(**overrides):
    raw = {
        "task_name": "Irregular Period Seasonal",
        "task_description": "Drifting period sinusoid.",
        "context_window": 512,
        "forecast_horizon": 64,
        "handle_missing": "interpolate",
        "normalization_method": "standard",
        "task_catalog": "synthetic",
        "dataset_category": "seasonality",
        "dataset_name": "irregular_period_seasonal",
        "target_type": "continuous_real",
        "series_length": 2048,
        "target_variable_names": ["y"],
        "covariate_variable_names": [],
    }
    raw.update(overrides)
    return raw


def test_generator_task_carries_no_file_name():
    config = build_task_config_from_raw(_raw())
    assert config.is_synthetic()
    assert config.file_name is None
    assert config.dataset_name == "irregular_period_seasonal"
    assert config.series_length == 2048
    assert config.target_type == "continuous_real"
    assert config.generator_params == {}


def test_application_task_still_derives_its_csv_name():
    config = build_task_config_from_raw(
        _raw(task_catalog="application", dataset_name="Daily_Corn_Futures")
    )
    assert not config.is_synthetic()
    assert config.file_name == "Daily_Corn_Futures.csv"


def test_mode_is_inferred_from_the_variable_lists():
    assert build_task_config_from_raw(_raw()).task_mode == "univariate"
    assert (
        build_task_config_from_raw(
            _raw(target_variable_names=["y1", "y2"])
        ).task_mode
        == "multivariate"
    )
    assert (
        build_task_config_from_raw(
            _raw(covariate_variable_names=["x"])
        ).task_mode
        == "covariate"
    )


def test_generator_params_are_passed_through():
    config = build_task_config_from_raw(_raw(generator_params={"sigma": 2.0}))
    assert config.generator_params == {"sigma": 2.0}


def test_generator_task_has_no_dataset_adapter():
    config = build_task_config_from_raw(_raw())
    with pytest.raises(AttributeError, match="no dataset file"):
        _ = config.dataset
