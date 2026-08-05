"""Shared helpers for constructing TaskConfig objects in tests."""

from __future__ import annotations

from tempus_bench.utils.configs import TaskConfig


def dummy_covariate_task_config(**overrides) -> TaskConfig:
    """Minimal covariate TaskConfig for preprocessor / tuner unit tests."""
    fields = {
        "task_name": "covariate_test",
        "task_path": "commerce_and_trade/covariate_test",
        "dataset_category": "commerce_and_trade",
        "dataset_name": "covariate_test",
        "forecast_horizon": 1,
        "context_window": 1,
        "handle_missing": "interpolate",
        "normalization_method": "standard",
        "file_name": "covariate_test.csv",
        "task_mode": "covariate",
        "target_variable_names": ["target"],
        "covariate_variable_names": ["cov1", "cov2"],
    }
    fields.update(overrides)
    return TaskConfig(**fields)


def dummy_univariate_task_config(**overrides) -> TaskConfig:
    """Minimal univariate TaskConfig for preprocessor / tuner unit tests."""
    fields = {
        "task_name": "norm_test",
        "task_path": "commerce_and_trade/norm_test",
        "dataset_category": "commerce_and_trade",
        "dataset_name": "norm_test",
        "forecast_horizon": 1,
        "context_window": 1,
        "handle_missing": "interpolate",
        "normalization_method": "standard",
        "file_name": "dummy.csv",
        "task_mode": "univariate",
        "target_variable_names": ["target"],
        "covariate_variable_names": [],
    }
    fields.update(overrides)
    return TaskConfig(**fields)
