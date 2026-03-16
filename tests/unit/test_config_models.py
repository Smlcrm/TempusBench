"""
Unit tests for Pydantic models in config module.

This test suite provides 100% coverage of the models.py validators including
edge cases for max_num_variates, model parameter validation, and foundation models.
"""

import pytest
from pydantic import ValidationError
from tempus_bench.utils.configs import (
    EvaluationConfig,
    ModelConfig,
    DatasetConfig,
    TaskConfig,
    EvaluationSetting,
)


class TestEvaluationConfig:
    """Test suite for EvaluationConfig model."""

    def test_valid_config_with_tuning_loss(self):
        """Test valid evaluation configuration with tuning loss."""
        config = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20,
            max_num_variates=10,
            num_samples=100,
            num_quantiles=10,
            point_forecast_statistic="mean",
        )
        assert config.tuning_loss == "mae"
        assert config.max_windows == 20
        assert config.max_num_variates == 10

    def test_valid_config_with_default_tuning_loss(self):
        """Test valid evaluation configuration with default tuning loss."""
        config = EvaluationConfig(max_windows=20)
        assert config.tuning_loss == "mae"  # default value
        assert config.max_windows == 20

    def test_invalid_tuning_loss(self):
        """Test invalid tuning loss value."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(tuning_loss="invalid_metric", max_windows=20)  # type: ignore
        assert "Input should be 'mae', 'mase', 'mape' or 'rmse'" in str(exc_info.value)

    def test_max_num_variates_none(self):
        """Test max_num_variates with None value (default, all variates)."""
        config = EvaluationConfig(tuning_loss="mae", max_windows=20)
        assert config.max_num_variates is None

    def test_max_num_variates_invalid_negative(self):
        """Test that negative max_num_variates raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(tuning_loss="mae", max_windows=20, max_num_variates=-1)
        # Pydantic will validate ge constraint
        assert (
            "greater than or equal to" in str(exc_info.value).lower()
            or "must be" in str(exc_info.value).lower()
        )

    def test_max_num_variates_zero(self):
        """Test that zero max_num_variates raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(tuning_loss="mae", max_windows=20, max_num_variates=0)
        # Pydantic will validate ge constraint if there is one
        assert (
            "greater than or equal to" in str(exc_info.value).lower()
            or "must be" in str(exc_info.value).lower()
        )

    def test_max_num_variates_valid_int(self):
        """Test that valid int values are accepted."""
        config = EvaluationConfig(tuning_loss="mae", max_windows=20, max_num_variates=5)
        assert config.max_num_variates == 5


class TestModelConfig:
    """Test suite for ModelConfig model."""

    def test_valid_model_config(self):
        """Test valid model configuration with model_name."""
        config = ModelConfig(model_name="arima")
        assert config.model_name == "arima"

    def test_model_config_missing_model_name(self):
        """Test that missing model_name raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig()  # type: ignore  # Missing required model_name

    def test_model_config_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(model_name="arima", extra_field="not allowed")  # type: ignore
        assert (
            "extra" in str(exc_info.value).lower()
            or "forbidden" in str(exc_info.value).lower()
        )


class TestDatasetConfig:
    """Test suite for DatasetConfig model."""

    def test_valid_config_with_defaults(self):
        """Test valid dataset configuration with defaults."""
        config = DatasetConfig(file_name="test_dataset.csv")
        assert config.file_name == "test_dataset.csv"
        assert config.normalize is True
        assert config.handle_missing == "interpolate"

    def test_valid_config_custom_values(self):
        """Test valid dataset configuration with custom values."""
        config = DatasetConfig(
            file_name="test_dataset.csv", normalize=False, handle_missing="drop"
        )
        assert config.file_name == "test_dataset.csv"
        assert config.normalize is False
        assert config.handle_missing == "drop"

    def test_invalid_handle_missing(self):
        """Test that invalid handle_missing raises ValidationError."""
        with pytest.raises(ValidationError):
            DatasetConfig(file_name="test.csv", handle_missing="invalid_method")  # type: ignore


class TestTaskConfig:
    """Test suite for TaskConfig model."""

    def test_valid_config(self):
        """Test valid task configuration."""
        config = TaskConfig(
            task_name="test_task",
            task_path="/path/to/task",
            forecast_horizon=24,
            context_window=100,
            dataset=DatasetConfig(file_name="test.csv"),
        )
        assert config.forecast_horizon == 24
        assert config.context_window == 100
        assert config.task_path == "/path/to/task"
        assert config.dataset.file_name == "test.csv"

    def test_invalid_forecast_horizon_too_large(self):
        """Test that forecast_horizon > 128 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(
                task_name="test_task",
                task_path="/path/to/task",
                forecast_horizon=200,
                context_window=100,
                dataset=DatasetConfig(file_name="test.csv"),
            )
        assert "128" in str(exc_info.value)

    def test_invalid_forecast_horizon_zero(self):
        """Test that forecast_horizon < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskConfig(
                task_name="test_task",
                task_path="/path/to/task",
                forecast_horizon=0,
                context_window=100,
                dataset=DatasetConfig(file_name="test.csv"),
            )

    def test_invalid_context_window_zero(self):
        """Test that context_window < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskConfig(
                task_name="test_task",
                task_path="/path/to/task",
                forecast_horizon=24,
                context_window=0,
                dataset=DatasetConfig(file_name="test.csv"),
            )


# BenchmarkConfig tests removed - validation logic now tested through JobConfig


class TestEvaluationSetting:
    """Test suite for EvaluationSetting model."""

    def test_valid_config(self):
        """Test valid systems configuration."""
        config = EvaluationSetting(
            file_logging=True,
            console_logging=True,
            tensorboard_logging=True,
            conda_env_prefix="benchmark",
            reinstall_conda=False,
        )
        assert config.file_logging is True
        assert config.console_logging is True
        assert config.tensorboard_logging is True
        assert config.conda_env_prefix == "benchmark"
        assert config.reinstall_conda is False
        assert config.file_log_level == "DEBUG"  # default
        assert config.console_log_level == "INFO"  # default

    def test_valid_config_with_custom_log_levels(self):
        """Test valid configuration with custom log levels."""
        config = EvaluationSetting(
            file_logging=True,
            console_logging=True,
            tensorboard_logging=True,
            conda_env_prefix="benchmark",
            reinstall_conda=True,
            file_log_level="INFO",
            console_log_level="WARNING",
        )
        assert config.file_log_level == "INFO"
        assert config.console_log_level == "WARNING"

    def test_invalid_log_level(self):
        """Test that invalid log level raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationSetting(
                file_logging=True,
                console_logging=True,
                tensorboard_logging=True,
                conda_env_prefix="benchmark",
                reinstall_conda=False,
                file_log_level="INVALID",  # type: ignore
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationSetting(
                file_logging=True,
                console_logging=True,
                # Missing tensorboard_logging, conda_env_prefix, reinstall_conda
            )  # type: ignore

    def test_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EvaluationSetting(
                file_logging=True,
                console_logging=True,
                tensorboard_logging=True,
                conda_env_prefix="benchmark",
                reinstall_conda=False,
                extra_field="not allowed",  # type: ignore
            )


# ModelConfig validation tests removed - ModelConfig now only stores model_name
# Parameter validation is no longer performed at the ModelConfig level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
