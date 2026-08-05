"""
Unit tests for ConfigManager class.

This test suite provides 100% coverage of the ConfigManager functionality including:
- Initialization and validation
- Benchmark config validation
- Settings validation
- Model settings validation
- Task config validation
- Helper methods
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from tempus_bench.utils.config_manager import ConfigManager, ValidationError
from tempus_bench.utils.configs import (
    TaskConfig,
    EvaluationSetting,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
)
from tempus_bench.utils.paths import get_available_models, find_task_directories


def _mock_task_manager(*, task_path: str = "*") -> Mock:
    manager = Mock()
    manager.task_path = task_path
    manager.evaluation_config = EvaluationConfig(task_path=task_path, max_windows=1)
    manager.logger = Mock()
    return manager


def _raw_task(**overrides):
    base = {
        "task_name": "test_task",
        "task_description": "demo",
        "task_catalog": "application",
        "dataset_category": "commerce_and_trade",
        "dataset_name": "test_task",
        "context_window": 50,
        "forecast_horizon": 24,
        "handle_missing": "interpolate",
        "normalization_method": "standard",
        "target_variable_names": ["series_a"],
        "covariate_variable_names": [],
    }
    base.update(overrides)
    return base


def _write_catalog_yaml(tasks_dir: Path, category: str, tasks: list[dict]) -> None:
    catalog = tasks_dir / "Application Tasks"
    catalog.mkdir(parents=True, exist_ok=True)
    body = "\n---\n".join(yaml.dump({"task": t}, sort_keys=False) for t in tasks)
    (catalog / f"{category}.yaml").write_text(body, encoding="utf-8")


@pytest.fixture
def sample_benchmark_config():
    """Sample benchmark configuration."""
    return {
        "evaluation": {
            "task_path": "*",
            "tuning_loss": "mae",
            "max_windows": 5,
            "max_num_variates": 10,
            "num_samples": 100,
            "num_quantiles": 10,
            "point_forecast_statistic": "mean",
        },
        "model": {
            "arima": {"p": [1, 2], "d": [1], "q": [1, 2]},
            "exponential_smoothing": {"trend": ["add"], "seasonal": ["null"]},
        },
    }


@pytest.fixture
def sample_settings_config():
    """Sample settings configuration."""
    return {
        "logging_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "console_logging": True,
        "tensorboard_logging": True,
        "runs_dir": "runs",
        "conda_env_prefix": "benchmark",
    }


class TestValidationError:
    """Test suite for ValidationError exception."""

    def test_exception_initialization(self):
        """Test that ValidationError can be initialized."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"


class TestManagerLoadConfig:
    """Test suite for _load_config method."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {"key": "value"}
        config_file.write_text(yaml.dump(config_data))

        result = ConfigManager._load_config(str(config_file))
        assert result == config_data

    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigManager._load_config("nonexistent.yaml")

    def test_load_config_empty_file(self, tmp_path):
        """Test that ValueError is raised for empty file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(
            ValueError, match="Configuration file is empty or invalid YAML"
        ):
            ConfigManager._load_config(str(config_file))

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test that ValueError is raised for invalid YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: [yaml: content")

        with pytest.raises(ValueError, match="Invalid YAML format"):
            ConfigManager._load_config(str(config_file))


class TestManagerGetAvailableModels:
    """Test suite for get_available_models utility function."""

    def test_get_available_models(self, tmp_path):
        """Test getting available models from flat models directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        arima_dir = models_dir / "arima"
        arima_dir.mkdir()
        (arima_dir / "arima_model.py").write_text("# model file")
        (arima_dir / "settings.yaml").write_text("python_version: '3.11'\n")

        deepar_dir = models_dir / "deepar"
        deepar_dir.mkdir()
        (deepar_dir / "deepar_model.py").write_text("# model file")
        (deepar_dir / "settings.yaml").write_text("python_version: '3.11'\n")

        with patch("tempus_bench.utils.paths.get_models_dir", return_value=models_dir):
            available_models = get_available_models()
            assert "arima" in available_models
            assert "deepar" in available_models
            assert len(available_models) == 2

    def test_get_available_models_no_models(self, tmp_path):
        """Test getting available models with empty directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with patch("tempus_bench.utils.paths.get_models_dir", return_value=models_dir):
            available_models = get_available_models()
            assert available_models == set()


class TestManagerConvertPydanticErrors:
    """Test suite for convert_pydantic_errors helper."""

    def test_convert_single_error(self):
        from tempus_bench.utils.configs import convert_pydantic_errors

        error = Mock()
        error.errors = Mock(return_value=[{"loc": ("field",), "msg": "error message"}])

        result = convert_pydantic_errors(error)
        assert "field" in result
        assert "error message" in result

    def test_convert_multiple_errors(self):
        from tempus_bench.utils.configs import convert_pydantic_errors

        error = Mock()
        error.errors = Mock(
            return_value=[
                {"loc": ("field1",), "msg": "error 1"},
                {"loc": ("field2",), "msg": "error 2"},
            ]
        )

        result = convert_pydantic_errors(error)
        assert "field1" in result
        assert "field2" in result
        assert "error 1" in result
        assert "error 2" in result


class TestFindTaskDirectories:
    """Test suite for find_task_directories / catalog discovery."""

    def test_find_all_task_directories(self, tmp_path):
        """Test finding all tasks with '*' pattern."""
        tasks_dir = tmp_path / "Tasks"
        _write_catalog_yaml(
            tasks_dir,
            "commerce_and_trade",
            [
                _raw_task(task_name="task1", dataset_name="task1"),
                _raw_task(task_name="task2", dataset_name="task2"),
            ],
        )

        with patch("tempus_bench.utils.paths.get_tasks_dir", return_value=tasks_dir):
            result = find_task_directories("*")
            assert len(result) == 2
            assert "task1" in result
            assert "task2" in result

    def test_find_specific_subdirectory(self, tmp_path):
        """Test finding tasks in a specific dataset category."""
        tasks_dir = tmp_path / "Tasks"
        _write_catalog_yaml(
            tasks_dir,
            "commerce_and_trade",
            [_raw_task(task_name="task1", dataset_name="task1")],
        )
        _write_catalog_yaml(
            tasks_dir,
            "manufacturing_and_industrial_production",
            [
                _raw_task(
                    task_name="task2",
                    dataset_category="manufacturing_and_industrial_production",
                    dataset_name="task2",
                )
            ],
        )

        with patch("tempus_bench.utils.paths.get_tasks_dir", return_value=tasks_dir):
            result = find_task_directories("commerce_and_trade/*")
            assert len(result) == 1
            assert "task1" in result

    def test_find_specific_task_directory(self, tmp_path):
        """Test finding a specific task by category + human name."""
        tasks_dir = tmp_path / "Tasks"
        _write_catalog_yaml(
            tasks_dir,
            "commerce_and_trade",
            [_raw_task(task_name="specific_task", dataset_name="specific_task")],
        )

        with patch("tempus_bench.utils.paths.get_tasks_dir", return_value=tasks_dir):
            result = find_task_directories("commerce_and_trade/specific_task")
            assert len(result) == 1
            assert "specific_task" in result

    def test_find_nonexistent_subdirectory(self, tmp_path):
        """Test finding tasks when category has no matches."""
        tasks_dir = tmp_path / "Tasks"
        tasks_dir.mkdir()
        (tasks_dir / "Application Tasks").mkdir()

        with patch("tempus_bench.utils.paths.get_tasks_dir", return_value=tasks_dir):
            result = find_task_directories("nonexistent/*")
            assert len(result) == 0

    def test_find_nonexistent_specific_task(self, tmp_path):
        """Test finding a specific task that doesn't exist."""
        tasks_dir = tmp_path / "Tasks"
        tasks_dir.mkdir()
        (tasks_dir / "Application Tasks").mkdir()

        with patch("tempus_bench.utils.paths.get_tasks_dir", return_value=tasks_dir):
            result = find_task_directories("commerce_and_trade/nonexistent_task")
            assert len(result) == 0


class TestManagerValidateModelSettings:
    """Test suite for validate_model_settings method."""

    def _manager(self, models=("arima",)):
        manager = Mock()
        manager.logger = Mock()
        manager.models_evaluated = list(models)
        manager.model_configs = {
            name: ModelConfig(model_name=name) for name in models
        }
        manager._load_config = lambda path: ConfigManager._load_config(path)
        return manager

    def _caps_yaml(self) -> str:
        return yaml.dump(
            {
                "python_version": "3.11",
                "device": "cpu",
                "capabilities": {
                    "covariates": "none",
                    "univariate": True,
                    "multivariate": False,
                },
            }
        )

    def test_validate_models_directory_not_found(self):
        """Missing settings for evaluated models raises ValidationError."""
        manager = self._manager()

        with patch(
            "tempus_bench.utils.config_manager.get_models_dir",
            return_value=Path("nonexistent"),
        ):
            with pytest.raises(ValidationError, match="Missing or invalid settings"):
                ConfigManager.init_model_setting(manager)

    def test_validate_model_settings_success(self, tmp_path):
        """Test successful validation of model settings."""
        models_dir = tmp_path / "models" / "arima"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "settings.yaml").write_text(self._caps_yaml())

        manager = self._manager()

        with patch(
            "tempus_bench.utils.config_manager.get_models_dir",
            return_value=models_dir.parent,
        ):
            result = ConfigManager.init_model_setting(manager)
            assert "arima" in result
            assert result["arima"]["python_version"] == "3.11"
            assert result["arima"]["device"] == "cpu"

    def test_validate_model_settings_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        models_dir = tmp_path / "models" / "arima"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "settings.yaml").write_text("invalid: [yaml")

        manager = self._manager()

        with patch(
            "tempus_bench.utils.config_manager.get_models_dir",
            return_value=models_dir.parent,
        ):
            with pytest.raises(ValueError):
                ConfigManager.init_model_setting(manager)

    def test_validate_model_settings_filters_by_main_model(self, tmp_path):
        """Test that only models in main.model are validated."""
        models_dir = tmp_path / "models"
        arima_dir = models_dir / "arima"
        arima_dir.mkdir(parents=True, exist_ok=True)
        (arima_dir / "settings.yaml").write_text(self._caps_yaml())

        prophet_dir = models_dir / "prophet"
        prophet_dir.mkdir()
        (prophet_dir / "settings.yaml").write_text(self._caps_yaml())

        manager = self._manager(models=("arima",))

        with patch(
            "tempus_bench.utils.config_manager.get_models_dir", return_value=models_dir
        ):
            result = ConfigManager.init_model_setting(manager)
            assert "arima" in result
            assert "prophet" not in result


class TestManagerValidateTaskConfigs:
    """Test suite for init_tasks with catalog documents."""

    def test_validate_task_configs_success(self):
        """Test successful loading of task configs from documents."""
        manager = _mock_task_manager()

        with patch(
            "tempus_bench.utils.task_assets.ensure_dataset_assets"
        ), patch(
            "tempus_bench.utils.paths.find_task_documents",
            return_value={"test_task": _raw_task()},
        ):
            result = ConfigManager.init_tasks(manager)
            assert "test_task" in result
            assert result["test_task"].task_name == "test_task"
            assert result["test_task"].forecast_horizon == 24
            assert result["test_task"].dataset_category == "commerce_and_trade"

    def test_validate_task_configs_validation_error(self):
        """Test that invalid task documents raise during build."""
        manager = _mock_task_manager()
        bad = _raw_task()
        del bad["context_window"]

        with patch(
            "tempus_bench.utils.task_assets.ensure_dataset_assets"
        ), patch(
            "tempus_bench.utils.paths.find_task_documents",
            return_value={"test_task": bad},
        ):
            with pytest.raises((ValidationError, KeyError, TypeError, ValueError)):
                ConfigManager.init_tasks(manager)

    def test_validate_task_configs_empty_documents(self):
        """Test that empty discovery result yields empty task map."""
        manager = _mock_task_manager()

        with patch(
            "tempus_bench.utils.task_assets.ensure_dataset_assets"
        ), patch(
            "tempus_bench.utils.paths.find_task_documents",
            return_value={},
        ):
            result = ConfigManager.init_tasks(manager)
            assert result == {}


class TestManagerFullIntegration:
    """Test suite for full ConfigManager initialization."""

    @patch("tempus_bench.utils.config_manager.LogManager")
    @patch("tempus_bench.utils.config_manager.get_project_root")
    @patch("tempus_bench.utils.config_manager.get_models_dir")
    @patch("tempus_bench.utils.paths.get_tasks_dir")
    @patch("tempus_bench.utils.task_assets.ensure_dataset_assets")
    def test_config_manager_full_initialization(
        self,
        mock_ensure_assets,
        mock_tasks_dir,
        mock_models_dir,
        mock_get_project_root,
        mock_logger,
        tmp_path,
    ):
        """Test full ConfigManager initialization with Tasks/ catalog layout."""
        project_root = tmp_path
        tempus_bench_dir = project_root / "tempus_bench"
        tempus_bench_dir.mkdir()
        config_dir = tempus_bench_dir / "config"
        config_dir.mkdir()

        config_file = project_root / "benchmark.yaml"
        config_data = {
            "evaluation": {
                "task_path": "commerce_and_trade/*",
                "tuning_loss": "mae",
                "max_windows": 5,
                "max_num_variates": 10,
            },
            "model": {"arima": {"p": [1, 2]}},
        }
        config_file.write_text(yaml.dump(config_data))

        settings_file = config_dir / "settings.yaml"
        settings_data = {
            "file_logging": True,
            "console_logging": True,
            "tensorboard_logging": True,
            "conda_env_prefix": "benchmark",
            "reinstall_conda": False,
            "verbose": False,
        }
        settings_file.write_text(yaml.dump(settings_data))

        models_dir = tmp_path / "models"
        arima_dir = models_dir / "arima"
        arima_dir.mkdir(parents=True)
        (arima_dir / "settings.yaml").write_text(
            yaml.dump(
                {
                    "python_version": "3.11",
                    "device": "cpu",
                    "capabilities": {
                        "covariates": "none",
                        "univariate": True,
                        "multivariate": False,
                    },
                }
            )
        )
        (arima_dir / "arima_model.py").write_text("# model file")

        tasks_dir = project_root / "Tasks"
        _write_catalog_yaml(
            tasks_dir,
            "commerce_and_trade",
            [_raw_task()],
        )

        mock_get_project_root.return_value = project_root
        mock_models_dir.return_value = models_dir
        mock_tasks_dir.return_value = tasks_dir
        mock_logger.return_value = Mock()

        manager = ConfigManager(str(config_file))
        assert manager.config_path == str(config_file)
        assert isinstance(manager.model_configs, dict)
        assert isinstance(manager.evaluation_setting, EvaluationSetting)
        assert "test_task" in manager.task_configs


class TestManagerTaskConfigBranch:
    """Test suite for task config validation branch."""

    def test_task_config_rejects_singular_target(self):
        """Singular target_variable_name is rejected by the loader."""
        manager = _mock_task_manager()
        bad = _raw_task()
        bad["target_variable_name"] = "y"
        del bad["target_variable_names"]

        with patch(
            "tempus_bench.utils.task_assets.ensure_dataset_assets"
        ), patch(
            "tempus_bench.utils.paths.find_task_documents",
            return_value={"test_task": bad},
        ):
            with pytest.raises(ValueError, match="target_variable_name"):
                ConfigManager.init_tasks(manager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
