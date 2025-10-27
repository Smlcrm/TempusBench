"""
Configuration manager for the benchmarking pipeline.

This module provides comprehensive validation and management of configuration files using Pydantic
to ensure they comply with the expected schema before execution. The ConfigManager class handles
validation of benchmark configurations, model settings, task configurations, and system settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from pydantic import ValidationError

from tempus_bench.utils.paths import get_dataset_path, get_models_dir, get_task_path, get_tasks_dir, get_configs_dir
from tempus_bench.utils.logger import get_logger
from .models import BenchmarkConfig, TaskConfig, ModelSettingsConfig, SystemsConfig, UnifiedConfig

# Global config manager instance
_global_config_manager = None

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class ConfigManager:
    """
    Configuration manager with comprehensive validation rules.

    This class handles the validation and management of all configuration files
    in the benchmarking pipeline, including benchmark configurations, model settings,
    task configurations, and system settings.

    Attributes:
        config_path (str): Path to the main benchmark configuration YAML file
        logger: Logger instance for configuration validation messages
        main (BenchmarkConfig): Validated main benchmark configuration
        settings (SystemsConfig): System settings from config/settings.yaml
        model_settings (Dict[str, ModelSettingsConfig]): Model execution settings (Python version, device, conda env)
        task_dirs (List[Path]): Task directories matching the task_path pattern
        task (Dict[str, List[TaskConfig]]): Validated task configurations (each task can have multiple configs)
        model (Dict[str, Any]): Model hyperparameters from main config
    """

    def __init__(self, config_path: str, logs_path: str):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the main benchmark configuration YAML file
            logs_path: Directory for log files

        Initializes configuration attributes:
            - self.main: Main benchmark configuration with model hyperparameters
            - self.settings: System settings (logging format, etc.)
            - self.model_settings: Model execution settings (Python version, device, conda env)
            - self.task_dirs: Task directories matching task_path pattern
            - self.task: Task configurations (forecast horizon, dataset settings) - supports multiple configs per task
            - self.model: Model hyperparameters extracted from main config
        """
        # Setup paths
        self.config_path = config_path
        self.main = self.validate_benchmark_config()
        self.settings = self.validate_benchmark_settings()
        console_logging = self.settings.console_logging
        file_logging = self.settings.file_logging
        self.logger = get_logger(logs_path, console_logging=console_logging, file_logging=file_logging)
        self.model_settings = self.validate_model_settings()
        self.task = self.validate_task_configs()

    def validate_benchmark_config(self) -> BenchmarkConfig:
        """
        Validate the main benchmark configuration file and return a BenchmarkConfig instance.

        This method loads the configuration file specified in self.config_path, validates it
        against the BenchmarkConfig schema, and checks model availability.

        Returns:
            Validated BenchmarkConfig instance

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Validate using Pydantic model
            config = self._load_config(self.config_path)
            config = BenchmarkConfig(**config)
            self._validate_model_availability(config)
            self.logger.info("ConfigValidator", "Configuration validation passed")
            return config

        except ValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            self.logger.error("ConfigValidator", f"Configuration validation failed: {error_msg}")
            raise ConfigValidationError(error_msg)

    def validate_benchmark_settings(self) -> SystemsConfig:
        """
        Validate the benchmark settings.yaml file.

        This method loads and validates the systems configuration file located at
        config/settings.yaml against the SystemsConfig schema.

        Returns:
            Validated SystemsConfig instance

        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If settings.yaml doesn't exist
        """
        config_dir = get_configs_dir()
        benchmark_settings_dir = config_dir / "settings.yaml"

        if not benchmark_settings_dir.exists():
            raise FileNotFoundError(f"Settings config not found: {benchmark_settings_dir}")

        try:
            with open(benchmark_settings_dir, 'r') as f:
                benchmark_settings_data = yaml.safe_load(f)

            # Validate using SystemsConfig
            benchmark_settings = SystemsConfig(**benchmark_settings_data)
            self.logger.debug("ConfigValidator", f"Systems config validated: {benchmark_settings_dir}")
            return benchmark_settings

        except ValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            self.logger.error("ConfigValidator", f"Systems config validation failed: {error_msg}")
            raise ConfigValidationError(f"Invalid systems config at {benchmark_settings_dir}: {error_msg}")
        except Exception as e:
            raise ConfigValidationError(f"Invalid systems config at {benchmark_settings_dir}: {e}")

    def validate_model_settings(self) -> Dict[str, ModelSettingsConfig]:
        """
        Validate model execution settings (Python version, device, conda environment).

        Returns:
            Dictionary mapping model names to their execution settings

        Raises:
            ConfigValidationError: If validation fails or models directory doesn't exist
        """
        models_dir = get_models_dir()
        if not models_dir.exists():
            raise ConfigValidationError(f"Models directory not found: {models_dir}")

        validated_settings = {}

        # Find all model settings.yaml files recursively in models_dir and use their parent folders as model names
        settings_files = list(models_dir.glob("**/settings.yaml"))
        for model_settings_path in settings_files:
            model_folder = model_settings_path.parent
            model_name = model_folder.name
            if model_name not in self.main.model: continue

            try:
                with open(model_settings_path, 'r') as f:
                    model_settings_data = yaml.safe_load(f)

                # Validate using ModelSettingsConfig
                model_settings = ModelSettingsConfig(**model_settings_data)
                validated_settings[model_name] = model_settings
                self.logger.debug("ConfigValidator", f"Model settings validated: {model_settings_path}")

            except ValidationError as e:
                error_msg = self._convert_pydantic_errors(e)
                raise ConfigValidationError(
                    f"Invalid model settings for '{model_name}' at {model_settings_path}: {error_msg}"
                )
            except Exception as e:
                raise ConfigValidationError(
                    f"Invalid model settings for '{model_name}' at {model_settings_path}: {e}"
                )

        return validated_settings

    def validate_task_configs(self) -> Dict[str, List[TaskConfig]]:
        """
        Validate all task.yaml files for task directories found based on the task_path pattern.

        This method validates task configurations for all task directories that match the pattern specified in the main benchmark configuration. Each task's configuration is validated against the TaskConfig schema. Supports multiple TaskConfigs per task using YAML multi-document format (separated by '---').

        Example task.yaml format:
        ```yaml
        task:
            forecast_horizon: 24
            context_window: 100
            dataset:
            name: dataset.csv
        ---
        task:
            forecast_horizon: 48
            context_window: 100
            dataset:
            name: dataset.csv
        ```

        Returns:
            Dictionary mapping task names to lists of validated TaskConfig instances

        Raises:
            ConfigValidationError: If validation fails
        """
        validated_configs = {}

        for task_dir in self.task_dirs:
            task_config_path = task_dir / "task.yaml"
            if not task_config_path.exists():
                raise ConfigValidationError(f"Task config not found: {task_config_path}")

            task_name = task_dir.name

            try:
                with open(task_config_path, 'r') as f:
                    # Load all YAML documents (supports multi-document format)
                    task_config_documents = list(yaml.safe_load_all(f))

                # Handle multiple task configs defined with '---' separator
                task_configs = []
                for doc in task_config_documents:
                    if not doc: continue  # Skip empty documents

                    # Extract the 'task' key if present
                    if isinstance(doc, dict) and 'task' in doc:
                        task_data = doc['task']
                    else:
                        raise ConfigValidationError(
                            f"All sections in {task_config_path} must contain a 'task' key."
                        )

                    # Validate and create TaskConfig
                    task_config = TaskConfig(**task_data)

                    # Validate that task name matches folder name
                    if task_config.name != task_name:
                        raise ConfigValidationError(
                            f"Task name '{task_config.name}' in {task_config_path} does not match folder name '{task_name}'"
                        )

                    # Validate that CSV file exists and matches task name
                    csv_file = task_dir / f"{task_name}.csv"
                    if not csv_file.exists():
                        raise ConfigValidationError(
                            f"CSV file not found in {task_dir}. Expected: {csv_file.name}"
                        )

                    task_configs.append(task_config)

                if not task_configs:
                    raise ConfigValidationError(f"No valid task configurations found in {task_config_path}")

                validated_configs[task_name] = task_configs
                self.logger.debug("ConfigValidator", f"Task config validated: {task_config_path} ({len(task_configs)} config(s))")

            except ValidationError as e:
                error_msg = self._convert_pydantic_errors(e)
                raise ConfigValidationError(f"Invalid task config at {task_config_path}: {error_msg}")
            except Exception as e:
                raise ConfigValidationError(f"Invalid task config at {task_config_path}: {e}")

        return validated_configs

    def generate_configs(self):
        """
        Yield unified configs for each task and each model. For each generated config,
        a separate benchmark config is constructed with task_path set to the path for that task,
        and the model dict containing only the model_name's hyperparameters.

        Yields:
            UnifiedConfig instances
        """
        from tempus_bench.config.models import BenchmarkConfig  # Avoid circular import if happens

        for task_name, task_configs in self.task.items():
            for task_config in task_configs:
                # Build the updated task_path for this task
                task_path = get_task_path(task_name)
                for model_name, model_params in self.model.items():
                    # Build a new BenchmarkConfig for this (task, model)
                    benchmark_dict = self.main.model_dump()
                    benchmark_dict["task_path"] = task_path
                    benchmark_dict["model"] = {model_name: model_params}
                    benchmark_dict["evaluation"]["metrics"] = 
                    benchmark = BenchmarkConfig(**benchmark_dict)
                    yield UnifiedConfig(
                        benchmark=benchmark,
                        settings=self.settings,
                        model_settings={model_name: self.model_settings[model_name]},
                        task_configs={task_name: [task_config]}
                    )

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file and return as dictionary.

        Args:
            config_path: Path to the configuration YAML file

        Returns:
            Dictionary containing the configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the file is empty or contains invalid YAML
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                raise ValueError("Configuration file is empty or invalid YAML")

            return config_data

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")

    def _validate_model_availability(self, config: BenchmarkConfig) -> None:
        """
        Validate that all specified models are available in the models directory.

        This method checks that each model specified in the benchmark configuration
        has a corresponding model file in the models directory structure.

        Args:
            config: The validated BenchmarkConfig instance

        Raises:
            ConfigValidationError: If any specified model is not available
        """
        # Check each model in the configuration
        available_models = self._get_available_models()
        self.model = config.model.model_dump()
        # Only validate models that are actually specified (not None)
        for model_name, model_params in self.model.items():
            if model_params is not None and model_name not in available_models:
                raise ConfigValidationError(
                    f"Model '{model_name}' is not available. "
                    f"Available models: {sorted(available_models)}"
                )

    def _get_available_models(self) -> set:
        """
        Get all available model names from the models directory.

        This method scans the models directory structure to find all model folders
        that contain a corresponding model file (e.g., arima_model.py).

        Returns:
            Set of available model names
        """
        available_models = set()
        models_dir = get_models_dir()
        # Look for model folders in any subdirectory
        for subdir in models_dir.iterdir():
            if subdir.is_dir(): # deterministic or stochastic
                for model_folder in subdir.iterdir():
                    if model_folder.is_dir(): # Check if it has a model file
                        model_file = model_folder / f"{model_folder.name}_model.py"
                        if model_file.exists():
                            available_models.add(model_folder.name)

        return available_models

    def _find_task_directories(self) -> List[Path]:
        """
        Find task directories based on the task_path pattern from the benchmark configuration.

        This method searches for task directories that match the pattern specified
        in the main benchmark configuration. Supported patterns include:
        - "*": All task directories
        - "univariate/*": All univariate task directories
        - "multivariate/*": All multivariate task directories
        - "specific_task": A specific task directory

        Returns:
            List of Path objects pointing to task directories
        """
        tasks_dir = get_tasks_dir()
        task_dirs = []

        pattern = self.main.task_path

        if pattern == "*":
            # Find all task directories
            for subdir in tasks_dir.iterdir():
                if subdir.is_dir():
                    for task_dir in subdir.iterdir():
                        if task_dir.is_dir():
                            task_dirs.append(task_dir)
        elif pattern.endswith("/*"):
            # Find directories in specific subdirectory
            subdir_name = pattern[:-2]
            subdir_path = tasks_dir / subdir_name
            if subdir_path.exists():
                for task_dir in subdir_path.iterdir():
                    if task_dir.is_dir():
                        task_dirs.append(task_dir)
        else:
            # Specific task directory
            task_path = tasks_dir / pattern
            if task_path.exists():
                task_dirs.append(task_path)

        return task_dirs

    def _convert_pydantic_errors(self, validation_error: ValidationError) -> str:
        """
        Convert Pydantic validation errors to a readable string format.

        This method takes a Pydantic ValidationError and formats it into a human-readable
        string that shows the field path and error message for each validation failure.

        Args:
            validation_error: The Pydantic ValidationError to convert

        Returns:
            Formatted error string with field paths and error messages
        """
        error_messages = []
        for error in validation_error.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_messages.append(f"{field_path}: {error['msg']}")
        return "; ".join(error_messages)
