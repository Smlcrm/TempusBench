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

from tempus_bench.utils.logger import get_logger
from tempus_bench.utils.tf_logger import get_tf_logger

from .models import BenchmarkConfig, JobConfig, TaskConfig, ModelSettingsConfig, BenchmarkSettingsConfig
from tempus_bench.utils.paths import get_models_dir, get_task_path, get_tasks_dir, get_configs_dir

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
        benchmark_config (BenchmarkConfig): Validated main benchmark configuration
        benchmark_settings (SystemsConfig): System settings from config/settings.yaml
        model_settings (Dict[str, ModelSettingsConfig]): Model execution settings (Python version, device, conda env)
        task_paths (List[Path]): Task paths matching the task_path pattern
        task (Dict[str, List[TaskConfig]]): Validated task configurations (each task can have multiple configs)
        model (Dict[str, Any]): Model hyperparameters from main config
    """

    def __init__(self, config_path: str, logs_path: str):
        """
        Initialize the configuration manager.

        This method performs initialization in the following order:
        1. Validates the main benchmark configuration (benchmark_config)
        2. Validates system settings (benchmark_settings)
        3. Validates model settings (only for models defined in benchmark_config)
        4. Finds task directories based on task_path pattern
        5. Validates task configurations for found task directories

        Args:
            config_path: Path to the main benchmark configuration YAML file
            logs_path: Directory for log files

        Initializes:
            - self.config_path: Configuration file path
            - self.benchmark_config: Main benchmark configuration with model hyperparameters
            - self.benchmark_settings: System settings (logging format, tensorboard, etc.)
            - self.task_paths: Task directories matching the task_path pattern from benchmark_config
            - self.model_settings: Model execution settings (Python version, device, conda env) for models in benchmark_config
            - self.task: Validated task configurations (each task can have multiple configs)
            - self.model: Model hyperparameters extracted from benchmark_config
        """
        # Setup paths and settings
        self.config_path = config_path
        self.logs_path = logs_path
        self.benchmark_settings = self.validate_benchmark_settings()
        self._setup_logging()

        self.benchmark_config = self.validate_benchmark_config()
        self.model_settings = self.validate_model_settings()
        self.task_paths = self._find_task_directories()
        self.task_configs = self.validate_task_configs()

    def _setup_logging(self):
        self.logger = get_logger(
            logs_path=self.logs_path,
            console_logging=self.benchmark_settings.console_logging,
            file_logging=self.benchmark_settings.file_logging,
            console_log_level=self.benchmark_settings.console_log_level,
            file_log_level=self.benchmark_settings.file_log_level
        )

    def validate_benchmark_config(self) -> BenchmarkConfig:
        """
        Validate the main benchmark configuration file.

        This method loads the configuration file specified in self.config_path, validates it
        against the BenchmarkConfig schema, and checks that all specified models are available.

        Returns:
            BenchmarkConfig: Validated benchmark configuration instance

        Raises:
            ConfigValidationError: If validation fails or models are not available
        """
        try:
            # Validate using Pydantic model
            config = self._load_config(self.config_path)
            config = BenchmarkConfig(**config)
            self._validate_model_availability(config)
            return config

        except ValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            raise ConfigValidationError(error_msg)

    def validate_benchmark_settings(self) -> BenchmarkSettingsConfig:
        """
        Validate the benchmark settings.yaml file.

        This method loads and validates the system settings configuration file located at
        config/settings.yaml against the BenchmarkSettingsConfig schema.

        Returns:
            BenchmarkSettingsConfig: Validated system settings instance

        Raises:
            FileNotFoundError: If settings.yaml doesn't exist
            ConfigValidationError: If validation fails
        """
        config_dir = get_configs_dir()
        benchmark_settings_dir = config_dir / "settings.yaml"

        if not benchmark_settings_dir.exists():
            raise FileNotFoundError(f"Settings config not found: {benchmark_settings_dir}")

        try:
            with open(benchmark_settings_dir, 'r') as f:
                benchmark_settings_data = yaml.safe_load(f)

            # Validate using BenchmarkSettingsConfig
            benchmark_settings = BenchmarkSettingsConfig(**benchmark_settings_data)
            return benchmark_settings

        except ValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            raise ConfigValidationError(f"Invalid systems config at {benchmark_settings_dir}: {error_msg}")
        except Exception as e:
            raise ConfigValidationError(f"Invalid systems config at {benchmark_settings_dir}: {e}")

    def validate_model_settings(self) -> Dict[str, ModelSettingsConfig]:
        """
        Validate model execution settings for models specified in benchmark_config.

        This method finds and validates model-specific settings.yaml files recursively
        in the models directory. Only processes models that are defined in
        self.benchmark_config.model.

        Returns:
            Dict[str, ModelSettingsConfig]: Dictionary mapping model names to their validated
                execution settings (Python version, device, conda environment)

        Raises:
            ConfigValidationError: If validation fails, models directory doesn't exist,
                or a model settings file is invalid
        """
        models_dir = get_models_dir()
        model_config = self.benchmark_config.model.model_dump()
        validated_settings = {}
        # Find all model settings.yaml files recursively in models_dir and use their parent folders as model names
        settings_files = list(models_dir.glob("**/settings.yaml"))
        for model_settings_path in settings_files:
            model_path = model_settings_path.parent
            model_name = model_path.name
            if model_config[model_name] is None: continue

            try:
                with open(model_settings_path, 'r') as f:
                    model_settings_data = yaml.safe_load(f)

                # Validate using ModelSettingsConfig
                model_settings = ModelSettingsConfig(**model_settings_data)
                validated_settings[model_name] = model_settings.model_dump()
                validated_settings[model_name].update(model_path=str(model_path))

            except Exception as e:
                error_msg = self._convert_pydantic_errors(e) if isinstance(e, ValidationError) else str(e)
                raise ConfigValidationError(
                    f"Invalid model settings for '{model_name}' at {model_settings_path}: {error_msg}"
                )

        return validated_settings

    def validate_task_configs(self) -> Dict[str, List[TaskConfig]]:
        """
        Validate task.yaml files for task directories found via task_path pattern.

        This method validates task configurations for all task directories in self.task_paths
        (which were determined by the task_path pattern in benchmark_config). Each task's
        configuration is validated against the TaskConfig schema. Supports multiple TaskConfigs
        per task using YAML multi-document format (separated by '---').

        Example task.yaml format:
        ```yaml
        task:
            name: soil_nature_univariate
            context_window: 256
            forecast_horizon: 16
            dataset:
                handle_missing: interpolate
                normalize: true
                file_name: soil_nature_univariate.csv
        ---
        task:
            name: soil_nature_univariate
            context_window: 512
            forecast_horizon: 16
            dataset:
                handle_missing: drop
                normalize: false
                file_name: soil_nature_univariate.csv
        ```

        Returns:
            Dict[str, List[TaskConfig]]: Dictionary mapping task names to lists of validated
                TaskConfig instances (supports multiple configs per task)

        Raises:
            ConfigValidationError: If validation fails, task.yaml not found, task name
                doesn't match folder name, or CSV file doesn't exist
        """
        validated_configs = {}

        for task_path in self.task_paths:
            task_config_path = task_path / "task.yaml"
            if not task_config_path.exists():
                raise ConfigValidationError(f"Task config not found: {task_config_path}")

            task_name = task_path.name

            try:
                with open(task_config_path, 'r') as f:
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
                    csv_file = task_path / task_config.dataset.file_name
                    if not csv_file.exists():
                        raise ConfigValidationError(
                            f"CSV file not found in {task_path}. Expected: {csv_file.name}"
                        )

                    task_configs.append(task_config)

                if not task_configs:
                    raise ConfigValidationError(f"No valid task configurations found in {task_config_path}")

                validated_configs[str(task_path)] = task_configs

            except Exception as e:
                error_msg = self._convert_pydantic_errors(e) if isinstance(e, ValidationError) else str(e)
                raise ConfigValidationError(f"Invalid task config at {task_config_path}: {error_msg}")

        return validated_configs

    def generate_run_configs(self):
        """
        Generate unified configurations for each task-model combination.

        This method yields UnifiedConfig instances that combine:
        - A benchmark configuration with the specific task path and single model hyperparameters
        - System settings
        - Model execution settings for the specific model
        - Task configurations for the specific task

        For each task in self.task and each model in self.model, a separate UnifiedConfig
        is generated with task_path set to the path for that task and model dict containing
        only that model's hyperparameters.

        Yields:
            UnifiedConfig: Unified configuration combining benchmark config, settings,
                model settings, and task configs for a single task-model combination
        """
        for task_name, task_configs in self.task_configs.items():
            for task_idx, task_config in enumerate(task_configs):
                # Build the updated task_path for this task
                task_path = get_task_path(task_name)
                for model_name, model_params in self.model.items():
                    # Build a new BenchmarkConfig for this (task, model)
                    benchmark_dict = self.benchmark_config.model_dump()
                    benchmark_dict["task_path"] = str(task_path)
                    benchmark_dict["model"] = {model_name: model_params}
                    yield JobConfig(
                        benchmark_config=BenchmarkConfig(**benchmark_dict),
                        benchmark_settings=self.benchmark_settings,
                        model_settings={model_name: self.model_settings[model_name]},
                        task_config=task_config
                    ), task_idx

    def _validate_model_availability(self, config: BenchmarkConfig) -> None:
        """
        Validate that all models specified in config are available.

        This method checks that each model specified in the benchmark configuration
        has a corresponding model file in the models directory structure. The models
        are validated against available models found in self._get_available_models().

        Args:
            config: The validated BenchmarkConfig instance

        Raises:
            ConfigValidationError: If any specified model is not available in the models directory
        """
        # Check each model in the configuration
        available_models = self._get_available_models()
        self.model = config.model.model_dump()
        # Only validate models that are actually specified (not None)
        for model_name, model_params in self.model.items():
            if model_name not in available_models:
                raise ConfigValidationError(
                    f"Model '{model_name}' is not available. "
                    f"Available models: {sorted(available_models)}"
                )

    def _get_available_models(self) -> set:
        """
        Get all available model names from the models directory structure.

        This method scans the models directory to find all model folders that contain
        a corresponding model file. The model file naming convention is {model_name}_model.py.

        Returns:
            set: Set of available model names found in the models directory
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
        Find task directories based on the task_path pattern from benchmark_config.

        This method searches for task directories that match the pattern specified in
        self.benchmark_config.task_path. Supported patterns include:
        - "*": All task directories
        - "univariate/*": All univariate task directories
        - "multivariate/*": All multivariate task directories
        - "specific_task": A specific task directory

        Returns:
            List[Path]: List of Path objects pointing to task directories that match the pattern
        """
        tasks_dir = get_tasks_dir()
        task_paths = []

        pattern = self.benchmark_config.task_path

        if pattern == "*":
            # Find all task directories
            for subdir in tasks_dir.iterdir():
                if subdir.is_dir():
                    for task_path in subdir.iterdir():
                        if task_path.is_dir():
                            task_paths.append(task_path)
        elif pattern.endswith("/*"):
            # Find directories in specific subdirectory
            subdir_name = pattern[:-2]
            subdir_path = tasks_dir / subdir_name
            if subdir_path.exists():
                for task_path in subdir_path.iterdir():
                    if task_path.is_dir():
                        task_paths.append(task_path)
        else:
            # Specific task directory
            task_path = tasks_dir / pattern
            if task_path.exists():
                task_paths.append(task_path)

        return task_paths

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file and return as dictionary.

        Args:
            config_path: Path to the configuration YAML file

        Returns:
            Dict[str, Any]: Dictionary containing the configuration data

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

    @staticmethod
    def _convert_pydantic_errors(validation_error: ValidationError) -> str:
        """
        Convert Pydantic validation errors to a readable string format.

        This method takes a Pydantic ValidationError and formats it into a human-readable
        string that shows the field path and error message for each validation failure.

        Args:
            validation_error: The Pydantic ValidationError to convert

        Returns:
            str: Formatted error string with field paths and error messages
        """
        error_messages = []
        for error in validation_error.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_messages.append(f"{field_path}: {error['msg']}")
        return "; ".join(error_messages)

class ConfigAdapterMixin:
    def __init__(self, config: JobConfig):
        self.job_config = config
        self.config = config.benchmark_config
        self.benchmark_settings = config.benchmark_settings
        self.model_settings = config.model_settings
        self.task_config = config.task_config
        self.eval_config = self.config.evaluation
        self.model_name, self.model_config = next(iter(self.config.model.model_dump(exclude_none=True).items()))
        self.tuning_loss = self.eval_config.tuning_loss
        self.dataset_path = get_task_path(self.task_config.name) / self.task_config.dataset.file_name
        self._setup_logging()

    def _setup_logging(self):
        self.logger = get_logger()
        self.tf_logger = get_tf_logger()