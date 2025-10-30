"""
Configuration manager for the benchmarking pipeline.

This module provides comprehensive validation and management of configuration files using Pydantic
to ensure they comply with the expected schema before execution. The Manager class handles
validation of benchmark configurations, model settings, task configurations, and system settings.
"""

import yaml

from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError as PydanticValidationError

from .configs import BenchmarkConfig, BenchmarkSettingsConfig, JobConfig, TaskConfig
from ..utils.logger import Logger
from ..utils.paths import get_configs_dir, get_models_dir, get_task_path, get_tasks_dir
from ..utils.tf_logger import get_tf_logger


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class Manager:
    """
    Configuration manager with comprehensive validation rules.

    This class handles the validation and management of all configuration files
    in the benchmarking pipeline, including benchmark configurations, model settings,
    task configurations, and system settings.

    Attributes:
        config_path (str): Path to the main benchmark configuration YAML file.
        benchmark_config (BenchmarkConfig): Validated benchmark definition loaded from `benchmark.yaml`.
        benchmark_settings (BenchmarkSettingsConfig): Global logging/runtime options from `config/settings.yaml`.
        model_settings (Dict[str, Any]): Model execution settings derived from each model's `settings.yaml`.
        task_paths (Dict[str, str]): Mapping from task names to absolute directory paths.
        task_configs (Dict[str, List[TaskConfig]]): Per-task lists of validated task configurations.
        model (Dict[str, Any]): Hyperparameter definitions extracted from the benchmark configuration.
    """

    def __init__(self, config_path: str, logs_path: str, logger: Logger):
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
            logger: Logger instance to use for logging

        Initializes:
            - self.config_path: Configuration file path.
            - self.benchmark_config: Main benchmark configuration with model hyperparameters.
            - self.benchmark_settings: System settings (logging format, tensorboard, etc.).
            - self.task_paths: Task directories matching the task_path pattern from benchmark_config.
            - self.model_settings: Model execution settings (Python version, device, conda env) for models in benchmark_config.
            - self.task_configs: Validated task configurations (each task can have multiple configs).
            - self.model: Model hyperparameters extracted from benchmark_config.
        """
        # Setup paths and settings
        self.config_path = config_path
        self.logs_path = logs_path
        self.logger = logger
        self.benchmark_settings = self.validate_benchmark_settings()

        self.benchmark_config = self.validate_benchmark_config()
        self.model_settings = self.validate_model_settings()
        self.task_paths = self._find_task_directories()
        self.logger.info("Manager.init", f"task_paths: {self.task_paths}")
        self.task_configs = self.validate_task_configs()
        self.logger.info("Manager.init", f"task_configs: {self.task_configs}")

    def validate_benchmark_config(self) -> BenchmarkConfig:
        """
        Validate the main benchmark configuration file.

        This method loads the configuration file specified in self.config_path, validates it
        against the BenchmarkConfig schema, and checks that all specified models are available.

        Returns:
            BenchmarkConfig: Validated benchmark configuration instance

        Raises:
            ValidationError: If validation fails or models are not available
        """
        try:
            # Validate using Pydantic model
            config = self._load_config(self.config_path)
            self.logger.info("Manager.validate_benchmark_config", f"Config: {config}")
            config = BenchmarkConfig(**config)
            self.logger.info("Manager.validate_benchmark_config", f"Validated config: {config}")
            self._validate_model_availability(config)
            self.logger.info("Manager.validate_benchmark_config", f"Validated model availability: {config}")
            return config

        except PydanticValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            raise ValidationError(error_msg)

    def validate_benchmark_settings(self) -> BenchmarkSettingsConfig:
        """
        Validate the benchmark settings.yaml file.

        This method loads and validates the system settings configuration file located at
        config/settings.yaml against the BenchmarkSettingsConfig schema.

        Returns:
            BenchmarkSettingsConfig: Validated system settings instance

        Raises:
            FileNotFoundError: If settings.yaml doesn't exist
            ValidationError: If validation fails
        """
        config_dir = get_configs_dir()
        benchmark_settings_dir = config_dir / "settings.yaml"

        if not benchmark_settings_dir.exists():
            raise FileNotFoundError(
                f"Settings config not found: {benchmark_settings_dir}"
            )

        try:
            with open(benchmark_settings_dir, "r") as f:
                benchmark_settings_data = yaml.safe_load(f)

            # Validate using BenchmarkSettingsConfig
            benchmark_settings = BenchmarkSettingsConfig(**benchmark_settings_data)
            return benchmark_settings

        except PydanticValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            raise ValidationError(
                f"Invalid systems config at {benchmark_settings_dir}: {error_msg}"
            )
        except Exception as e:
            raise ValidationError(
                f"Invalid systems config at {benchmark_settings_dir}: {e}"
            )

    def validate_model_settings(self) -> Dict[str, Any]:
        """
        Validate model execution settings for models specified in benchmark_config.

        This method finds and validates model-specific settings.yaml files recursively
        in the models directory. Only processes models that are defined in
        self.benchmark_config.model.

        Returns:
            Dict[str, Any]: Dictionary mapping model names to their validated
                execution settings (Python version, device, conda environment)

        Raises:
            ValidationError: If validation fails, models directory doesn't exist,
                or a model settings file is invalid
        """
        models_dir = get_models_dir()
        model_config = self.benchmark_config.model.model_dump(exclude_none=True)
        validated_settings = {}
        # Find all model settings.yaml files recursively in models_dir and use their parent folders as model names
        settings_files = list(models_dir.glob("**/settings.yaml"))
        for model_settings_path in settings_files:
            model_path = model_settings_path.parent
            model_name = model_path.name
            if model_name not in model_config:
                continue

            try:
                with open(model_settings_path, "r") as f:
                    model_settings_data = yaml.safe_load(f)

                model_settings_data.update(model_path=str(model_path))
                model_settings = dict[str, Any](**model_settings_data)
                validated_settings[model_name] = model_settings
                validated_settings[model_name].update(model_path=str(model_path))

            except Exception as e:
                error_msg = (
                    self._convert_pydantic_errors(e)
                    if isinstance(e, PydanticValidationError)
                    else str(e)
                )
                raise ValidationError(
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
            Dict[str, List[TaskConfig]]: Dictionary mapping task directory paths (as strings)
                to lists of validated TaskConfig instances (supports multiple configs per task).

        Raises:
            ValidationError: If validation fails, task.yaml not found, task name
                doesn't match folder name, or CSV file doesn't exist
        """
        validated_configs = {}
        self.logger.info("Manager.validate_task_configs", f"task_paths: {self.task_paths}")
        for task_name, task_path in self.task_paths.items():
            self.logger.info("Manager.validate_task_configs", f"task_name: {task_name}")
            self.logger.info("Manager.validate_task_configs", f"task_path: {task_path}")
            task_config_path = Path(task_path) / "task.yaml"
            if not task_config_path.exists():
                raise ValidationError(f"Task config not found: {task_config_path}")

            try:
                with open(task_config_path, "r") as f:
                    task_config_documents = list(yaml.safe_load_all(f))

                # Handle multiple task configs defined with '---' separator
                task_configs = []
                for doc in task_config_documents:
                    if not doc:
                        continue  # Skip empty documents
                    # Extract the 'task' key if present
                    if isinstance(doc, dict) and "task" in doc:
                        task_data = doc["task"]
                    else:
                        raise ValidationError(
                            f"All sections in {task_config_path} must contain a 'task' key."
                        )

                    # Validate and create TaskConfig
                    task_config = TaskConfig(**task_data)

                    # Validate that task name matches folder name
                    if task_config.name != task_name:
                        raise ValidationError(
                            f"Task name '{task_config.name}' in {task_config_path} does not match folder name '{task_name}'"
                        )

                    # Validate that CSV file exists and matches task name
                    csv_file = Path(task_path) / task_config.dataset.file_name
                    if not csv_file.exists():
                        raise ValidationError(
                            f"CSV file not found in {task_path}. Expected: {csv_file.name}"
                        )

                    task_configs.append(task_config)

                if not task_configs:
                    raise ValidationError(
                        f"No valid task configurations found in {task_config_path}"
                    )

                validated_configs[str(task_path)] = task_configs

            except Exception as e:
                error_msg = (
                    self._convert_pydantic_errors(e)
                    if isinstance(e, PydanticValidationError)
                    else str(e)
                )
                raise ValidationError(
                    f"Invalid task config at {task_config_path}: {error_msg}"
                )

        return validated_configs

    def generate_run_configs(self):
        """
        Generate unified configurations for each task-model combination.

        This method yields JobConfig instances that combine:
        - A benchmark configuration with the specific task path and single model hyperparameters
        - System settings
        - Model execution settings for the specific model
        - Task configurations for the specific task

        For each task in self.task and each model in self.model, a separate JobConfig
        is generated with task_path set to the path for that task and model dict containing
        only that model's hyperparameters.

        Yields:
            JobConfig: Aggregated configuration combining benchmark config, settings,
                model execution metadata, and a single task configuration.
        """
        self.logger.info("Manager.generate_run_configs", f"task_configs: {self.task_configs}")
        for task_name, task_configs in self.task_configs.items():
            self.logger.info("Manager.generate_run_configs", f"task_name: {task_name}")
            for task_idx, task_config in enumerate(task_configs):
                self.logger.info("Manager.generate_run_configs", f"task_idx: {task_idx}")
                # Build the updated task_path for this task
                task_path = get_task_path(task_name)
                for model_name, model_params in self.model.items():
                    self.logger.info("Manager.generate_run_configs", f"model_name: {model_name}")
                    # Build a new BenchmarkConfig for this (task, model)
                    benchmark_dict = self.benchmark_config.model_dump()
                    benchmark_dict["task_path"] = str(task_path)
                    benchmark_dict["model"] = {model_name: model_params}
                    yield JobConfig(
                        benchmark_config=BenchmarkConfig(**benchmark_dict),
                        benchmark_settings=self.benchmark_settings,
                        model_settings={
                            **self.model_settings[model_name],
                            "model_name": model_name
                        },
                        task_config=task_config,
                        task_paths=self.task_paths,
                        config_path=self.config_path,
                        logs_path=self.logs_path,
                        logger=self.logger,
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
            ValidationError: If any specified model is not available in the models directory
        """
        # Check each model in the configuration
        available_models = self._get_available_models()
        self.model = config.model.model_dump(exclude_none=True)
        # Only validate models that are actually specified (not None)
        for model_name, model_params in self.model.items():
            if model_name not in available_models:
                raise ValidationError(
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
        # Look for model folders directly in the models directory
        for model_folder in models_dir.iterdir():
            if (
                model_folder.is_dir()
                and not model_folder.name.startswith("__")
                and not model_folder.name.startswith(".")
            ):
                # Search for a Python model file inside the folder
                py_files = list(model_folder.glob("*.py"))
                expected_model_file = model_folder / f"{model_folder.name}_model.py"
                found_model_file = None

                for py_file in py_files:
                    if py_file.name.endswith("_model.py"):
                        if py_file.name != expected_model_file.name:
                            raise Exception(
                                f"Model file convention not followed in folder '{model_folder.name}'. "
                                f"Expected filename: '{model_folder.name}_model.py', found '{py_file.name}'"
                            )
                        found_model_file = py_file

                if found_model_file and found_model_file.exists():
                    available_models.add(model_folder.name)
                else:
                    raise Exception(
                        f"Model file not found in folder '{model_folder.name}'. "
                        f"Expected filename: '{model_folder.name}_model.py'"
                    )
        self.logger.info("Manager._get_available_models", f"Available models: {available_models}")
        return available_models

    def _find_task_directories(self) -> Dict[str, str]:
        """
        Find task directories based on the task_path pattern from benchmark_config.

        This method searches for task directories that match the pattern specified in
        self.benchmark_config.task_path. Supported patterns include:
        - "*": All task directories
        - "univariate/*": All univariate task directories
        - "multivariate/*": All multivariate task directories
        - "specific_task": A specific task directory

        Returns:
            Dict[str, str]: Mapping from task names to absolute directory paths that match the pattern.
        """
        tasks_dir = get_tasks_dir()
        task_paths = {}

        pattern = self.benchmark_config.task_path

        if pattern == "*":
            # Find all task directories
            for subdir in tasks_dir.iterdir():
                if subdir.is_dir():
                    for task_path in subdir.iterdir():
                        if task_path.is_dir():
                            task_paths[task_path.name] = str(task_path)
        elif pattern.endswith("/*"):
            # Find directories in specific subdirectory
            subdir_name = pattern[:-2]
            subdir_path = tasks_dir / subdir_name
            if subdir_path.exists():
                for task_path in subdir_path.iterdir():
                    if task_path.is_dir():
                        task_paths[task_path.name] = str(task_path)
        else:
            # Specific task directory
            self.logger.info("Manager._find_task_directories", f"pattern: {pattern}")
            task_path = tasks_dir / pattern
            self.logger.info("Manager._find_task_directories", f"task_path: {task_path}")
            if task_path.exists():
                self.logger.info("Manager._find_task_directories", f"task_path exists: {task_path}")
                task_paths[task_path.name] = str(task_path)
            self.logger.info("Manager._find_task_directories", f"task_path exists: {task_path.exists()}")
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
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                raise ValueError("Configuration file is empty or invalid YAML")

            return config_data

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")

    @staticmethod
    def _convert_pydantic_errors(validation_error: PydanticValidationError) -> str:
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
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_messages.append(f"{field_path}: {error['msg']}")
        return "; ".join(error_messages)
