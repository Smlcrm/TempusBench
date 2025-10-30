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

from .configs import (
    EvaluationConfig,
    EvaluationSettings,
    DatasetConfig,
    JobConfig,
    ModelConfig,
    TaskConfig,
    convert_pydantic_errors,
)
from ..utils.logger import Logger
from ..utils.paths import (
    get_project_root,
    get_configs_dir,
    get_models_dir,
    get_task_path,
    get_tasks_dir,
    find_task_directories,
)
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
        task_path (str): Task path pattern from the benchmark configuration.
        models_evalated (KeysView): Keys of models to be evaluated.
        run_path (str): Path to run directory for outputs.
        logger (Logger): Logger instance for logging.
        evaluation_config (EvaluationConfig): Evaluation configuration from the benchmark configuration.
        evaluation_settings (EvaluationSettings): Global logging/runtime options from `tasks/settings.yaml`.
        models_config (Dict[str, ModelConfig]): Model hyperparameters for each model.
        model_settings (Dict[str, Any]): Model execution settings derived from each model's `settings.yaml`.
        task_configs (Dict[str, TaskConfig]): Mapping from task names to validated task configurations.
    """

    def __init__(self, config_path: str, run_path: str, logger: Logger):
        """
        Initialize the configuration manager.

        This method performs initialization in the following order:
        1. Loads the main benchmark configuration
        2. Loads evaluation settings from tasks/settings.yaml
        3. Extracts models to be evaluated
        4. Initializes evaluation configuration and settings
        5. Initializes model hyperparameters and settings
        6. Initializes task configurations

        Args:
            config_path: Path to the main benchmark configuration YAML file
            run_path: Directory for run outputs (evaluations, plots, etc.)
            logger: Logger instance to use for logging

        Initializes:
            - self.config_path: Configuration file path.
            - self.task_path: Task path pattern from the benchmark configuration.
            - self.models_evaluated: Keys of models to be evaluated.
            - self.run_path: Path to run directory.
            - self.logger: Logger instance.
            - self.evaluation_config: Evaluation configuration.
            - self.evaluation_settings: System settings (logging format, tensorboard, etc.).
            - self.model_configs: Model hyperparameters for each model.
            - self.model_settings: Model execution settings (Python version, device, conda env) for models.
            - self.task_configs: Validated task configurations.
        """
        # Setup paths and settings
        self.config_path = config_path

        config_data = self._load_config(self.config_path)
        evaluation_settings = self._load_config(
            get_project_root() / "tempus_bench" / "tasks" / "settings.yaml"
        )

        self.models_evaluated = config_data["model"].keys()

        self.task_path = config_data["task_path"]

        self.run_path = run_path
        self.logger = logger

        self.evaluation_config = EvaluationConfig(**config_data["evaluation"])
        self.evaluation_settings = EvaluationSettings(**evaluation_settings)

        self.model_configs = self.init_models_config(config_data["model"])
        self.model_settings = self.init_model_settings()

        self.task_configs = self.init_tasks()

    def init_tasks(self) -> Dict[str, TaskConfig]:
        """
        Initialize the tasks.

        Discovers and loads all TaskConfig objects from task.yaml files in directories
        found via find_task_directories using self.task_path.

        Returns:
            Dict[str, TaskConfig]: Mapping of task directory names to their validated TaskConfig objects.

        Raises:
            FileNotFoundError or ValidationError if a task.yaml file is missing or invalid.
        """

        task_configs = {}
        tasks = find_task_directories(self.task_path)  # Dict[str, str]: name => path

        for task_name, task_path in tasks.items():
            from pathlib import Path

            task_config_path = Path(task_path) / "task.yaml"

            task_data = self._load_config(task_config_path)
            dataset_config = DatasetConfig(**task_data["task"].pop("dataset"))
            task_configs[task_name] = TaskConfig(
                name=task_name,
                **task_data["task"],
                task_path=task_path,
                dataset=dataset_config,
            )

        return task_configs

    def init_models_config(
        self, models_config: Dict[str, Any]
    ) -> Dict[str, ModelConfig]:
        """
        Initialize the model hyperparameters.

        This method initializes the model hyperparameters.

        Returns:
            Dict[str, ModelConfig]: Dictionary mapping model names to their validated ModelConfig objects.
        """
        model_hparams = {}
        for model_name in self.models_evaluated:
            model_hparams[model_name] = ModelConfig(**models_config[model_name])
        return model_hparams

    def init_model_settings(self) -> Dict[str, Any]:
        """
        Validate model execution settings for models specified in the configuration.

        This method finds and validates model-specific settings.yaml files recursively
        in the models directory. Only processes models that are defined in
        self.models_evaluated.

        Returns:
            Dict[str, Any]: Dictionary mapping model names to their validated
                execution settings (Python version, device, conda environment)

        Raises:
            ValidationError: If validation fails, models directory doesn't exist,
                or a model settings file is invalid
        """
        model_settings = {}
        models_dir = get_models_dir()

        # Find all model settings.yaml files recursively in models_dir
        settings_files = list(models_dir.glob("**/settings.yaml"))
        for model_settings_path in settings_files:
            model_name = model_settings_path.parent.name
            if model_name not in self.models_evaluated:
                continue
            else:
                model_settings[model_name] = self._load_config(model_settings_path)

        return model_settings

    def generate_run_configs(self):
        """
        Generate unified configurations for each task-model combination.

        This method yields JobConfig instances that combine:
        - A benchmark configuration with the specific task path and single model hyperparameters
        - System settings
        - Model execution settings for the specific model
        - Task configurations for the specific task

        For each task and each model, a separate JobConfig is generated.

        Yields:
            Tuple[JobConfig, int]: A tuple of (JobConfig, task_idx) where JobConfig is the
                aggregated configuration and task_idx is always 0 (for backward compatibility).
        """
        for task_name, task_config in self.task_configs.items():
            for model_name in self.models_evaluated:
                yield  JobConfig(
                        evaluation_config=self.evaluation_config,
                        evaluation_settings=self.evaluation_settings,
                        modelsx_config=self.model_configs[model_name],
                        models_setting=self.model_settings[model_name],
                        task_config=task_config,
                        run_path=self.run_path,
                        logger=self.logger,
                    )

    @staticmethod
    def _load_config(path: str | Path) -> Dict[str, Any]:
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
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                raise ValueError("Configuration file is empty or invalid YAML")

            return config_data

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {path}: {e}")
