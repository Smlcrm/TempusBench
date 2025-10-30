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
    ModelHParams,
    TaskConfig,
    convert_pydantic_errors,
    load_and_validate_task_configs,
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
        evaluation (EvaluationConfig): Evaluation configuration from the benchmark configuration.
        model (ModelHParams): Model configuration with hyperparameters from the benchmark configuration.
        benchmark_settings (EvaluationSettings): Global logging/runtime options from `config/settings.yaml`.
        model_settings (Dict[str, Any]): Model execution settings derived from each model's `settings.yaml`.
        task_paths (Dict[str, str]): Mapping from task names to absolute directory paths.
        task_configs (Dict[str, List[TaskConfig]]): Per-task lists of validated task configurations.
    """

    def __init__(self, config_path: str, run_path: str, logger: Logger):
        """
        Initialize the configuration manager.

        This method performs initialization in the following order:
        1. Validates the main benchmark configuration
        2. Validates system settings (benchmark_settings)
        3. Validates model settings (only for models defined in config)
        4. Finds task directories based on task_path pattern
        5. Validates task configurations for found task directories

        Args:
            config_path: Path to the main benchmark configuration YAML file
            run_path: Directory for run outputs (evaluations, plots, etc.)
            logger: Logger instance to use for logging

        Initializes:
            - self.config_path: Configuration file path.
            - self.task_path: Task path pattern from the benchmark configuration.
            - self.evaluation: Evaluation configuration.
            - self.model: Model configuration with hyperparameters.
            - self.benchmark_settings: System settings (logging format, tensorboard, etc.).
            - self.task_paths: Task directories matching the task_path pattern.
            - self.model_settings: Model execution settings (Python version, device, conda env) for models.
            - self.task_configs: Validated task configurations (each task can have multiple configs).
        """
        # Setup paths and settings
        self.config_path = config_path

        config_data = self._load_config(self.config_path)
        evaluation_settings = self._load_config(get_project_root() / "tempus_bench" / "tasks" / "settings.yaml")
        
        self.models_evalated = config_data["model"].keys()
        
        self.task_path = config_data["task_path"]

        self.run_path = run_path
        self.logger = logger

        self.evaluation_config = EvaluationConfig(**config_data["evaluation"])
        self.evaluation_settings = EvaluationSettings(**evaluation_settings)
        
        self.models_hparams = self.init_models_hparams(config_data["model"])
        self.models_settings = self.init_models_settings()
        
        self.task_configs = self.init_tasks()

    # Evaluation config
    # Evaluation settings
    # Model params
    # Model settings
    # Dataset Config
    # Task Config
    # Logger

    # Job config (for each task-model combination, fixing model settings and benchmark settings):
    # Evaluation config
    # Evaluation settings
    # Model hparams
    # Model settings
    # Task config
    # Logger

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

    def init_models_hparams(self, models_hparams: Dict[str, Any]) -> Dict[str, ModelHParams]:
        """
        Initialize the model hyperparameters.

        This method initializes the model hyperparameters.

        Returns:
            Dict[str, ModelHParams]: Dictionary mapping model names to their validated ModelHParams objects.
        """
        model_hparams = {}
        for model_name in self.models_evalated:
            model_hparams[model_name] = ModelHParams(**models_hparams[model_name])
        return model_hparams

    def init_models_settings(self) -> Dict[str, Any]:
        """
        Validate model execution settings for models specified in the configuration.

        This method finds and validates model-specific settings.yaml files recursively
        in the models directory. Only processes models that are defined in
        self.model.

        Returns:
            Dict[str, Any]: Dictionary mapping model names to their validated
                execution settings (Python version, device, conda environment)

        Raises:
            ValidationError: If validation fails, models directory doesn't exist,
                or a model settings file is invalid
        """
        model_setting = {}
        models_dir = get_models_dir()

        # Find all model settings.yaml files recursively in models_dir
        settings_files = list(models_dir.glob("**/settings.yaml"))
        for model_settings_path in settings_files:
            model_name = model_settings_path.parent.name
            if (
                model_name
                not in self.models_evalated
            ):
                continue

            else:
                model_settings[model_name] = self._load_config(model_settings_path)

        return model_setting

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
        for task_name, task_configs in self.task_configs.items():
            for task_idx, task_config in enumerate(task_configs):
                # Build the updated task_path for this task
                task_path = get_task_path(task_name)
                model_dict = self.model.model_dump(exclude_none=True)
                for model_name, model_params in model_dict.items():
                    # Create a ModelHParams with only this single model
                    single_model_config = ModelHParams(**{model_name: model_params})

                    yield JobConfig(
                        task_path=str(task_path),
                        evaluation=self.evaluation,
                        model=single_model_config,
                        benchmark_settings=self.benchmark_settings,
                        model_settings={model_name: self.model_settings[model_name]},
                        task_config=task_config,
                        task_paths=self.task_paths,
                        run_path=self.run_path,
                        logger=self.logger,
                    ), task_idx


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
