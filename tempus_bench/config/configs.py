"""
Pydantic configuration models for validation and type safety.

This module defines all configuration models using Pydantic for comprehensive
validation, type checking, and documentation of the benchmarking pipeline.
"""

import yaml

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from ..utils.paths import get_available_models, get_models_dir


######################################################## UTILITY FUNCTIONS ########################################################


def convert_pydantic_errors(validation_error: PydanticValidationError) -> str:
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


    def validate_task_configs(self) -> Dict[str, List[TaskConfig]]:
        """
        Validate task.yaml files for task directories found via task_path pattern.

        This method validates task configurations for all task directories in self.task_paths
        (which were determined by the task_path pattern in the configuration). Each task's
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

        for task_name, task_path_str in self.task_paths.items():
            task_path = Path(task_path_str)
            task_config_path = task_path / "task.yaml"

            try:
                # Use validation function from configs.py
                task_configs = load_and_validate_task_configs(
                    task_config_path, task_name, task_path
                )
                validated_configs[task_path_str] = task_configs

            except (FileNotFoundError, ValueError) as e:
                raise ValidationError(str(e))

        return validated_configs


def validate_task_config_with_context(
    task_config: "TaskConfig", task_name: str, task_path: Path
) -> None:
    """
    Validate a TaskConfig instance against external context (folder name and file existence).

    This function performs validations that require external context:
    - Validates that task name matches folder name
    - Validates that CSV file exists in the task directory

    Args:
        task_config: The TaskConfig instance to validate
        task_name: The expected task name (from folder name)
        task_path: Path to the task directory

    Raises:
        ValueError: If task name doesn't match folder name or CSV file doesn't exist
    """
    # Validate that task name matches folder name
    if task_config.name != task_name:
        raise ValueError(
            f"Task name '{task_config.name}' in task.yaml does not match folder name '{task_name}'"
        )

    # Validate that CSV file exists and matches task name
    csv_file = task_path / task_config.dataset.file_name
    if not csv_file.exists():
        raise ValueError(
            f"CSV file not found in {task_path}. Expected: {csv_file.name}"
        )


def load_and_validate_task_configs(
    task_config_path: Path, task_name: str, task_path: Path
) -> List["TaskConfig"]:
    """
    Load and validate task configurations from a task.yaml file.

    This function loads task.yaml files that may contain multiple configurations
    separated by '---' (YAML multi-document format), validates each configuration
    against the TaskConfig schema, and performs context-aware validations.

    Args:
        task_config_path: Path to the task.yaml file
        task_name: Expected task name (from folder name)
        task_path: Path to the task directory

    Returns:
        List[TaskConfig]: List of validated TaskConfig instances

    Raises:
        FileNotFoundError: If task.yaml doesn't exist
        ValueError: If validation fails or no valid configs found
    """
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_config_path}")

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
                raise ValueError(
                    f"All sections in {task_config_path} must contain a 'task' key."
                )

            # Validate and create TaskConfig
            task_config = TaskConfig(**task_data)

            # Perform context-aware validation
            validate_task_config_with_context(task_config, task_name, task_path)

            task_configs.append(task_config)

        if not task_configs:
            raise ValueError(
                f"No valid task configurations found in {task_config_path}"
            )

        return task_configs

    except Exception as e:
        error_msg = (
            convert_pydantic_errors(e)
            if isinstance(e, PydanticValidationError)
            else str(e)
        )
        raise ValueError(f"Invalid task config at {task_config_path}: {error_msg}")


######################################################## CONFIGURATION MODELS ########################################################


class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""

    model_config = ConfigDict(extra="forbid")

    tuning_loss: Optional[Literal["mae", "mase", "mape", "rmse"]] = Field(
        default="mae",
        description="Tuning loss for trainable models such as ARIMA, LSTM, DeepAR, SVR. "
        "Only deterministic (point) metrics are allowed: mae, mase, mape, rmse.",
    )
    max_windows: int = Field(
        default=10,
        ge=1,
        description="Maximum number of rolling windows to generate for evaluation",
    )
    max_num_variates: Optional[int] = Field(
        default=None,
        description="Maximum number of variates to extract from dataset for evaluation (use None for all variates)",
    )
    num_samples: int = Field(
        default=100,
        ge=1,
        description="Number of samples to generate for stochastic metrics",
    )
    num_quantiles: int = Field(
        default=10,
        ge=1,
        description="Number of quantiles to compute for quantile-based metrics",
    )
    point_forecast_statistic: str = Field(
        default="mean",
        description="Statistic to use for converting stochastic predictions to point forecasts",
    )

    @field_validator("max_num_variates")
    @classmethod
    def validate_max_num_variates(cls, v):
        """Validate max_num_variates is either inf or a positive number."""
        if v == None:
            return v
        if isinstance(v, (int, float)) and v < 1:
            raise ValueError("max_num_variates must be at least 1 or inf")
        return v

    @field_validator("tuning_loss")
    @classmethod
    def validate_tuning_loss(cls, v):
        """Validate that tuning_loss is a deterministic metric."""
        allowed_metrics = {"mae", "mase", "mape", "rmse"}
        if v is not None and v not in allowed_metrics:
            raise ValueError(
                f"tuning_loss must be one of: {', '.join(sorted(allowed_metrics))}"
            )
        return v


class ModelHParams(BaseModel):
    """Model configuration model."""

    model_config = ConfigDict(extra="forbid")

    # TODO: Remove the necessity of having these models in the config, they should be infered from the model folder
    # Traditional models with hyperparameters
    exponential_smoothing: Optional[Dict[str, List[Any]]] = None
    seasonal_naive: Optional[Dict[str, List[Any]]] = None
    croston_classic: Optional[Dict[str, List[Any]]] = None
    theta: Optional[Dict[str, List[Any]]] = None
    arima: Optional[Dict[str, List[Any]]] = None
    xgboost: Optional[Dict[str, List[Any]]] = None
    random_forest: Optional[Dict[str, List[Any]]] = None
    svr: Optional[Dict[str, List[Any]]] = None
    prophet: Optional[Dict[str, List[Any]]] = None
    lstm: Optional[Dict[str, List[Any]]] = None
    varmax: Optional[Dict[str, List[Any]]] = None
    lafn: Optional[Dict[str, Any]] = None

    # Foundation models (no hyperparameters)
    chronos: Optional[Dict[str, Any]] = None
    deepar: Optional[Dict[str, Any]] = None
    tiny_time_mixer: Optional[Dict[str, Any]] = None
    moirai: Optional[Dict[str, Any]] = None
    moirai_moe: Optional[Dict[str, Any]] = None
    moment: Optional[Dict[str, Any]] = None
    timesfm: Optional[Dict[str, Any]] = None
    lagllama: Optional[Dict[str, Any]] = None
    toto: Optional[Dict[str, Any]] = None
    tabpfn: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_model_availability(self):
        """
        Validate that all models specified in config are available.

        This method checks that each model specified in the configuration
        has a corresponding model file in the models directory structure.

        Raises:
            ValueError: If any specified model is not available in the models directory
        """
        available_models = get_available_models()
        model_dict = self.model_dump(exclude_none=True)

        for model_name in model_dict.keys():
            if model_name not in available_models:
                raise ValueError(
                    f"Model '{model_name}' is not available. "
                    f"Available models: {sorted(available_models)}"
                )
        return self

    @field_validator("*", mode="before")
    @classmethod
    def validate_model_parameters(cls, v, info):
        """Validate model parameters structure."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError(f"Model parameters must be a dict, got {type(v).__name__}")

        # Traditional models should have lists of values for hyperparameter tuning
        for param_name, param_val in v.items():
            if not isinstance(param_val, list):
                raise ValueError(
                    f"Parameter '{param_name}' for model '{info.field_name}' must be a list of values, "
                    f"got {type(param_val).__name__}"
                )
            if len(param_val) == 0:
                raise ValueError(
                    f"Parameter '{param_name}' for model '{info.field_name}' cannot be an empty list"
                )

        return v


######################################################## TASK CONFIGS ########################################################


class DatasetConfig(BaseModel):
    """Dataset configuration model for individual task folders."""

    model_config = ConfigDict(extra="forbid")

    handle_missing: Literal[
        "interpolate", "mean", "median", "drop", "forward_fill", "backward_fill"
    ] = Field(default="interpolate", description="Strategy for handling missing values")
    file_name: str = Field(..., description="Dataset file name")
    normalize: bool = Field(default=True, description="Whether to normalize the data")


class TaskConfig(BaseModel):
    """Task configuration model for individual task folders."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Task name (must match folder name)")

    task_path: str = Field(..., description="Task path")
    
    forecast_horizon: int = Field(
        ..., ge=1, le=128, description="Number of steps to forecast ahead (max 128)"
    )
    context_window: int = Field(
        ..., ge=1, description="Number of context steps for training"
    )
    dataset: DatasetConfig = Field(
        ..., description="Dataset configuration for this task"
    )


######################################################## TASK CONFIGS ########################################################


class EvaluationSettings(BaseModel):
    """Systems configuration model."""

    model_config = ConfigDict(extra="forbid")

    file_logging: bool = Field(..., description="Enable file logging")
    file_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="DEBUG", description="File logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    console_logging: bool = Field(..., description="Enable console logging")
    console_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Console logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    tensorboard_logging: bool = Field(..., description="Enable TensorBoard logging")
    conda_env_prefix: str = Field(..., description="Prefix for conda environment names")


class JobConfig(BaseModel):
    """Unified configuration model for the benchmarking pipeline (single-model only)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Fields from BenchmarkConfig
    task_path: str = Field(
        ...,
        description="Task path pattern (supports wildcards like '*', 'univariate/*', etc.)",
    )
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")
    model: ModelHParams = Field(..., description="Model configuration")

    # Other JobConfig fields
    benchmark_settings: EvaluationSettings = Field(
        ..., description="Benchmark settings"
    )
    model_settings: Dict[str, Any] = Field(..., description="Model execution settings")
    task_config: TaskConfig = Field(..., description="Task configuration")
    task_paths: Dict[str, str] = Field(..., description="Task paths")
    run_path: str = Field(..., description="Path to run directory for outputs")
    logger: Any = Field(..., description="Logger instance for logging")

    @model_validator(mode="after")
    def validate_tuning_loss_usage(self):
        """
        Ensure that 'tuning_loss' is properly defined for models that require it
        for training/optimization, and not defined for models that don't use it.

        Models which have more than 1 possible hyperparameter configuration are
        considered "trainable" and require a tuning_loss.
        """
        # We infer trainable models as those with >1 unique hyperparameter grid combination.
        from itertools import product

        model_dict = self.model.model_dump(exclude_none=True)
        trainable_models = set()
        for model_name, params in model_dict.items():
            if not isinstance(params, dict) or not params:
                continue
            # For each param, get the number of options
            param_lists = [
                v for v in params.values() if isinstance(v, list) and len(v) > 0
            ]
            if not param_lists:
                continue
            # Calculate the total number of possible hyperparam configs
            n_configs = 1
            for l in param_lists:
                n_configs *= len(l)
            if n_configs > 1:
                trainable_models.add(model_name)

        for model_name in trainable_models:
            if self.evaluation.tuning_loss is None:
                raise ValueError(
                    f"Model '{model_name}' requires 'tuning_loss' parameter for training "
                    f"but it is not defined. Please add 'tuning_loss' to the model configuration."
                )
        return self

    @model_validator(mode="after")
    def validate_single_model_and_existence(self):
        model_settings = self.model_settings

        # Check that only a single model is referenced in both places
        # model is an instance of ModelHParams; extract the list of models with non-None config
        model_config = self.model.model_dump(exclude_none=True)
        if len(model_config) != 1:
            raise ValueError("JobConfig must reference exactly one model")
        model_in_config = list(model_config.keys())[0]

        if len(model_settings) != 1:
            raise ValueError("model_settings must contain exactly one model")
        model_in_settings = list(model_settings.keys())[0]

        if model_in_config != model_in_settings:
            raise ValueError(
                f"Model names do not match in JobConfig: model in config is '{model_in_config}', "
                f"but in model_settings is '{model_in_settings}'"
            )

        model_path = Path(model_settings[model_in_config]["model_path"])
        model_file = model_path / f"{model_in_config}_model.py"
        if not (model_path.exists() and model_path.is_dir()):
            raise ValueError(f"Model directory does not exist: {model_path}")
        if not model_file.exists():
            raise ValueError(f"Model file does not exist: {model_file}")

        return self

    @staticmethod
    def _load_config(config_path: str | Path) -> Dict[str, Any]:
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


# Evaluation params
# Evaluation settings
# Model params
# Model settings
# Dataset Config
# Task Config
# Logger

# TODO: Rename this file models.puy