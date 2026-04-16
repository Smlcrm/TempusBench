"""
Pydantic configuration models for validation and type safety.

This module defines all configuration models using Pydantic for comprehensive
validation, type checking, and documentation of the benchmarking pipeline.
"""

import yaml

from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional

# Upper bound for ``evaluation.max_windows`` in YAML/programmatic configs (rolling-window count).
MAX_EVALUATION_WINDOWS: Final[int] = 5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from tempus_bench.utils.paths import get_available_models

######################################################## UTILITY FUNCTIONS ########################################################


def convert_pydantic_errors(validation_error: PydanticValidationError) -> str:
    """
    Convert Pydantic validation errors to a readable string format.

    This function takes a Pydantic ValidationError and formats it into a
    human-readable string that shows the field path and error message for
    each validation failure.

    Args:
        validation_error (PydanticValidationError): The Pydantic ValidationError
            to convert.

    Returns:
        str: Formatted error string with field paths and error messages
            separated by semicolons.
    """
    error_messages = []
    for error in validation_error.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_messages.append(f"{field_path}: {error['msg']}")
    return "; ".join(error_messages)


######################################################## CONFIGURATION MODELS ########################################################


class EvaluationConfig(BaseModel):
    """
    Evaluation configuration model.

    This class defines the configuration parameters for model evaluation including
    tuning loss selection, window generation, and metric computation settings.

    Attributes:
        tuning_loss (Optional[Literal["mae", "mase", "mape", "rmse"]]): Loss metric
            for hyperparameter tuning. Only deterministic metrics allowed.
        max_windows (int): Maximum number of rolling windows to generate for evaluation
            (capped at MAX_EVALUATION_WINDOWS inclusive).
        max_num_variates (Optional[int]): Maximum number of variates to extract from
            dataset. None means all variates.
        num_samples (int): Number of samples to generate for stochastic metrics.
        num_quantiles (int): Number of quantiles to compute for quantile-based metrics.
        point_forecast_statistic (str): Statistic to use for converting stochastic
            predictions to point forecasts (e.g., "mean").
    """

    model_config = ConfigDict(extra="forbid")

    task_path: Optional[str] = Field(default=None, description="Single task path pattern")
    task_paths: Optional[List[str]] = Field(
        default=None,
        description="List of task paths; when set, overrides task_path for multiple tasks",
    )

    @model_validator(mode="after")
    def require_task_path_or_paths(self):
        if not self.task_path and not self.task_paths:
            raise ValueError("Either task_path or task_paths must be provided")
        if self.task_path and self.task_paths:
            raise ValueError("Provide task_path OR task_paths, not both")
        if self.task_paths and len(self.task_paths) == 0:
            raise ValueError("task_paths cannot be empty")
        return self

    tuning_loss: Optional[Literal["mae", "mase", "mape", "rmse"]] = Field(
        default="mae",
        description="Tuning loss for trainable models such as ARIMA, LSTM, DeepAR, SVR. "
        "Only deterministic (point) metrics are allowed: mae, mase, mape, rmse.",
    )
    max_windows: int = Field(
        default=MAX_EVALUATION_WINDOWS,
        ge=1,
        le=MAX_EVALUATION_WINDOWS,
        description=(
            "Maximum number of rolling windows to generate for evaluation "
            f"(must be between 1 and {MAX_EVALUATION_WINDOWS}, inclusive)"
        ),
    )
    max_num_variates: Optional[int] = Field(
        default=None,
        ge=1,
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
    point_forecast_statistic: Literal["mean", "median"] = Field(
        default="mean",
        description="Statistic to use for converting stochastic predictions to point forecasts",
    )


class ModelConfig:
    """
    Model configuration model.

    This class defines the configuration for a single model including its name
    and hyperparameter search space. The hyperparameter values are specified as
    lists to enable grid search over all combinations.

    Attributes:
        model_name (str): Name of the model (must match folder name in models directory).
        model_config (Dict[str, List[Any]]): Dictionary mapping hyperparameter names
            to lists of candidate values for grid search.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate: every attribute other than model_name is a list
        for attr, val in self.__dict__.items():
            if attr == "model_name":
                continue
            if not isinstance(val, list):
                raise ValueError(
                    f"ModelConfig: Attribute '{attr}' for model '{model_name}' must be a list of values, "
                    f"got {type(val).__name__}."
                )
            if len(val) == 0:
                raise ValueError(
                    f"ModelConfig: Attribute '{attr}' for model '{model_name}' cannot be an empty list."
                )


######################################################## TASK CONFIGS ########################################################


class DatasetConfig(BaseModel):
    """
    Dataset configuration model for individual task folders.

    This class defines dataset-specific configuration including file name,
    missing value handling strategy, and normalization settings.

    Attributes:
        handle_missing (Literal[...]): Strategy for handling missing values.
            Options: 'interpolate', 'mean', 'median', 'drop', 'forward_fill', 'backward_fill'.
        file_name (str): Dataset file name (CSV file).
        normalize (bool): Whether to normalize the data using StandardScaler.
    """

    model_config = ConfigDict(extra="forbid")

    handle_missing: Literal[
        "interpolate", "mean", "median", "drop", "forward_fill", "backward_fill"
    ] = Field(default="interpolate", description="Strategy for handling missing values")
    file_name: str = Field(..., description="Dataset file name")
    normalize: bool = Field(default=True, description="Whether to normalize the data")


class TaskConfig(BaseModel):
    """
    Task configuration model for individual task folders.

    This class defines task-specific configuration including task name, paths,
    forecast horizon, context window, and dataset settings.

    Attributes:
        name (str): Task name (must match folder name).
        task_path (str): Path to the task directory.
        forecast_horizon (int): Number of steps to forecast ahead (1-128).
        context_window (int): Number of context steps for training (>=1).
        dataset (DatasetConfig): Dataset configuration for this task.
    """

    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(..., description="Task name (must match folder name)")
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


######################################################## EVALUATION SETTINGS ########################################################


class EvaluationSetting(BaseModel):
    """
    System-wide evaluation settings configuration.

    This class defines global settings for logging, TensorBoard, and conda
    environment management across all models and tasks.

    Attributes:
        file_logging (bool): Enable file logging.
        file_log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR"]): File logging level.
        console_logging (bool): Enable console logging.
        console_log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR"]): Console logging level.
        tensorboard_logging (bool): Enable TensorBoard logging.
        conda_env_prefix (str): Prefix for conda environment names.
        reinstall_conda (bool): Whether to reinstall conda environments for each model.
    """

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
    reinstall_conda: bool = Field(
        ..., description="Whether to reinstall the conda environments for each model"
    )
    verbose: bool = Field(default=False, description="Whether to print verbose output")


class JobConfig:
    """
    Unified configuration model for a single benchmarking job.

    This class aggregates all configuration components for a single job execution,
    combining evaluation settings, model configuration, task configuration, and
    runtime settings into a single object.

    Attributes:
        evaluation_config (EvaluationConfig): Evaluation-specific configuration.
        evaluation_setting (EvaluationSetting): System-wide evaluation settings.
        model_config (ModelConfig): Model-specific configuration and hyperparameters.
        model_setting (Dict[str, Any]): Model execution settings (Python version,
            device, conda environment).
        task_config (TaskConfig): Task-specific configuration including dataset settings.
        run_path (str): Path to run directory for outputs.
        task_datasets_dir (Optional[str]): Dir for ``{task_name}.pkl`` (BenchmarkRunner temp dir).
        logger (LoggerManager): Logger instance for this job.
    """

    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        evaluation_setting: EvaluationSetting,
        model_config: ModelConfig,
        model_setting: Dict[str, Any],
        task_config: TaskConfig,
        run_path: str,
        task_datasets_dir: Optional[str] = None,
    ):
        """
        Initialize job configuration with all components.

        Args:
            evaluation_config (EvaluationConfig): Evaluation-specific configuration.
            evaluation_setting (EvaluationSetting): System-wide evaluation settings.
            model_config (ModelConfig): Model-specific configuration and hyperparameters.
            model_setting (Dict[str, Any]): Model execution settings.
            task_config (TaskConfig): Task-specific configuration.
            run_path (str): Path to run directory for outputs.
            task_datasets_dir: Directory for pickled task datasets; set by BenchmarkRunner.
        """

        self.evaluation_config = evaluation_config
        self.evaluation_setting = evaluation_setting
        self.model_config = model_config
        self.model_setting = model_setting
        self.task_config = task_config
        self.run_path = run_path
        self.task_datasets_dir = task_datasets_dir

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this JobConfig object to a dictionary for JSON serialization.

        This method serializes all components of this JobConfig into a
        dictionary format that can be safely converted to JSON. Pydantic models
        are converted using model_dump(), and ModelConfig attributes are extracted
        from its __dict__.

        Returns:
            Dict[str, Any]: Dictionary representation of the JobConfig suitable
                for JSON serialization.
        """
        d: Dict[str, Any] = {
            "evaluation_config": self.evaluation_config.model_dump(),
            "evaluation_setting": self.evaluation_setting.model_dump(),
            "model_config": {
                "model_name": self.model_config.model_name,
                **{
                    k: v
                    for k, v in self.model_config.__dict__.items()
                    if k != "model_name"
                },
            },
            "model_setting": self.model_setting,
            "task_config": self.task_config.model_dump(),
            "run_path": self.run_path,
        }
        if self.task_datasets_dir is not None:
            d["task_datasets_dir"] = self.task_datasets_dir
        return d
