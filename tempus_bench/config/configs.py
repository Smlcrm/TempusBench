"""
Pydantic configuration models for validation and type safety.

This module defines all configuration models using Pydantic for comprehensive
validation, type checking, and documentation of the benchmarking pipeline.
"""

import yaml

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from ..utils.logger import Logger


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

    # @field_validator("max_num_variates")
    # @classmethod
    # def validate_max_num_variates(cls, v):
    #     """Validate max_num_variates is either inf or a positive number."""
    #     if v == None:
    #         return v
    #     if isinstance(v, (int, float)) and v < 1:
    #         raise ValueError("max_num_variates must be at least 1 or inf")
    #     return v

    # @field_validator("tuning_loss")
    # @classmethod
    # def validate_tuning_loss(cls, v):
    #     """Validate that tuning_loss is a deterministic metric."""
    #     allowed_metrics = {"mae", "mase", "mape", "rmse"}
    #     if v is not None and v not in allowed_metrics:
    #         raise ValueError(
    #             f"tuning_loss must be one of: {', '.join(sorted(allowed_metrics))}"
    #         )
    #     return v


class ModelConfig(BaseModel):
    """Model configuration model."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., description="Model name")

    # @model_validator(mode="after")
    # def validate_model_availability(self):
    #     """
    #     Validate that all models specified in config are available.

    #     This method checks that each model specified in the configuration
    #     has a corresponding model file in the models directory structure.

    #     Raises:
    #         ValueError: If any specified model is not available in the models directory
    #     """
    #     available_models = get_available_models()
    #     model_dict = self.model_dump(exclude_none=True)

    #     for model_name in model_dict.keys():
    #         if model_name not in available_models:
    #             raise ValueError(
    #                 f"Model '{model_name}' is not available. "
    #                 f"Available models: {sorted(available_models)}"
    #             )
    #     return self

    # @field_validator("*", mode="before")
    # @classmethod
    # def validate_model_parameters(cls, v, info):
    #     """Validate model parameters structure."""
    #     if v is None:
    #         return v

    #     if not isinstance(v, dict):
    #         raise ValueError(f"Model parameters must be a dict, got {type(v).__name__}")

    #     # Traditional models should have lists of values for hyperparameter tuning
    #     for param_name, param_val in v.items():
    #         if not isinstance(param_val, list):
    #             raise ValueError(
    #                 f"Parameter '{param_name}' for model '{info.field_name}' must be a list of values, "
    #                 f"got {type(param_val).__name__}"
    #             )
    #         if len(param_val) == 0:
    #             raise ValueError(
    #                 f"Parameter '{param_name}' for model '{info.field_name}' cannot be an empty list"
    #             )

    #     return v


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


######################################################## EVALUATION SETTINGS ########################################################


class EvaluationSetting(BaseModel):
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
    reinstall_conda: bool = Field(..., description="Whether to reinstall the conda environments for each model")

class JobConfig:
    """Unified configuration model for the benchmarking pipeline (single-model only)."""

    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        evaluation_setting: EvaluationSetting,
        model_config: ModelConfig,
        model_setting: Dict[str, Any],
        task_config: TaskConfig,
        run_path: str
    ):

        self.evaluation_config = evaluation_config
        self.evaluation_setting = evaluation_setting
        self.model_config = model_config
        self.model_setting = model_setting
        self.task_config = task_config
        self.run_path = run_path
