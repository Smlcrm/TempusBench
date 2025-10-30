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

    # Core configuration components (matching Manager initialization)
    evaluation_config: EvaluationConfig = Field(
        ..., description="Evaluation configuration"
    )
    evaluation_settings: EvaluationSettings = Field(
        ..., description="Evaluation settings (logging, tensorboard, conda env prefix)"
    )
    model_configs: Dict[str, Any] = Field(..., description="Model hyperparameters")
    models_setting: Dict[str, Any] = Field(..., description="Model execution settings")
    task_config: TaskConfig = Field(..., description="Task configuration")

    # Additional fields
    run_path: str = Field(..., description="Path to run directory for outputs")
    logger: Any = Field(..., description="Logger instance for logging")

    # @model_validator(mode="after")
    # def validate_tuning_loss_usage(self):
    #     """
    #     Ensure that 'tuning_loss' is properly defined for models that require it
    #     for training/optimization, and not defined for models that don't use it.

    #     Models which have more than 1 possible hyperparameter configuration are
    #     considered "trainable" and require a tuning_loss.
    #     """
    #     # We infer trainable models as those with >1 unique hyperparameter grid combination.
    #     trainable_models = set()
    #     for model_name, params in self.model_configs.items():
    #         if not isinstance(params, dict) or not params:
    #             continue
    #         # For each param, get the number of options
    #         param_lists = [
    #             v for v in params.values() if isinstance(v, list) and len(v) > 0
    #         ]
    #         if not param_lists:
    #             continue
    #         # Calculate the total number of possible hyperparam configs
    #         n_configs = 1
    #         for l in param_lists:
    #             n_configs *= len(l)
    #         if n_configs > 1:
    #             trainable_models.add(model_name)

    #     for model_name in trainable_models:
    #         if self.evaluation_config.tuning_loss is None:
    #             raise ValueError(
    #                 f"Model '{model_name}' requires 'tuning_loss' parameter for training "
    #                 f"but it is not defined. Please add 'tuning_loss' to the model configuration."
    #             )
    #     return self

    # @model_validator(mode="after")
    # def validate_single_model_and_existence(self):
    #     # Check that only a single model is referenced in both model_configs and models_setting
    #     if len(self.model_configs) != 1:
    #         raise ValueError(
    #             "JobConfig must reference exactly one model in model_configs"
    #         )
    #     model_in_hparams = list(self.model_configs.keys())[0]

    #     if len(self.models_setting) != 1:
    #         raise ValueError("models_setting must contain exactly one model")
    #     model_in_settings = list(self.models_setting.keys())[0]

    #     if model_in_hparams != model_in_settings:
    #         raise ValueError(
    #             f"Model names do not match in JobConfig: model in model_configs is '{model_in_hparams}', "
    #             f"but in models_setting is '{model_in_settings}'"
    #         )

    #     model_path = Path(self.models_setting[model_in_hparams]["model_path"])
    #     model_file = model_path / f"{model_in_hparams}_model.py"
    #     if not (model_path.exists() and model_path.is_dir()):
    #         raise ValueError(f"Model directory does not exist: {model_path}")
    #     if not model_file.exists():
    #         raise ValueError(f"Model file does not exist: {model_file}")

    #     return self
