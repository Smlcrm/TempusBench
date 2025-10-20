"""
Configuration validation utility for the benchmarking pipeline.

This module provides comprehensive validation of configuration files using Pydantic
to ensure they comply with the expected schema before execution.
"""

import os
import yaml
import logging

from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, constr

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class Dataset(BaseModel):
    """Dataset configuration model."""
    # Use constr with regex that matches *, anything ending in /*, or univariate/multivariate paths
    name: constr(pattern=r'^(\*|.*\/\*|(univariate|multivariate)\/.+)$') = Field(..., description="Dataset name or path pattern")
    normalize: bool = Field(..., description="Whether to normalize the data")
    handle_missing: Literal['interpolate', 'mean', 'median', 'drop', 'forward_fill', 'backward_fill'] = Field(
        ..., description="Strategy for handling missing values"
    )

class Evaluation(BaseModel):
    """Evaluation configuration model."""
    metrics: List[str] = Field(..., min_items=1, description="List of evaluation metrics")
    logging: Optional[bool] = Field(default=False, description="Enable logging")

class Task(BaseModel):
    """Task configuration model."""

    task_type: Literal['deterministic', 'stochastic'] = Field(..., description="Task type")
    forecast_horizon: int = Field(..., ge=1, le=128, description="Forecast horizon (max 128)")
    context_window: int = Field(..., ge=1, description="Context window size")
    max_windows: int = Field(..., ge=1, description="Maximum number of windows to generate")
    max_num_variates: int = Field(..., ge=1, description="Maximum number of variates to extract from dataset")
    tuning_loss: str = Field(..., description="Single metric for hyperparameter tuning, must be compatible with task type")
    dataset: Dataset = Field(..., description="Dataset configuration")

    ALLOWED_METRICS: ClassVar[Dict[str, List[str]]] = {
        'deterministic': ['mae', 'rmse', 'mape', 'mase'],
        'stochastic': ['crps', 'quantile_loss', 'interval_score', 'mae', 'rmse']
    }

    @model_validator(mode='after')
    def validate_task_consistency(self):
        if not self.tuning_loss in Task.ALLOWED_METRICS.get(self.task_type):
            raise ValueError(
                f"Invalid tuning_loss '{self.tuning_loss}' for {self.task_type} task_type. "
                f"Allowed metrics: {self.ALLOWED_METRICS[self.task_type]}"
            )
        return self

class BenchmarkConfig(BaseModel):
    """Root configuration model for the benchmarking pipeline."""

    task: Task = Field(..., description="Task configuration")
    evaluation: Evaluation = Field(..., description="Evaluation configuration")
    model: Dict[str, Optional[Dict[str, Any]]] = Field(..., description="Model configurations")

    @model_validator(mode='after')
    def validate_evaluation_consistency(self):
        """Validate evaluation metrics and tuning_loss for consistency and task type."""
        task_type = self.task.task_type

        # Validate evaluation metrics
        allowed_metrics = Task.ALLOWED_METRICS.get(task_type, [])
        invalid_metrics = [m for m in self.evaluation.metrics if m not in allowed_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics for {task_type} task_type: {invalid_metrics}. "
                f"Allowed: {Task.ALLOWED_METRICS[task_type]}"
            )

        # Ensure tuning_loss is present in metrics list
        if self.task.tuning_loss not in self.evaluation.metrics:
            raise ValueError(
                f"tuning_loss '{self.task.tuning_loss}' is not present in evaluation.metrics list."
            )

        return self

    @field_validator('model')
    @classmethod
    def validate_model_configuration(cls, v):
        """Validate model names and parameters structure."""
        # Validate model names exist as subdirectories
        current_dir = Path(__file__).parent
        models_dir = current_dir.parent / 'models'

        if not models_dir.exists():
            raise ValueError(f"Models directory not found: {models_dir}")

        # Collect all model folders under any subfolder of models/
        valid_models = set()
        for subdir in models_dir.iterdir():
            if subdir.is_dir():
                for model_folder in subdir.iterdir():
                    if model_folder.is_dir():
                        valid_models.add(model_folder.name)

        # Validate model names
        invalid_models = [name for name in v.keys() if name not in valid_models]
        if invalid_models:
            raise ValueError(
                f"Invalid model names: {invalid_models}. "
                f"Valid models: {sorted(valid_models)}")

        # Validate model parameters structure
        for model_name, model_params in v.items():
            if model_params is not None and not isinstance(model_params, dict):
                raise ValueError(
                    f"Model parameters for '{model_name}' must be a dict or None, "
                    f"got {type(model_params).__name__}")

            if isinstance(model_params, dict):
                for param_name, param_val in model_params.items():
                    if not isinstance(param_val, list):
                        raise ValueError(
                            f"Parameter '{param_name}' for model '{model_name}' must be a list of values, "
                            f"got {type(param_val).__name__}")

                    # Additional validation: ensure list is not empty if provided
                    if len(param_val) == 0:
                        raise ValueError(
                            f"Parameter '{param_name}' for model '{model_name}' cannot be an empty list. "
                            f"Provide a list with at least one value or set the model to None.")

        return v

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file and return as dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ConfigValidationError("Configuration file is empty or invalid YAML")

        return config

    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML format: {e}")

def _convert_pydantic_errors(validation_error: ValidationError) -> str:
    """Convert Pydantic validation errors to a readable string."""
    error_messages = []
    for error in validation_error.errors():
        field_path = " -> ".join(str(loc) for loc in error['loc'])
        error_messages.append(f"{field_path}: {error['msg']}")
    return "; ".join(error_messages)

def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    config = load_config(config_path)

    try:
        # Validate using Pydantic model
        BenchmarkConfig(**config)
        logger.info(f"Configuration validation passed for {config_path}")
        return True

    except ValidationError as e:
        error_msg = _convert_pydantic_errors(e)
        logger.error(f"Configuration validation failed: {error_msg}")
        raise ConfigValidationError(error_msg)

