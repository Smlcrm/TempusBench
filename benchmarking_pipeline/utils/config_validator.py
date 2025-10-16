"""
Configuration validation utility for the benchmarking pipeline.

This module provides comprehensive validation of configuration files using Pydantic
to ensure they comply with the expected schema before execution.
"""

import os
import re
import yaml
import logging

from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError, constr

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    def __init__(self, message: str, field_path: str = "", details: str = ""):
        self.message = message
        self.field_path = field_path
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.field_path:
            return f"Config validation error at '{self.field_path}': {self.message}"
        return f"Config validation error: {self.message}"

class Dataset(BaseModel):
    """Dataset configuration model."""
    # Use constr with regex that matches *, anything ending in /*, or univariate/multivariate paths
    name: constr(pattern=r'^(\*|.*\/\*|(univariate|multivariate)\/.+)$') = Field(..., description="Dataset name or path pattern")
    normalize: bool = Field(..., description="Whether to normalize the data")
    handle_missing: Literal['interpolate', 'mean', 'median', 'drop', 'forward_fill', 'backward_fill'] = Field(
        ..., description="Strategy for handling missing values"
    )
    chunks: Optional[int] = Field(default=1, ge=1, description="Number of data chunks")

class Evaluation(BaseModel):
    """Evaluation configuration model."""
    metrics: List[str] = Field(..., min_items=1, description="List of evaluation metrics")
    tensorboard: Optional[bool] = Field(default=False, description="Enable TensorBoard logging")

class Task(BaseModel):
    """Task configuration model."""

    type: Literal['deterministic', 'stochastic'] = Field(..., description="Task type")
    forecast_horizon: int = Field(..., ge=1, description="Forecast horizon")
    context_window: int = Field(..., ge=1, description="Context window size")
    dataset: Dataset = Field(..., description="Dataset configuration")
    evaluation: Evaluation = Field(..., description="Evaluation configuration")

    @field_validator('evaluation')
    @classmethod
    def validate_evaluation_metrics(cls, v, info):
        """Validate evaluation metrics based on task type."""
        # Get task type from the parent data
        if info.data and 'type' in info.data:
            task_type = info.data['type']
            allowed_metrics = {
                'deterministic': ['mae', 'rmse', 'mape', 'smape', 'mase'],
                'stochastic': ['crps', 'quantile_loss', 'interval_score', 'mae', 'rmse']
            }

            if task_type in allowed_metrics:
                invalid_metrics = [m for m in v.metrics if m not in allowed_metrics[task_type]]
                if invalid_metrics:
                    raise ValueError(
                        f"Invalid metrics for {task_type} task: {invalid_metrics}. "
                        f"Allowed: {allowed_metrics[task_type]}")
        return v


class BenchmarkConfig(BaseModel):
    """Root configuration model for the benchmarking pipeline."""

    task: Task = Field(..., description="Task configuration")
    model: Dict[str, Optional[Dict[str, Any]]] = Field(..., description="Model configurations")

    @field_validator('model')
    @classmethod
    def validate_model_names(cls, v):
        """Validate that model names match folders in anyvariate directory."""
        # Get the path to the anyvariate models directory
        current_dir = Path(__file__).parent
        anyvariate_dir = current_dir.parent / 'models' / 'anyvariate'

        if not anyvariate_dir.exists():
            raise ValueError(f"Anyvariate models directory not found: {anyvariate_dir}")

        # Get list of valid model names from directory structure
        valid_models = [d.name for d in anyvariate_dir.iterdir() if d.is_dir()]

        # Validate each model name
        invalid_models = [name for name in v.keys() if name not in valid_models]
        if invalid_models:
            raise ValueError(
                f"Invalid model names: {invalid_models}. "
                f"Valid models: {sorted(valid_models)}")

        return v

    @field_validator('model')
    @classmethod
    def validate_model_parameters(cls, v):
        """Validate model parameters structure."""
        for model_name, model_params in v.items():
            if model_params is not None and not isinstance(model_params, dict):
                raise ValueError(
                    f"Model parameters for '{model_name}' must be a dict or None, "
                    f"got {type(model_params).__name__}")
        return v

class ConfigValidator:
    """
    Validates configuration files using Pydantic models.

    This class provides backward compatibility with the old ConfigValidator interface
    while using Pydantic for validation under the hood.
    """

    def __init__(self):
        """Initialize the validator."""
        pass

    def validate_config(self, config: Dict[str, Any], config_path: str = "") -> bool:
        """
        Validate a configuration dictionary against the schema.

        Args:
            config: Configuration dictionary to validate
            config_path: Path to the config file (for error reporting)

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            BenchmarkConfig(**config)
            logger.info(f"Configuration validation passed for {config_path or 'config'}")
            return True
        except ValidationError as e:
            # Convert Pydantic validation errors to ConfigValidationError
            error_messages = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error['loc'])
                error_messages.append(f"{field_path}: {error['msg']}")

            error_msg = "; ".join(error_messages)
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ConfigValidationError(error_msg, config_path)

    def validate_config_file(self, config_path: str) -> bool:
        """
        Load and validate a configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            True if validation passes

        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ConfigValidationError("Configuration file is empty or invalid YAML")

            return self.validate_config(config, config_path)

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML format: {e}")


def validate_config_file(config_path: str) -> bool:
    """
    Convenience function to validate a config file.

    Args:
        config_path: Path to the configuration file

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    validator = ConfigValidator()
    return validator.validate_config_file(config_path)


def validate_config_dict(config: Dict[str, Any]) -> bool:
    """
    Convenience function to validate a config dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator()
    return validator.validate_config(config)


def load_and_validate_config(config_path: str) -> BenchmarkConfig:
    """
    Load and validate a configuration file, returning the validated Pydantic model.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated BenchmarkConfig instance

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ConfigValidationError("Configuration file is empty or invalid YAML")

        # Validate using Pydantic model and return it directly
        return BenchmarkConfig(**config)

    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML format: {e}")
    except ValidationError as e:
        # Convert Pydantic validation errors to ConfigValidationError
        error_messages = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_messages.append(f"{field_path}: {error['msg']}")

        error_msg = "; ".join(error_messages)
        raise ConfigValidationError(error_msg, config_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        validate_config_file(config_file)
        print(f"✅ Configuration file '{config_file}' is valid!")
        sys.exit(0)
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)