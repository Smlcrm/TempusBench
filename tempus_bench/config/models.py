"""
Pydantic models for configuration validation and type safety.

This module defines all configuration models using Pydantic for comprehensive
validation, type checking, and documentation of the benchmarking pipeline.
"""

from typing import Dict, List, Optional, Any, Literal, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import yaml
import numpy as np


class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""

    model_config = ConfigDict(extra="forbid")

    tuning_loss: str = Field(
        ...,
        description="Primary metric for hyperparameter optimization"
    )
    max_windows: int = Field(
        ...,
        ge=1,
        description="Maximum number of rolling windows to generate for evaluation"
    )
    max_num_variates: float = Field(
        default=float('inf'),
        description="Maximum number of variates to extract from dataset for evaluation (inf for all)"
    )

    @field_validator('max_num_variates')
    @classmethod
    def validate_max_num_variates(cls, v):
        """Validate max_num_variates is either inf or a positive number."""
        if v == float('inf'):
            return v
        if isinstance(v, (int, float)) and v < 1:
            raise ValueError("max_num_variates must be at least 1 or inf")
        return v
    num_samples: int = Field(
        default=100,
        ge=1,
        description="Number of samples to generate for stochastic metrics"
    )
    num_quantiles: int = Field(
        default=10,
        ge=1,
        description="Number of quantiles to compute for quantile-based metrics"
    )
    point_forecast_statistic: str = Field(
        default="mean",
        description="Statistic to use for converting stochastic predictions to point forecasts"
    )


class ModelConfig(BaseModel):
    """Model configuration model."""

    model_config = ConfigDict(extra="forbid")

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

    @field_validator('*', mode='before')
    @classmethod
    def validate_model_parameters(cls, v, info):
        """Validate model parameters structure."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError(f"Model parameters must be a dict, got {type(v).__name__}")

        # For foundation models, parameters should be simple key-value pairs
        foundation_models = {
            'chronos', 'deepar', 'tiny_time_mixer', 'moirai', 'moirai_moe',
            'moment', 'timesfm', 'lagllama', 'toto', 'tabpfn'
        }

        if info.field_name in foundation_models:
            # Foundation models can have any parameters but should be simple values
            return v
        else:
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


class BenchmarkConfig(BaseModel):
    """Root configuration model for the benchmarking pipeline."""

    model_config = ConfigDict(extra="forbid")

    task_path: str = Field(..., description="Task path pattern (supports wildcards like '*', 'univariate/*', etc.)")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")
    model: ModelConfig = Field(..., description="Model configuration")


class DatasetConfig(BaseModel):
    """Dataset configuration model for individual task folders."""

    model_config = ConfigDict(extra="forbid")

    handle_missing: Literal[
        'interpolate', 'mean', 'median', 'drop', 'forward_fill', 'backward_fill'
    ] = Field(
        default='interpolate',
        description="Strategy for handling missing values"
    )
    name: str = Field(
        ...,
        description="Dataset name"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the data"
    )


class TaskConfig(BaseModel):
    """Task configuration model for individual task folders."""

    model_config = ConfigDict(extra="forbid")

    forecast_horizon: int = Field(
        ...,
        ge=1,
        le=128,
        description="Number of steps to forecast ahead (max 128)"
    )
    context_window: int = Field(
        ...,
        ge=1,
        description="Number of context steps for training"
    )
    dataset: DatasetConfig = Field(
        ...,
        description="Dataset configuration for this task"
    )


class ModelSettingsConfig(BaseModel):
    """Model settings configuration model."""

    model_config = ConfigDict(extra="forbid")

    python_version: str = Field(
        default="3.11",
        description="Python version for conda environment"
    )
    device: Literal["cpu", "gpu"] = Field(
        default="cpu",
        description="Device to use for model execution"
    )


class SystemsConfig(BaseModel):
    """Systems configuration model."""

    model_config = ConfigDict(extra="forbid")

    logging_format: str = Field(
        ...,
        description="Log message format"
    )
    file_logging: bool = Field(
        ...,
        description="Enable file logging"
    )
    console_logging: bool = Field(
        ...,
        description="Enable console logging"
    )
    tensorboard_logging: bool = Field(
        ...,
        description="Enable TensorBoard logging"
    )
    runs_dir: str = Field(
        ...,
        description="Directory for storing run outputs"
    )
    conda_env_prefix: str = Field(
        ...,
        description="Prefix for conda environment names"
    )