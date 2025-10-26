"""
Pydantic models for configuration validation and type safety.

This module defines all configuration models using Pydantic for comprehensive
validation, type checking, and documentation of the benchmarking pipeline.
"""

from typing import Dict, List, Optional, Any, Literal, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class DatasetConfig(BaseModel):
    """Dataset configuration model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Dataset name or path pattern (supports wildcards like '*', 'univariate/*', etc.)"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the data"
    )
    handle_missing: Literal[
        'interpolate', 'mean', 'median', 'drop', 'forward_fill', 'backward_fill'
    ] = Field(
        default='interpolate',
        description="Strategy for handling missing values"
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""

    model_config = ConfigDict(extra="forbid")

    metrics: List[str] = Field(
        ...,
        min_length=1,
        description="List of evaluation metrics to compute"
    )
    tuning_loss: str = Field(
        ...,
        description="Primary metric for hyperparameter optimization"
    )
    max_windows: int = Field(
        ...,
        ge=1,
        description="Maximum number of rolling windows to generate for evaluation"
    )
    max_num_variates: int = Field(
        ...,
        ge=1,
        description="Maximum number of variates to extract from dataset for evaluation"
    )
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


class TaskConfig(BaseModel):
    """Task configuration model."""

    model_config = ConfigDict(extra="forbid")

    task_type: Literal['deterministic', 'stochastic'] = Field(
        ...,
        description="Type of forecasting task"
    )
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
        description="Dataset configuration"
    )

    # Allowed metrics for each task type
    ALLOWED_METRICS: Dict[str, List[str]] = {
        'deterministic': ['mae', 'rmse', 'mape', 'mase'],
        'stochastic': ['crps', 'quantile_score', 'weighted_interval_score', 'mae', 'rmse']
    }

    @model_validator(mode='after')
    def validate_task_consistency(self):
        """Validate task configuration consistency."""
        # Task validation is now simpler since tuning_loss moved to evaluation
        return self


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


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    model_config = ConfigDict(extra="forbid")

    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        description="Logging level"
    )
    format: str = Field(
        default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        description="Log message format"
    )
    file_logging: bool = Field(
        default=True,
        description="Enable file logging"
    )
    console_logging: bool = Field(
        default=True,
        description="Enable console logging"
    )
    tensorboard_logging: bool = Field(
        default=True,
        description="Enable TensorBoard logging"
    )


class PathsConfig(BaseModel):
    """Paths configuration model."""

    model_config = ConfigDict(extra="forbid")

    datasets_dir: str = Field(
        default="tempus_bench/datasets",
        description="Directory containing datasets"
    )
    runs_dir: str = Field(
        default="runs",
        description="Directory for storing run outputs"
    )
    logs_dir: str = Field(
        default="logs",
        description="Directory for storing log files"
    )
    models_dir: str = Field(
        default="tempus_bench/models",
        description="Directory containing model implementations"
    )
    configs_dir: str = Field(
        default="tempus_bench/configs",
        description="Directory containing configuration files"
    )


class SystemConfig(BaseModel):
    """System configuration model."""

    model_config = ConfigDict(extra="forbid")

    python_version: str = Field(
        default="3.11",
        description="Python version for conda environments"
    )
    conda_env_prefix: str = Field(
        default="benchmark",
        description="Prefix for conda environment names"
    )
    max_parallel_jobs: int = Field(
        default=1,
        ge=1,
        description="Maximum number of parallel jobs"
    )
    memory_limit_gb: Optional[int] = Field(
        default=None,
        ge=1,
        description="Memory limit in GB for model execution"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Timeout in seconds for model execution"
    )


class BenchmarkConfig(BaseModel):
    """Root configuration model for the benchmarking pipeline."""

    model_config = ConfigDict(extra="forbid")

    task: TaskConfig = Field(..., description="Task configuration")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    paths: PathsConfig = Field(default_factory=PathsConfig, description="Paths configuration")
    system: SystemConfig = Field(default_factory=SystemConfig, description="System configuration")

    @model_validator(mode='after')
    def validate_evaluation_consistency(self):
        """Validate evaluation metrics and tuning_loss for consistency."""
        task_type = self.task.task_type

        # Define allowed metrics for each task type
        ALLOWED_METRICS = {
            'deterministic': ['mae', 'rmse', 'mape', 'mase'],
            'stochastic': ['crps', 'quantile_score', 'weighted_interval_score', 'mae', 'rmse']
        }

        allowed_metrics = ALLOWED_METRICS.get(task_type, [])

        # Validate evaluation metrics
        invalid_metrics = [m for m in self.evaluation.metrics if m not in allowed_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics for {task_type} task_type: {invalid_metrics}. "
                f"Allowed: {allowed_metrics}"
            )

        # Ensure tuning_loss is present in metrics list
        if self.evaluation.tuning_loss not in self.evaluation.metrics:
            raise ValueError(
                f"tuning_loss '{self.evaluation.tuning_loss}' is not present in evaluation.metrics list."
            )

        return self
