"""
Pydantic models for configuration validation and type safety.

This module defines all configuration models using Pydantic for comprehensive
validation, type checking, and documentation of the benchmarking pipeline.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from tempus_bench.utils.paths import get_models_dir

class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""

    model_config = ConfigDict(extra="forbid")

    tuning_loss: Optional[Literal["mae", "mase", "mape", "rmse"]] = Field(
        default="mae",
        description="Tuning loss for trainable models such as ARIMA, LSTM, DeepAR, SVR. "
                    "Only deterministic (point) metrics are allowed: mae, mase, mape, rmse."
    )
    max_windows: int = Field(
        ...,
        ge=1,
        description="Maximum number of rolling windows to generate for evaluation"
    )
    max_num_variates: int = Field(
        default=None,
        description="Maximum number of variates to extract from dataset for evaluation (inf for all)"
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

    @field_validator('max_num_variates')
    @classmethod
    def validate_max_num_variates(cls, v):
        """Validate max_num_variates is either inf or a positive number."""
        if v == float('inf'):
            return v
        if isinstance(v, (int, float)) and v < 1:
            raise ValueError("max_num_variates must be at least 1 or inf")
        return v

    @field_validator('tuning_loss')
    @classmethod
    def validate_tuning_loss(cls, v):
        """Validate that tuning_loss is a deterministic metric."""
        allowed_metrics = {"mae", "mse", "rmse", "smape", "mape", "mdape", "medae", "r2"}
        if v is not None and v not in allowed_metrics:
            raise ValueError(f"tuning_loss must be one of: {', '.join(sorted(allowed_metrics))}")
        return v


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
    varmax: Optional[Dict[str, List[Any]]] = None

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
        # For foundation models, parameters should be simple key-value pairs
        foundation_models = {
            'chronos', 'deepar', 'tiny_time_mixer', 'moirai', 'moirai_moe',
            'moment', 'timesfm', 'lagllama', 'toto', 'tabpfn'
        }

        """Validate model parameters structure."""
        if v is None: return v

        if not isinstance(v, dict):
            raise ValueError(f"Model parameters must be a dict, got {type(v).__name__}")

        if info.field_name in foundation_models:
            # Foundation models must have no parameters
            if v:  # v is a dict and must be empty
                raise ValueError(
                    f"Foundation model '{info.field_name}' must not define any parameters, "
                    f"but got: {v}"
                )
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

    @model_validator(mode='after')
    def validate_tuning_loss_usage(self):
        """
        Ensure that 'tuning_loss' is only defined for models that actually use it
        for training/optimization, not for inference or evaluation.
        """
        # Only these models implement training procedures depending on tuning_loss
        tuning_loss_training_models = {'arima', 'lstm', 'deepar', 'svr'}

        for model_name, model_config in self.model_dump().items():
            if model_config is None: continue

            if 'tuning_loss' in model_config:
                if model_name not in tuning_loss_training_models:
                    raise ValueError(
                        f"Model '{model_name}' cannot define 'tuning_loss'. Only models that support "
                        f"loss-based training with a selectable loss (ARIMA, LSTM, DeepAR, SVR) "
                        f"may define 'tuning_loss'."
                    )
        return self

class BenchmarkConfig(BaseModel):
    """Root configuration model for the benchmarking pipeline."""

    model_config = ConfigDict(extra="forbid")

    task_path: str = Field(..., description="Task path pattern (supports wildcards like '*', 'univariate/*', etc.)")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")
    model: ModelConfig = Field(..., description="Model configuration")

    @model_validator(mode='after')
    def validate_tuning_loss_model_compatibility(self):
        """
        Validate that tuning losses are compatible with the models being run.

        Rules:
        1. If only deterministic_tuning_loss is defined, it must work with deterministic models
        2. If only stochastic_tuning_loss is defined, it must work with stochastic models
        3. If stochastic_tuning_loss is defined and only deterministic models are in the config,
            then deterministic_tuning_loss must also be defined
        """
        # Get models directory
        models_dir = get_models_dir()

        # Categorize models based on their folder location
        deterministic_models = set()

        # Scan deterministic directory
        deterministic_dir = models_dir / 'deterministic'
        if deterministic_dir.exists():
            for model_folder in deterministic_dir.iterdir():
                if model_folder.is_dir():
                    # Check if model file exists
                    model_file = model_folder / f"{model_folder.name}_model.py"
                    if model_file.exists():
                        deterministic_models.add(model_folder.name)

        # Get models defined in the config (non-None models)
        defined_models = [name for name, config in self.model.model_dump().items() if config is not None]

        # Categorize defined models
        has_deterministic = any(model in deterministic_models for model in defined_models)
        # Validation logic
        if has_deterministic and self.evaluation.deterministic_tuning_loss is None:
            raise ValueError(
                f"Configuration requires 'deterministic_tuning_loss' (models: {defined_models}) "
                f"but it is not specified. Please set 'deterministic_tuning_loss' when using deterministic models."
            )

        return self

class DatasetConfig(BaseModel):
    """Dataset configuration model for individual task folders."""

    model_config = ConfigDict(extra="forbid")

    handle_missing: Literal[
        'interpolate', 'mean', 'median', 'drop', 'forward_fill', 'backward_fill'
    ] = Field(
        default='interpolate',
        description="Strategy for handling missing values"
    )
    file_name: str = Field(
        ...,
        description="Dataset file name"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the data"
    )

class TaskConfig(BaseModel):
    """Task configuration model for individual task folders."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Task name (must match folder name)"
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

class BenchmarkSettingsConfig(BaseModel):
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

class UnifiedConfig(BaseModel):
    """Unified configuration model for the benchmarking pipeline (single-model only)."""

    model_config = ConfigDict(extra="forbid")

    benchmark_config: BenchmarkConfig = Field(..., description="Benchmark configuration (must reference a single model)")
    benchmark_settings: BenchmarkSettingsConfig = Field(..., description="Benchmark settings")
    model_settings: Dict[str, ModelSettingsConfig] = Field(..., description="Model execution settings (must be for a single model)")
    task_configs: Dict[str, List[TaskConfig]] = Field(..., description="Task configurations")

    @model_validator(mode="after")
    def validate_single_model_and_existence(cls, values):
        benchmark_config = values.get('benchmark_config')
        model_settings = values.get('model_settings')

        # Check that only a single model is referenced in both places
        # model_in_benchmark is an instance of ModelConfig; extract the list of models with non-None config
        model_config = benchmark_config.model.model_dump()
        if len(model_config) != 1:
            raise ValueError("benchmark_config must reference exactly one model")
        model_in_benchmark = list(model_config.keys())[0]

        if len(model_settings) != 1:
            raise ValueError("model_settings must contain exactly one model")
        model_in_settings = list(model_settings.keys())[0]

        if model_in_benchmark != model_in_settings:
            raise ValueError(
                f"Model names do not match in UnifiedConfig: model in benchmark_config is '{model_in_benchmark}', "
                f"but in model_settings is '{model_in_settings}'"
            )

        models_dir = get_models_dir()
        model_dir = models_dir / model_in_benchmark
        model_file = model_dir / f"{model_in_benchmark}_model.py"
        if not (model_dir.exists() and model_dir.is_dir()):
            raise ValueError(f"Model directory does not exist: {model_dir}")
        if not model_file.exists():
            raise ValueError(f"Model file does not exist: {model_file}")

        return values