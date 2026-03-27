"""
Base model class that defines the interface for all traditional time series forecasting models.

This abstract base class provides a common interface for traditional statistical and machine learning
models used in time series forecasting. It handles configuration management, training, prediction,
evaluation, and model persistence.

All traditional models (ARIMA, LSTM, XGBoost, etc.) should inherit from this class and implement
the required abstract methods.
"""

import inspect
import os

from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from pydantic import BaseModel as PydanticBaseModel

from ..utils.configs import JobConfig
from ..pipeline.metric_registry import MetricRegistry


class BaseModel(ABC):
    """
    Abstract base class for traditional time series forecasting models.

    This class provides a unified interface for training, prediction, and evaluation
    of traditional time series forecasting models. It handles configuration management,
    data preprocessing, and evaluation metrics computation.

    Attributes:
        config: Configuration dictionary containing model and dataset parameters
        training_loss: Primary loss function for training
        forecast_horizon: Number of steps to forecast ahead
        is_fitted: Whether the model has been trained
        evaluator: MetricRegistry instance for computing metrics
    """

    def __init__(
        self,
        params: Dict[str, Any],
        settings: Dict[str, Any] | None = None,
        ParamsClass: PydanticBaseModel | None = None,
    ):
        """
        Initialize the base model with validated hyperparameters and runtime settings.

        Args:
            params: Raw hyperparameters chosen for the current training run.
            settings: Model-level execution configuration (device, seed, etc.). Defaults to empty dict.
            ParamsClass: Pydantic schema used to validate and coerce `params`.
        """
        super().__init__()

        # Setup parameters
        if settings is None: settings = {}

        self.params_class = ParamsClass

        self.metric_registry = MetricRegistry()
        self.set_params(**params)
        self.set_attrs(**settings)  # Settings

    @abstractmethod
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "BaseModel":
        """
        Train the model on given data.

        Args:
            y_context: Context window used to initialise the model before fitting.
            y_target: Segment used for supervised optimisation during tuning or evaluation.
            timestamps_context: Timestamp index aligned with `y_context`.
            timestamps_target: Timestamp index aligned with `y_target`.
            x_context: Optional covariate data aligned with `y_context`,
                shape (num_steps_context, num_covariates).
            x_target: Optional covariate data aligned with `y_target`,
                shape (num_steps_target, num_covariates).

        Returns:
            BaseModel: The fitted model instance.
        """
        pass

    @abstractmethod
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Generate predictions for the target time steps.

        Args:
            y_context (np.ndarray): Context window used for prediction initialization,
                shape (num_steps_context, num_targets).
            timestamps_context (np.ndarray): Timestamp index aligned with y_context,
                shape (num_steps_context,).
            timestamps_target (np.ndarray): Timestamp index for prediction targets,
                shape (num_steps_target,).
            x_context (Optional[np.ndarray]): Optional covariate data aligned with y_context,
                shape (num_steps_context, num_covariates).
            x_target (Optional[np.ndarray]): Optional covariate data aligned with timestamps_target,
                shape (num_steps_target, num_covariates).
            **kwargs (dict): Additional keyword arguments for model-specific prediction
                parameters (e.g., freq, num_samples for stochastic models).

        Returns:
            np.ndarray: Predicted values. Shape depends on model type:
                - Deterministic: (num_steps_target, num_targets)
                - Stochastic: (num_samples, num_steps_target, num_targets)
                - Hybrid: Tuple of (point_forecasts, samples)
        """
        pass

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics between true and predicted values using the MetricRegistry class.

        This method computes evaluation metrics as configured in evaluation.metrics

        Args:
            y_true: True target values (ndarray, shape [num_steps, num_features])
            y_pred: Predicted values (ndarray, shape [num_steps, num_features])

        Returns:
            Dict[str, float]: Dictionary of computed evaluation metrics (from evaluation.metrics)
        """
        return self.metric_registry.compute_metrics(
            y_true, y_pred, model_type=self.model_type, **kwargs  # type: ignore
        )

    def get_params(self):
        """
        Get the current model parameters.

        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return self.params

    def set_params(self, **params: Dict[str, Any]) -> "BaseModel":
        """
        Set model parameters.

        Args:
            **params: Model parameters to set

        Returns:
            self: The model instance with updated parameters
        """
        validated_params = self.params_class.model_validate(params)
        self.params = validated_params  # Store validated params for get_params() and model_dump()

        self.set_attrs(**validated_params.model_dump())
        self.is_fitted = False  # Mark as unfitted if parameters change
        return self

    @staticmethod
    def resolve_weights_path(hf_id: str) -> str:
        """Return a local FUSE path for ``hf_id`` if ``MODEL_WEIGHTS_PATH`` is set and
        the directory exists, otherwise return the original HuggingFace identifier."""
        root = os.environ.get("MODEL_WEIGHTS_PATH", "").strip()
        if not root:
            print(
                "[tempusbench-weights] MODEL_WEIGHTS_PATH unset; "
                f"using HuggingFace/repo id (may download from internet): {hf_id!r}",
                flush=True,
            )
            return hf_id
        local = Path(root) / hf_id
        if local.is_dir() and any(local.iterdir()):
            print(
                "[tempusbench-weights] Using local weights (e.g. GCS FUSE bucket), "
                f"not downloading from internet: root={root!r} hf_id={hf_id!r} -> {local!s}",
                flush=True,
            )
            return str(local)
        print(
            "[tempusbench-weights] MODEL_WEIGHTS_PATH set but snapshot missing or empty; "
            f"falling back to HuggingFace id (may use internet): root={root!r} hf_id={hf_id!r} "
            f"expected_dir={local!s}",
            flush=True,
        )
        return hf_id

    def set_attrs(self, **attrs: Dict[str, Any]):
        """
        Map validated settings onto the instance for ergonomic access.

        Args:
            **attrs: Arbitrary attributes sourced from the settings dictionary.
        """
        # Validate that provided setting names do not clash with existing attributes or parameter keys
        reserved_keys = set(self.__dict__.keys())

        attr_keys = set(attrs.keys())
        clashing_with_reserved = attr_keys & reserved_keys
        clashing_keys = sorted(list(clashing_with_reserved))
        if clashing_keys:
            raise ValueError(
                f"Setting names clash with reserved or parameter keys: {clashing_keys}. "
                f"Please rename these settings."
            )

        if "hf_model_name" in attrs and attrs["hf_model_name"]:
            attrs["hf_model_name"] = self.resolve_weights_path(attrs["hf_model_name"])

        self.settings = attrs
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's properties and performance.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            "model_type": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "parameters": self.get_params(),
        }

def validate_covariate_support(
    x_context: Optional[np.ndarray],
    x_target: Optional[np.ndarray],
    supports_past_only: bool,
    supports_future_only: bool,
    supports_both: bool,
    model_name: str,
) -> None:
    """
    Raise ValueError when covariates are provided in an unsupported configuration.

    Args:
        x_context: Past covariate data (None if not provided).
        x_target: Future covariate data (None if not provided).
        supports_past_only: Model can use x_context alone.
        supports_future_only: Model can use x_target alone.
        supports_both: Model can use x_context and x_target together.
        model_name: Model name for error messages.

    Raises:
        ValueError: When an unsupported covariate combination is passed.
    """
    has_past = x_context is not None
    has_future = x_target is not None

    if not has_past and not has_future:
        return

    if has_past and not has_future:
        if not supports_past_only:
            raise ValueError(
                f"{model_name} does not support past covariates (x_context) only. "
                "Do not pass x_context without x_target."
            )
        return

    if has_future and not has_past:
        if not supports_future_only:
            raise ValueError(
                f"{model_name} does not support future covariates (x_target) only. "
                "Do not pass x_target without x_context."
            )
        return

    if has_past and has_future:
        if not supports_both:
            raise ValueError(
                f"{model_name} does not support both past and future covariates "
                "(x_context and x_target) together. "
                "Use x_context only, or do not pass covariates."
            )


def validate_inputs(func):
    """
    Decorator to validate input shapes for train/predict methods.

    Shape convention (all models, foundation and non-foundation):
    - y_context, y_target: 2D arrays (num_steps, num_targets) with num_targets >= 1
    - timestamps_context, timestamps_target: 1D arrays (num_steps,)
    - x_context, x_target: Optional 2D arrays (num_steps, num_covariates)
    - Matching dimensions between related parameters

    Models must convert to their library's internal format (e.g. HF PatchTSMixer
    expects (batch, seq_len, channels); MOMENT expects (batch, channels, seq_len)).
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get function signature to map args to parameter names
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments

        # Extract the relevant parameters
        y_context = params.get("y_context")
        y_target = params.get("y_target")
        timestamps_context = params.get("timestamps_context")
        timestamps_target = params.get("timestamps_target")
        x_context = params.get("x_context")
        x_target = params.get("x_target")

        # Validate y_context (required parameter)
        if y_context is None:
            raise ValueError("y_context cannot be None")

        if not isinstance(y_context, np.ndarray):
            raise TypeError(f"y_context must be np.ndarray, got {type(y_context)}")

        if y_context.ndim != 2:
            raise ValueError(
                f"y_context must be 2D array, got {y_context.ndim}D with shape {y_context.shape}"
            )

        num_steps_context, num_targets_context = y_context.shape
        if num_steps_context == 0 or num_targets_context == 0:
            raise ValueError(
                f"y_context cannot have zero dimensions, got shape {y_context.shape}"
            )

        # Validate y_target if present
        if y_target is not None:
            if not isinstance(y_target, np.ndarray):
                raise TypeError(f"y_target must be np.ndarray, got {type(y_target)}")

            if y_target.ndim != 2:
                raise ValueError(
                    f"y_target must be 2D array, got {y_target.ndim}D with shape {y_target.shape}"
                )

            num_steps_target, num_targets_target = y_target.shape
            if num_steps_target == 0 or num_targets_target == 0:
                raise ValueError(
                    f"y_target cannot have zero dimensions, got shape {y_target.shape}"
                )

            # Check that num_targets match
            if num_targets_target != num_targets_context:
                raise ValueError(
                    f"y_target must have same num_targets as y_context: "
                    f"expected {num_targets_context}, got {num_targets_target}"
                )

        # Validate timestamps_context if present
        if timestamps_context is not None:
            if not isinstance(timestamps_context, np.ndarray):
                raise TypeError(
                    f"timestamps_context must be np.ndarray, got {type(timestamps_context)}"
                )

            if timestamps_context.ndim != 1:
                raise ValueError(
                    f"timestamps_context must be 1D array, got {timestamps_context.ndim}D "
                    f"with shape {timestamps_context.shape}"
                )

            # Match with y_context
            if len(timestamps_context) != num_steps_context:
                raise ValueError(
                    f"timestamps_context length must match y_context num_steps: "
                    f"expected {num_steps_context}, got {len(timestamps_context)}"
                )

        # Validate timestamps_target if present
        if timestamps_target is not None:
            if not isinstance(timestamps_target, np.ndarray):
                raise TypeError(
                    f"timestamps_target must be np.ndarray, got {type(timestamps_target)}"
                )

            if timestamps_target.ndim != 1:
                raise ValueError(
                    f"timestamps_target must be 1D array, got {timestamps_target.ndim}D "
                    f"with shape {timestamps_target.shape}"
                )

            if len(timestamps_target) == 0:
                raise ValueError("timestamps_target cannot be empty")

            # Match with y_target if both present
            if y_target is not None:
                num_steps_target = y_target.shape[0]
                if len(timestamps_target) != num_steps_target:
                    raise ValueError(
                        f"timestamps_target length must match y_target num_steps: "
                        f"expected {num_steps_target}, got {len(timestamps_target)}"
                    )

        # Validate x_context if present
        if x_context is not None:
            if not isinstance(x_context, np.ndarray):
                raise TypeError(
                    f"x_context must be np.ndarray, got {type(x_context)}"
                )

            if x_context.ndim != 2:
                raise ValueError(
                    f"x_context must be 2D array, got {x_context.ndim}D "
                    f"with shape {x_context.shape}"
                )

            # Match with y_context num_steps
            if x_context.shape[0] != num_steps_context:
                raise ValueError(
                    f"x_context num_steps must match y_context num_steps: "
                    f"expected {num_steps_context}, got {x_context.shape[0]}"
                )

        # Validate x_target if present
        if x_target is not None:
            if not isinstance(x_target, np.ndarray):
                raise TypeError(
                    f"x_target must be np.ndarray, got {type(x_target)}"
                )

            if x_target.ndim != 2:
                raise ValueError(
                    f"x_target must be 2D array, got {x_target.ndim}D "
                    f"with shape {x_target.shape}"
                )

            # Match with y_target num_steps if y_target present, otherwise with timestamps_target
            if y_target is not None:
                num_steps_target = y_target.shape[0]
                if x_target.shape[0] != num_steps_target:
                    raise ValueError(
                        f"x_target num_steps must match y_target num_steps: "
                        f"expected {num_steps_target}, got {x_target.shape[0]}"
                    )
            elif timestamps_target is not None:
                if x_target.shape[0] != len(timestamps_target):
                    raise ValueError(
                        f"x_target num_steps must match timestamps_target length: "
                        f"expected {len(timestamps_target)}, got {x_target.shape[0]}"
                    )

            # Check that x_context and x_target have matching num_covariates if both present
            if x_context is not None:
                if x_target.shape[1] != x_context.shape[1]:
                    raise ValueError(
                        f"x_target must have same num_covariates as x_context: "
                        f"expected {x_context.shape[1]}, got {x_target.shape[1]}"
                    )

        # Call the original function
        return func(self, *args, **kwargs)

    return wrapper
