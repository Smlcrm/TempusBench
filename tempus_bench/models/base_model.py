"""
Base model class that defines the interface for all traditional time series forecasting models.

This abstract base class provides a common interface for traditional statistical and machine learning
models used in time series forecasting. It handles configuration management, training, prediction,
evaluation, and model persistence.

All traditional models (ARIMA, LSTM, XGBoost, etc.) should inherit from this class and implement
the required abstract methods.
"""

import inspect

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, Optional

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
        if settings is None:
            settings = {}

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
        **kwargs: dict,
    ) -> "BaseModel":
        """
        Train the model on given data.

        Args:
            y_context: Context window used to initialise the model before fitting.
            y_target: Segment used for supervised optimisation during tuning or evaluation.
            timestamps_context: Timestamp index aligned with `y_context`.
            timestamps_target: Timestamp index aligned with `y_target`.

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

        self.set_attrs(**validated_params.model_dump())
        self.is_fitted = False  # Mark as unfitted if parameters change
        return self

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


def validate_inputs(func):
    """
    Decorator to validate input shapes for train/predict methods.

    Validates:
    - y_context, y_target: 2D arrays (num_steps, num_targets) with num_targets >= 1
    - timestamps_context, timestamps_target: 1D arrays (num_steps,)
    - Matching dimensions between related parameters
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

        # Call the original function
        return func(self, *args, **kwargs)

    return wrapper
