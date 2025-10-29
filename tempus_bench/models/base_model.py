"""
Base model class that defines the interface for all traditional time series forecasting models.

This abstract base class provides a common interface for traditional statistical and machine learning
models used in time series forecasting. It handles configuration management, training, prediction,
evaluation, and model persistence.

All traditional models (ARIMA, LSTM, XGBoost, etc.) should inherit from this class and implement
the required abstract methods.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from pydantic import BaseModel as PydanticBaseModel
from functools import wraps
import inspect

from tempus_bench.config.models import JobConfig
from tempus_bench.metrics.evaluation import Evaluator
from tempus_bench.config.config import ConfigAdapterMixin
from tempus_bench.config.models import JobConfig
from tempus_bench.utils.logger import get_logger

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
        evaluator: Evaluator instance for computing metrics
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any], ParamsClass: PydanticBaseModel):
        """
        Initialize the base model.

        Args:
            config_path: Path to the configuration YAML file
            logs_path: Directory for storing log files (required)
            hyperparameters: Model-specific hyperparameters to inject into config
        """
        super().__init__()
        self.logger = get_logger()
        # Setup parameters
        self.params_class = ParamsClass
        self.model_name = self.model_class_name.replace('Model', '').lower()
        self.evaluator = Evaluator()
        self.set_params(**params)
        self.set_attrs(**settings) # Settings


    @abstractmethod
    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "BaseModel":
        """
        Train the model on given data.

        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            y_start_date: The start date timestamp for y_context and y_target in string form

        Returns:
            self: The fitted model instance
        """
        pass

    @abstractmethod
    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        pass

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.

        This method computes evaluation metrics as configured in evaluation.metrics

        Args:
            y_true: True target values (ndarray, shape [num_steps, num_features])
            y_pred: Predicted values (ndarray, shape [num_steps, num_features])

        Returns:
            Dict[str, float]: Dictionary of computed loss metrics (from evaluation.metrics)
        """
        return self.evaluator.evaluate(y_true, y_pred, **kwargs)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        pass

    def get_params(self) -> Dict[str, Any]:
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
        self.params = self.params_class(**params)
        self.set_attrs(**params)
        self.is_fitted = False  # Mark as unfitted if parameters change
        return self

    def set_attrs(self, **attrs: Dict[str, Any]):
        """
        Set model parameters.

        Args:
            **params: Model parameters to set

        Returns:
            self: The model instance with updated parameters
        """
        self.settings = attrs
        for key, value in attrs.items(): setattr(self, key, value)

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
        y_context = params.get('y_context')
        y_target = params.get('y_target')
        timestamps_context = params.get('timestamps_context')
        timestamps_target = params.get('timestamps_target')

        # Validate y_context (required parameter)
        if y_context is None:
            raise ValueError("y_context cannot be None")

        if not isinstance(y_context, np.ndarray):
            raise TypeError(f"y_context must be np.ndarray, got {type(y_context)}")

        if y_context.ndim != 2:
            raise ValueError(f"y_context must be 2D array, got {y_context.ndim}D with shape {y_context.shape}")

        num_steps_context, num_targets_context = y_context.shape
        if num_steps_context == 0 or num_targets_context == 0:
            raise ValueError(f"y_context cannot have zero dimensions, got shape {y_context.shape}")

        # Validate y_target if present
        if y_target is not None:
            if not isinstance(y_target, np.ndarray):
                raise TypeError(f"y_target must be np.ndarray, got {type(y_target)}")

            if y_target.ndim != 2:
                raise ValueError(f"y_target must be 2D array, got {y_target.ndim}D with shape {y_target.shape}")

            num_steps_target, num_targets_target = y_target.shape
            if num_steps_target == 0 or num_targets_target == 0:
                raise ValueError(f"y_target cannot have zero dimensions, got shape {y_target.shape}")

            # Check that num_targets match
            if num_targets_target != num_targets_context:
                raise ValueError(
                    f"y_target must have same num_targets as y_context: "
                    f"expected {num_targets_context}, got {num_targets_target}"
                )

        # Validate timestamps_context if present
        if timestamps_context is not None:
            if not isinstance(timestamps_context, np.ndarray):
                raise TypeError(f"timestamps_context must be np.ndarray, got {type(timestamps_context)}")

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
                raise TypeError(f"timestamps_target must be np.ndarray, got {type(timestamps_target)}")

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
