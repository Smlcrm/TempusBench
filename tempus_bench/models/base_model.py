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

from itertools import product
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Union, Optional
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.config.models import JobConfig
from tempus_bench.metrics.evaluation import Evaluator
from tempus_bench.config.config import ConfigAdapterMixin
from tempus_bench.config.models import JobConfig

class BaseModelParams(PydanticBaseModel):
    pass

class BaseModel(ABC, ConfigAdapterMixin):
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

    def __init__(self, config: JobConfig, logs_path: str, ParamsClass: PydanticBaseModel):
        """
        Initialize the base model.

        Args:
            config_path: Path to the configuration YAML file
            logs_path: Directory for storing log files (required)
            hyperparameters: Model-specific hyperparameters to inject into config
        """

        super().__init__(config, logs_path)
        self.params_class = ParamsClass
        self.hyperparameter_grid = self.get_hyperparameter_grid(self.params_class)
        self._setup_logging()

        self.model_class_name = self.__class__.__name__
        if self.model_name != self.model_class_name.replace('Model', '').lower():
            raise ValueError(f"Model Class {self.model_class_name} is not compatible with the model name in the config: {self.model_name}")
        self.evaluator = Evaluator(config, logs_path)
        self.is_fitted = False

    @abstractmethod
    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
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
    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            y_context: Recent/past target values (for sequence models, optional for ARIMA)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            freq: Frequency string (e.g., 'H', 'D', 'M') - MUST be provided from CSV data

        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon)

        Raises:
            ValueError: If freq is None or empty - frequency must always be read from CSV data
        """
        if freq is None or freq == "":
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )
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
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Train and evaluate model on given data.

        Args:
            X: Evaluation features
            y: True target values

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Get predictions
        predictions = self.predict(X)
        return self.compute_loss(y, predictions)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.

        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return self.model_config

    def set_params(self, **params: Dict[str, Any]) -> "BaseModel":
        """
        Set model parameters.

        Args:
            **params: Model parameters to set

        Returns:
            self: The model instance with updated parameters
        """
        self.model_config.update(params)
        self.is_fitted = False  # Mark as unfitted if parameters change

        return self

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

    def get_hyperparameter_grid(self, ParamsClass: PydanticBaseModel = None) -> list[dict]:
        """
        Generate a hyperparameter grid using Cartesian product of lists in self.model_config.

        This method uses the already initialized model_config (dict of parameter name: list of values)
        to build all combinations of hyperparameters.
        """
        params = self.model_config
        if ParamsClass is None: ParamsClass = self.params_class
        # Foundation models may have no hyperparameters configured
        if not params: return [{}]

        def _is_valid_combination(combination: dict) -> bool:
            return True

        values_lists = [params[k] for k in params.keys()]
        grid: list[dict] = []
        param_keys = list(params.keys())  # Explicitly capture the parameter keys in order
        for values_tuple in product(*values_lists):
            combo = dict(zip(param_keys, values_tuple))
            combo = ParamsClass(**combo).model_dump()
            if _is_valid_combination(combo):
                grid.append(combo)

        return grid