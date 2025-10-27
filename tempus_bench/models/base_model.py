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

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional

from tempus_bench.utils.logger import get_logger
from tempus_bench.config import get_config_manager
from tempus_bench.config.models import UnifiedConfig
from tempus_bench.metrics.evaluation import Evaluator
from tempus_bench.utils.tf_logger import get_tf_logger

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

    def __init__(self, config: UnifiedConfig, logs_path: str):
        """
        Initialize the base model.

        Args:
            config_path: Path to the configuration YAML file
            logs_path: Directory for storing log files (required)
            hyperparameters: Model-specific hyperparameters to inject into config
        """
        self.config = config.benchmark.model_dump()
        self.model_name = self.__class__.__name__.replace('Model', '').lower()
        if len(self.config['model']) > 1:
            raise ValueError("Only one model is allowed to be defined in the config")
        # check if self.config['model'] unique key coincides with the model name
        if self.model_name not in self.config['model']:
            raise ValueError(f"Model parameters for {self.model_name} not found in config")
        self.model_config = self.config["model"][self.model_name]
        self.model_settings = config.model_settings[self.model_name].model_dump()
        self.eval_config = self.config["benchmark"]["evaluation"]
        self.stochastic_tuning_loss = self.eval_config['stochastic_tuning_loss']
        self.deterministic_tuning_loss = self.eval_config['deterministic_tuning_loss']
        self.logger = get_logger(logs_path)
        self.tf_logger = get_tf_logger(Path(logs_path).parent / 'tensorboard')
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