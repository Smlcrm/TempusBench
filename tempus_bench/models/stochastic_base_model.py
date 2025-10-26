"""
Stochastic base model class that defines the interface for all stochastic time series forecasting models.

This abstract base class provides a common interface for stochastic models that generate
probabilistic forecasts through sampling. It handles configuration management, training,
prediction, evaluation, and model persistence.

All stochastic models (Chronos, DeepAR, LagLlama, Moirai, Toto, etc.) should inherit
from this class and implement the required abstract methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Tuple, List, Optional
import pandas as pd
import json
import os
import pickle
from tempus_bench.models.base_model import BaseModel
from tempus_bench.metrics.evaluation import Evaluator


class StochasticBaseModel(BaseModel):
    """
    Abstract base class for stochastic time series forecasting models.

    This class provides a unified interface for training, prediction, and evaluation
    of stochastic time series forecasting models. It handles configuration management,
    data preprocessing, and evaluation metrics computation.

    Key differences from BaseModel:
    - predict() returns samples instead of point forecasts
    - compute_point_forecast() method to derive point forecasts from samples
    - Handles mixed deterministic/stochastic evaluation metrics

    Attributes:
        config: Configuration dictionary containing model and dataset parameters
        training_loss: Primary loss function for training
        forecast_horizon: Number of steps to forecast ahead
        is_fitted: Whether the model has been trained
        evaluator: Evaluator instance for computing metrics
    """

    def __init__(self, config: Dict[str, Any], logs_dir: str):
        """
        Initialize the stochastic base model.

        Args:
            config: Configuration dictionary containing model parameters
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
                - dataset: dict containing dataset configuration
                - evaluation: dict containing evaluation configuration
            logs_dir: Directory for storing log files (required)
        """
        super().__init__(config, logs_dir)

        # Get evaluation configuration
        evaluation_cfg = self.config["evaluation"]
        self.num_samples = evaluation_cfg["num_samples"]
        self.point_forecast_statistic = evaluation_cfg["point_forecast_statistic"]

        # Validate point forecast statistic
        if self.point_forecast_statistic != "mean":
            raise ValueError(f"Only 'mean' point forecast statistic is supported, got '{self.point_forecast_statistic}'")

    @abstractmethod
    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained stochastic model.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            freq: Frequency string (e.g., 'H', 'D', 'M') - MUST be provided from CSV data

        Returns:
            np.ndarray: Model prediction samples with shape (num_samples, forecast_horizon, num_targets)
                       For univariate: (num_samples, forecast_horizon, 1)

        Raises:
            ValueError: If freq is None or empty - frequency must always be read from CSV data
        """
        if freq is None or freq == "":
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )
        pass

    def compute_point_forecast(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute point forecast from samples based on evaluation.point_forecast_statistic.

        Args:
            samples: Prediction samples with shape (num_samples, forecast_horizon, num_targets)

        Returns:
            np.ndarray: Point forecast with shape (forecast_horizon, num_targets)

        Raises:
            ValueError: If point_forecast_statistic is not supported
        """
        if self.point_forecast_statistic == "mean":
            return np.mean(samples, axis=0)
        else:
            raise ValueError(f"Point forecast statistic '{self.point_forecast_statistic}' not supported")

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.

        This method computes evaluation metrics as configured in evaluation.metrics.
        For stochastic metrics, it uses the samples directly.
        For deterministic metrics, it computes point forecasts from samples.

        Args:
            y_true: True target values (ndarray, shape [num_steps, num_targets])
            y_pred: Predicted samples (ndarray, shape [num_samples, num_steps, num_targets])

        Returns:
            Dict[str, float]: Dictionary of computed loss metrics (from evaluation.metrics)
        """
        # Store for TensorBoard logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred

        # Compute point forecast for deterministic metrics
        y_pred_point = self.compute_point_forecast(y_pred)

        # Use evaluator to compute evaluation metrics
        # Pass both samples and point forecast to evaluator
        return self.evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            **kwargs
        )

    def get_last_eval_true_pred(self):
        """
        Return the last y_true and y_pred used in compute_loss for TensorBoard logging.

        Returns:
            Tuple of (y_true, y_pred) arrays
        """
        return self._last_y_true, self._last_y_pred

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
            "num_samples": self.num_samples,
            "point_forecast_statistic": self.point_forecast_statistic,
        }
