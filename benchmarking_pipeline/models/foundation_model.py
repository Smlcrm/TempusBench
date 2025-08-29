"""
Base model class that defines the interface for all foundation models.

This abstract base class provides a common interface for modern foundation models used in time series
forecasting. Foundation models are large pre-trained models that can be fine-tuned for specific tasks.

Examples include Chronos, LagLlama, Moirai, TimesFM, and other transformer-based models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Tuple, List, Optional
import pandas as pd
import json
import os
import pickle
from benchmarking_pipeline.pipeline.evaluator import Evaluator


class FoundationModel(ABC):
    """
    Abstract base class for foundation time series forecasting models.
    
    This class provides a unified interface for training, prediction, and evaluation
    of foundation models. Foundation models are typically large pre-trained models
    that can be fine-tuned for specific forecasting tasks.
    
    Attributes:
        config: Configuration dictionary containing model and dataset parameters
        training_loss: Primary loss function for training
        forecast_horizon: Number of steps to forecast ahead
        is_fitted: Whether the model has been trained
        evaluator: Evaluator instance for computing metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the foundation model.
        
        Args:
            config: Configuration dictionary containing model parameters
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
                - dataset: dict containing dataset configuration
            config_file: Path to a JSON configuration file
        """
        if config_file is not None:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        self.config = config or {}
        
        # Handle nested model configurations (e.g., config['model']['chronos'])
        # Extract model-specific config if it exists
        model_config = self._extract_model_config(self.config)
        
        # Foundation models don't require training_loss since they're pre-trained
        # self.training_loss = None  # Not needed for foundation models
        
        # Determine forecast horizon from model configuration keys if present
        # Common names across foundation models: prediction_length, horizon_len, pdt
        horizon = None
        model_config = self._extract_model_config(self.config)
        for key in ('prediction_length', 'horizon_len', 'pdt', 'forecast_horizon'):
            if key in model_config:
                horizon = model_config[key]
                break
        if isinstance(horizon, list) and len(horizon) > 0:
            horizon = horizon[0]
        self.forecast_horizon = horizon
        
        self.is_fitted = False
        
        # Initialize evaluator with the full config to access evaluation.metrics
        self.evaluator = Evaluator(config=self.config)
        
        # For logging last eval
        self._last_y_true = None
        self._last_y_pred = None
    
    def _extract_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model-specific configuration from nested config structure.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Dict[str, Any]: Model-specific configuration
        """
        # If config has a 'model' section, look for the specific model type
        if 'model' in config:
            model_section = config['model']
            # Find the first model configuration (e.g., 'chronos', 'lagllama', etc.)
            for model_name, model_config in model_section.items():
                if isinstance(model_config, dict):
                    return model_config
        
        # If no nested structure, return the config as-is
        return config
    
    @abstractmethod
    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray]], 
              y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_start_date: Optional[str] = None
    ) -> 'FoundationModel':
        """
        Train/fine-tune the foundation model on given data.
        
        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            x_context: Past exogenous variables - used during tuning and testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            x_target: Future exogenous variables - if provided, can be used with x_context for training
            y_start_date: The start date timestamp for y_context and y_target in string form
            x_start_date: The start date timestamp for x_context and x_target in string form
            
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions using the trained foundation model.
        
        Args:
            y_context: Recent/past target values
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon)
        """
        pass
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None, y_train: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Name of the loss function to use (defaults to training_loss)
            y_train: Training target values for metrics like MASE
            
        Returns:
            Dict[str, float]: Dictionary of computed loss metrics
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Handle shape mismatches
        if y_pred.ndim == 2 and y_true.ndim == 1:
            # If predictions are 2D and true values are 1D, flatten predictions
            if y_pred.shape[0] == 1:
                # Single prediction row, flatten it
                y_pred = y_pred.flatten()
            elif y_pred.shape[1] == 1:
                # Single prediction column, flatten it
                y_pred = y_pred.flatten()
            else:
                # Multiple predictions, take the first row
                y_pred = y_pred[0]
        
        # Ensure both arrays have the same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        
        # Store for TensorBoard logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # Use evaluator to compute all metrics
        return self.evaluator.evaluate(y_pred, y_true, y_train=y_train)
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params: Dict[str, Any]) -> 'FoundationModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        pass
    
    def get_last_eval_true_pred(self):
        """
        Return the last y_true and y_pred used in compute_loss for TensorBoard logging.
        
        Returns:
            Tuple of (y_true, y_pred) arrays
        """
        return self._last_y_true, self._last_y_pred