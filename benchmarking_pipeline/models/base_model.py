"""
Base model class that defines the interface for all traditional time series forecasting models.

This abstract base class provides a common interface for traditional statistical and machine learning
models used in time series forecasting. It handles configuration management, training, prediction,
evaluation, and model persistence.

All traditional models (ARIMA, LSTM, XGBoost, etc.) should inherit from this class and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Tuple, List, Optional
import pandas as pd
import json
import os
import pickle
from benchmarking_pipeline.pipeline.evaluator import Evaluator


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
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the base model.
        
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
        
        # Handle nested model configurations (e.g., config['model']['arima'])
        # Extract model-specific config if it exists
        model_config = self._extract_model_config(self.config)
        
        # training_loss is optional for statistical models that don't require training
        self.training_loss = model_config.get('training_loss', None)
        
        # Determine forecast horizon from model configuration keys if present
        # Common names across models: forecast_horizon, prediction_length, horizon_len, pdt
        horizon = None
        for key in ('forecast_horizon', 'prediction_length', 'horizon_len', 'pdt'):
            if key in model_config:
                horizon = model_config[key]
                break
        # If hyperparameter grid provides a list, take the first value for initialization
        if isinstance(horizon, list) and len(horizon) > 0:
            horizon = horizon[0]
        self.forecast_horizon = horizon
        
        self.is_fitted = False
        
        # Initialize evaluator with the full config to access evaluation.metrics
        # We need to pass the original config, not the extracted model config
        print(f"[DEBUG] BaseModel: Original config parameter: {config}")
        print(f"[DEBUG] BaseModel: Self.config after extraction: {self.config}")
        self.evaluator = Evaluator(config=config)
        
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
            # Find the first model configuration (e.g., 'arima', 'lstm', etc.)
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
    ) -> 'BaseModel':
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
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None,
        freq: str = None
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
            raise ValueError("Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks.")
        pass
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None, y_train: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        This method computes evaluation metrics as configured in evaluation.metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Name of the loss function to use (defaults to training_loss)
            y_train: Training target values (required for MASE calculation)
            
        Returns:
            Dict[str, float]: Dictionary of computed loss metrics (from evaluation.metrics)
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        if y_train is not None and isinstance(y_train, pd.Series):
            y_train = y_train.values
            
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
        
        # Use evaluator to compute evaluation metrics (from evaluation.metrics)
        # Pass y_train for metrics like MASE that require it
        return self.evaluator.evaluate(y_pred, y_true, y_train=y_train)
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
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
        loss = self.compute_loss(y, predictions)
        return loss
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params: Dict[str, Any]) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params()
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        # Load model state from file
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.set_params(**model_state['params'])
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's properties and performance.
        
        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'parameters': self.get_params()
        }

    def get_last_eval_true_pred(self):
        """
        Return the last y_true and y_pred used in compute_loss for TensorBoard logging.
        
        Returns:
            Tuple of (y_true, y_pred) arrays
        """
        return self._last_y_true, self._last_y_pred 