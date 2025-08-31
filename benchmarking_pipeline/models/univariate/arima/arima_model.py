"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides an ARIMA model implementation for univariate time series forecasting.
ARIMA models combine autoregression, differencing, and moving average components to
capture temporal dependencies in time series data.

The model supports both seasonal and non-seasonal ARIMA variants and can handle
exogenous variables for enhanced forecasting performance.
"""
import pdb
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel


class ArimaModel(BaseModel):
    """
    ARIMA model for univariate time series forecasting.
    
    This class implements the ARIMA model with support for:
    - Non-seasonal ARIMA(p,d,q) models
    - Seasonal ARIMA(p,d,q)(P,D,Q,s) models
    - Exogenous variable support
    - Rolling window predictions
    - Model persistence and loading
    
    Attributes:
        p: AR order (autoregressive)
        d: Differencing order (integration)
        q: MA order (moving average)
        s: Seasonality period
        model_: Fitted ARIMA model instance
        loss_function: Loss function for training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ARIMA model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - p: int, AR order (default: 1)
                - d: int, differencing order (default: 1)
                - q: int, MA order (default: 1)
                - s: int, seasonality period (default: 1)
                - loss_function: str, loss function for training (default: 'mae')
                - forecast_horizon: int, number of steps to forecast ahead
        """
        super().__init__(config)
        
        # Extract ARIMA-specific parameters
        if 'p' not in self.model_config:
            raise ValueError("p must be specified in config")
        if 'd' not in self.model_config:
            raise ValueError("d must be specified in config")
        if 'q' not in self.model_config:
            raise ValueError("q must be specified in config")
        if 's' not in self.model_config:
            raise ValueError("s must be specified in config")
        if 'training_loss' not in self.model_config:
            raise ValueError("training_loss must be specified in config")
        
        # Initialize model state
        self.model_ = None
        self.is_fitted = False
        self.training_loss = self.model_config['training_loss']
        
        # forecast_horizon is inherited from parent class (BaseModel)
        
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ArimaModel":
        """
        Train the ARIMA model on given data.

        Args:
            y_context: Past target values - training data (required)
            y_target: Future target values (not used in training, for compatibility)
            timestamps_context: Timestamps for y_context (not used in ARIMA)
            timestamps_target: Timestamps for y_target (not used in ARIMA)
            freq: Frequency string (required by interface, not used in ARIMA)

        Returns:
            self: The fitted model instance

        Note:
            ARIMA models only use y_context for training.
            y_target, timestamps_context, timestamps_target, and freq are ignored to prevent data leakage.
        """
        # Convert y_context to numpy array if needed
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        else:
            endog = y_context

        # No exogenous variables supported
        exog = None

        # Use seasonal_order only if seasonal period is greater than 1
        if self.model_config['s'] > 1:
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config['p'],
                    self.model_config['d'],
                    self.model_config['q'],
                ),
                seasonal_order=(0, 0, 0, self.model_config['s']),
                exog=exog,
            )
        else:
            # Non-seasonal ARIMA
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config['p'],
                    self.model_config['d'],
                    self.model_config['q'],
                ),
                exog=exog,
            )

        self.model_ = model.fit()
        self.is_fitted = True
        return self
        
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Make predictions using the trained ARIMA model, rolling forward using the fitted model.

        Args:
            y_context: Recent/past target values (not used for ARIMA prediction)
            timestamps_context: Timestamps for y_context (not used for ARIMA prediction)
            timestamps_target: Timestamps for the prediction horizon (used to determine forecast length)
            freq: Frequency string (must be provided from CSV data, required)

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If model is not fitted, freq is not provided, or forecast length cannot be determined
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        if freq is None or freq == "":
            raise ValueError("Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks.")
        if timestamps_target is None:
            raise ValueError("timestamps_target must be provided to determine forecast horizon for ARIMA.")
        forecast_steps = len(timestamps_target)
        if forecast_steps <= 0:
            raise ValueError("Forecast horizon must be positive (timestamps_target must be non-empty).")

        forecast = self.model_.forecast(steps=forecast_steps, exog=None)
        forecast_array = np.asarray(forecast)

        self._last_y_pred = forecast_array.reshape(-1, 1)

        return self._last_y_pred
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'p': self.model_config['p'],
            'd': self.model_config['d'],
            'q': self.model_config['q'],
            's': self.model_config['s'],
            'loss_function': self.training_loss,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'ArimaModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        # Track if model parameters that affect fitting are changed
        model_params_changed = False
        
        for key, value in params.items():
            if key in ('dataset',):
                # Skip dataset parameter - it's not an ARIMA model parameter
                continue
            elif hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['p', 'd', 'q', 's'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.model_config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed and self.is_fitted:
            self.model_ = None
            self.is_fitted = False
            
        return self
        
    def save(self, path: str) -> None:
        """
        Save the ARIMA model to disk.
        
        Args:
            path: Path to save the model
            
        Raises:
            ValueError: If model is not fitted
            RuntimeError: If saving fails
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'model_config': self.model_config,
            'is_fitted': self.is_fitted,
            'params': self.get_params(),
            'model': self.model_
        }
        
        try:
            # Save model state to file
            with open(path, 'wb') as f:
                pickle.dump(model_state, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {str(e)}")
        
    def load(self, path: str) -> None:
        """
        Load the ARIMA model from disk.
        
        Args:
            path: Path to load the model from
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        try:
            # Load model state from file
            with open(path, 'rb') as f:
                model_state = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.set_params(**model_state['params'])
        self.model_ = model_state['model']
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the ARIMA model's properties and performance.
        
        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        summary = {
            'model_type': 'ARIMA',
            'order': (self.model_config['p'], self.model_config['d'], self.model_config['q']),
            'seasonal_period': self.model_config['s'],
            'is_fitted': self.is_fitted,
            'forecast_horizon': self.forecast_horizon,
            'loss_function': self.training_loss
        }
        
        if self.is_fitted and self.model_ is not None:
            # Add fitted model information
            summary.update({
                'aic': getattr(self.model_, 'aic', None),
                'bic': getattr(self.model_, 'bic', None),
                'hqic': getattr(self.model_, 'hqic', None),
                'llf': getattr(self.model_, 'llf', None),  # Log-likelihood
                'nobs': getattr(self.model_, 'nobs', None)  # Number of observations
            })
            
        return summary
        
    def get_last_eval_true_pred(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the last evaluation's true values and predictions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_true, y_pred) from last evaluation
        """
        if hasattr(self, '_last_y_true') and hasattr(self, '_last_y_pred'):
            return self._last_y_true, self._last_y_pred
        else:
            return None, None 