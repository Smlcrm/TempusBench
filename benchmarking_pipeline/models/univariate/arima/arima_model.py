"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides an ARIMA model implementation for univariate time series forecasting.
ARIMA models combine autoregression, differencing, and moving average components to
capture temporal dependencies in time series data.

The model supports both seasonal and non-seasonal ARIMA variants and can handle
exogenous variables for enhanced forecasting performance.
"""

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
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
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
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        
        # Extract ARIMA-specific parameters
        if 'p' not in self.config:
            raise ValueError("p must be specified in config")
        if 'd' not in self.config:
            raise ValueError("d must be specified in config")
        if 'q' not in self.config:
            raise ValueError("q must be specified in config")
        if 's' not in self.config:
            raise ValueError("s must be specified in config")
        if 'training_loss' not in self.config:
            raise ValueError("training_loss must be specified in config")
        
        self.p = int(self.config['p'])
        self.d = int(self.config['d'])
        self.q = int(self.config['q'])
        self.s = int(self.config['s'])
        
        # Initialize model state
        self.model_ = None
        self.is_fitted = False
        self.training_loss = self.config['training_loss']
        
        # forecast_horizon is inherited from parent class (BaseModel)
        
    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              y_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: Optional[str] = None, 
              **kwargs
    ) -> 'ArimaModel':
        """
        Train the ARIMA model on given data.
        
        Args:
            y_context: Past target values - training data
            y_target: Future target values (not used in training, for compatibility)
            y_start_date: Start date for y_context (not used in ARIMA)
            **kwargs: Additional keyword arguments
            
        Returns:
            self: The fitted model instance
            
        Note:
            ARIMA models only use y_context for training.
            y_target is ignored to prevent data leakage.
        """
        # Convert y_context to numpy array if needed
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        else:
            endog = y_context
        
        # No exogenous variables supported
        exog = None
        
        # Use seasonal_order only if seasonal period is greater than 1
        if self.s > 1:
            model = ARIMA(endog=endog, order=(self.p, self.d, self.q), 
                         seasonal_order=(0, 0, 0, self.s), exog=exog)
        else:
            # Non-seasonal ARIMA
            model = ARIMA(endog=endog, order=(self.p, self.d, self.q), exog=exog)
        
        self.model_ = model.fit()
        self.is_fitted = True
        return self
        
    def predict(self, 
                y_context: Union[pd.Series, np.ndarray] = None,
                y_target: Union[pd.Series, np.ndarray] = None, 
                y_start_date: Optional[str] = None, 
                forecast_horizon: Optional[int] = None,
                **kwargs
    ) -> np.ndarray:
        """
        Make predictions using the trained ARIMA model.
        
        Args:
            y_context: Recent target values (not used in ARIMA prediction)
            y_target: Target values to predict (used to determine forecast length)
            y_start_date: Start date for y_context (not used)
            forecast_horizon: Number of steps to forecast (overrides y_target length if provided)
            **kwargs: Additional keyword arguments
        
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon)
            
        Raises:
            ValueError: If model is not fitted or forecast length cannot be determined
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Determine forecast length from y_target
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")
        
        # DEBUG: See what y_target actually contains
        print(f"[ARIMA DEBUG] y_target type: {type(y_target)}")
        print(f"[ARIMA DEBUG] y_target length: {len(y_target)}")
        print(f"[ARIMA DEBUG] y_target shape: {y_target.shape if hasattr(y_target, 'shape') else 'no shape'}")
        print(f"[ARIMA DEBUG] y_target first 5 values: {y_target[:5] if hasattr(y_target, '__getitem__') else y_target}")
        
        # Use y_target length to determine forecast steps (like working version from commit 912fceba)
        forecast_steps = len(y_target)
        
        if forecast_steps <= 0:
            raise ValueError("Forecast horizon must be positive.")
        
        # No exogenous variables supported
        exog = None
        
        # Generate forecast
        forecast = self.model_.forecast(steps=forecast_steps, exog=exog)
        
        # DEBUG: Show actual forecast values and compare with true values
        print(f"[ARIMA DEBUG] Forecast type: {type(forecast)}")
        print(f"[ARIMA DEBUG] Forecast length: {len(forecast) if hasattr(forecast, '__len__') else 'no len'}")
        print(f"[ARIMA DEBUG] Forecast first 10 values: {forecast[:10] if hasattr(forecast, '__getitem__') else forecast}")
        print(f"[ARIMA DEBUG] y_target first 10 values: {y_target[:10] if hasattr(y_target, '__getitem__') else y_target}")
        
        # Store predictions and true values for evaluation (simplified like working version)
        # Convert forecast to numpy array if it's a pandas Series
        if hasattr(forecast, 'values'):
            forecast_array = forecast.values
        else:
            forecast_array = np.array(forecast)
            
        self._last_y_pred = forecast_array.reshape(1, -1)
        if y_target is not None:
            self._last_y_true = y_target.reshape(1, -1) if hasattr(y_target, 'reshape') else np.array(y_target).reshape(1, -1)
        
        # DEBUG: Show final stored values
        print(f"[ARIMA DEBUG] Final _last_y_pred shape: {self._last_y_pred.shape}")
        print(f"[ARIMA DEBUG] Final _last_y_true shape: {self._last_y_true.shape}")
        print(f"[ARIMA DEBUG] Final _last_y_pred first 5 values: {self._last_y_pred.flatten()[:5]}")
        print(f"[ARIMA DEBUG] Final _last_y_true first 5 values: {self._last_y_true.flatten()[:5]}")
        
        return self._last_y_pred
        
    def rolling_predict(self, 
                       y_context: Union[pd.Series, np.ndarray], 
                       y_target: Union[pd.Series, np.ndarray] = None, 
                       y_start_date: Optional[str] = None, 
                       **kwargs
    ) -> np.ndarray:
        """
        Perform rolling window predictions for time series forecasting.
        This method retrains the model at each step with updated data.
        
        Args:
            y_context: Initial training data
            y_target: Target values to predict
            y_start_date: Start date for y_context (not used)
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Rolling predictions with shape (1, len(y_target))
            
        Raises:
            ValueError: If y_target is not provided
        """
        if y_target is None:
            raise ValueError("y_target must be provided for rolling predictions.")
        
        predictions = []
        current_context = y_context.copy()
        
        # Perform rolling predictions
        for i in range(len(y_target)):
            # Train model on current context
            self.train(y_context=current_context)
            
            # Make one-step prediction
            pred = self.predict(y_target=y_target[i:i+1])
            
            predictions.append(pred[0, 0])  # Extract single prediction value
            
            # Update context for next iteration
            current_context = np.append(current_context, y_target[i:i+1])
            if len(current_context) > len(y_context):  # Keep context size manageable
                current_context = current_context[-len(y_context):]
        
        # Store final predictions and true values
        self._last_y_pred = np.array(predictions).reshape(1, -1)
        # Convert y_target to numpy array and reshape
        if hasattr(y_target, 'values'):
            y_target_array = y_target.values
        else:
            y_target_array = np.array(y_target)
        self._last_y_true = y_target_array.reshape(1, -1)
        
        return self._last_y_pred
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            's': self.s,
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
                self.config[key] = value
        
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
            'config': self.config,
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
            'order': (self.p, self.d, self.q),
            'seasonal_period': self.s,
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