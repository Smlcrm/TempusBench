"""
Multivariate ARIMA model implementation.

This model extends the univariate ARIMA to handle multiple target variables simultaneously.
Uses Vector Autoregression (VAR) which is the multivariate extension of ARIMA models.
Design choices (similar to multivariate LSTM):
- Uses VAR from statsmodels for multivariate time series modeling
- Handles multiple targets simultaneously in one model
- Predicts all targets in a single forward pass
- Supports differencing for non-stationary series
"""

import os
import numpy as np
import math
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Union, Tuple, Optional
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")


class MultivariateARIMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Multivariate ARIMA model with given configuration.

        Args:
            config: Configuration dictionary containing model parameters
                - p: int, VAR order (autoregressive order)
                - d: int, differencing order for stationarity
                - maxlags: int, maximum number of lags to consider
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config)
        if "trend" not in self.model_config:
            raise ValueError("trend must be specified in config")
        if "maxlags" not in self.model_config:
            self.model_config["maxlags"] = 20
        if "ic" not in self.model_config:
            raise ValueError("ic must be specified in config")

        self.model = None

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "MultivariateARIMAModel":
        """
        Train the Multivariate ARIMA model on given data.

        TECHNIQUE: Vector Autoregression (VAR) for Multiple Time Series
        - Extends ARIMA to handle multiple interdependent time series
        - Each variable depends on its own lags and lags of other variables
        - Captures cross-dependencies between multiple targets
        - Applies differencing if needed to achieve stationarity
        - Uses Maximum Likelihood Estimation for parameter fitting

        Args:
            y_context: Past target values (time series) - used for training (can be DataFrame for multivariate)
            y_target: Future target values (optional, for extended training)
            y_start_date: The start date timestamp for y_context and y_target
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        timestamps_context = self.convert_to_datetimeindex(timestamps_context)
        if not self.is_fitted:
            self.model = VAR(
                endog=y_context, exog=None, dates=timestamps_context, freq=freq
            )

        self.results = self.model.fit(
            maxlags=self.model_config["maxlags"],
            ic=self.model_config["ic"],
            trend=self.model_config["trend"],
        )

        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Multivariate ARIMA model.

        TECHNIQUE: VAR Forecasting for Multiple Time Series
        - Uses fitted VAR model to predict multiple steps ahead
        - Predicts all targets simultaneously using their interdependencies
        - Handles both in-sample and out-of-sample forecasting
        - Reverses differencing to get predictions in original scale

        Args:
            y_context: Past target values for prediction context
            y_target: Future target values (used to determine prediction length)
            x_context: Past exogenous variables (optional, ignored for now)
            x_target: Future exogenous variables (optional, ignored for now)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call train first.")

        forecast_horizon = timestamps_target.shape[0]
        lag_order = self.results.k_ar
        forecast_steps = len(timestamps_target)

        y_context = y_context[-lag_order:, :]
        forecasts = self.results.forecast(y_context, steps=forecast_steps)

        forecasts = np.array(forecasts)
        if len(forecasts.shape) == 1:
            forecasts = np.expand_dims(forecasts, axis=-1)

        return forecasts

    def convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                # Convert from nanoseconds to datetime
                if timestamps[0] > 1e18:  # Likely nanoseconds
                    timestamps = pd.to_datetime(timestamps, unit="ns")
                elif timestamps[0] > 1e15:  # Likely microseconds
                    timestamps = pd.to_datetime(timestamps, unit="us")
                elif timestamps[0] > 1e12:  # Likely milliseconds
                    timestamps = pd.to_datetime(timestamps, unit="ms")
                else:  # Likely seconds
                    timestamps = pd.to_datetime(timestamps, unit="s")
            else:
                timestamps = pd.to_datetime(timestamps)
        else:
<<<<<<< HEAD
            differenced_context = context_data.copy()
        
        forecast_steps = len(y_target)
        
        # Calculate how many prediction windows we need
        num_windows = math.ceil(forecast_steps / self.forecast_horizon)
        
        all_predictions = []
        current_data = differenced_context.copy()
        
        for window in range(num_windows):
            # Predict using VAR model
            steps_to_predict = min(self.forecast_horizon, forecast_steps - len(all_predictions))
            
            # VAR forecast
            forecast_result = self.fitted_model.forecast(
                current_data.values[-self.p:], 
                steps=steps_to_predict
            )
            
            # Add predictions to list
            all_predictions.extend(forecast_result)
            
            # Update data for next window if needed
            if window < num_windows - 1:
                # Add predictions to current_data for next iteration
                pred_df = pd.DataFrame(forecast_result, columns=current_data.columns)
                current_data = pd.concat([current_data, pred_df], ignore_index=True)
        
        # Convert to numpy array
        predictions = np.array(all_predictions[:forecast_steps])
        
        # Reverse differencing if it was applied
        if self.d > 0:
            # Create DataFrame for integration (no column names needed)
            pred_df = pd.DataFrame(predictions)
            integrated_predictions = self._integrate_data(pred_df, self.original_data, self.d)
            predictions = integrated_predictions.values
        
        return predictions
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'p': self.p,
            'd': self.d,
            'maxlags': self.maxlags,
            'num_targets': self.num_targets,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateARIMAModel':
        """
        Set model parameters.

        Args:
            **params: Model parameters to set

        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if key in ['p', 'd', 'maxlags', 'forecast_horizon']:
                value = int(value)
            self.model_config[key] = value
            # Also update the attribute if it exists
            if hasattr(self, key):
                setattr(self, key, value)
        return self
        
    def save(self, path: str) -> None:
        """
        Save the Multivariate ARIMA model to disk.
        
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
            'params': self.get_params(),
            'fitted_model': self.fitted_model,
            'original_data': self.original_data,
            'differenced_data': self.differenced_data
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
    def load(self, path: str) -> None:
        """
        Load the Multivariate ARIMA model from disk.
        
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
        self.fitted_model = model_state['fitted_model']
        self.original_data = model_state['original_data']
        self.differenced_data = model_state['differenced_data']
        self.set_params(**model_state['params'])
        
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Name of the loss function to use (defaults to training_loss)
            
        Returns:
            Dict[str, float]: Dictionary of computed loss metrics
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        elif isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        if y_train is not None:
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.values
            elif isinstance(y_train, pd.Series):
                y_train = y_train.values
            
        # Store for logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # For multivariate data, compute loss for each target and average
        if y_true.ndim == 2 and y_pred.ndim == 2:
            # Multivariate case - compute loss for each target and average
            all_metrics = {}
            for i in range(y_true.shape[1]):
                target_true = y_true[:, i]
                target_pred = y_pred[:, i]
                target_train = None
                if y_train is not None:
                    target_train = y_train[:, i] if y_train.ndim == 2 else y_train
                
                # Ensure both arrays have the same length
                min_length = min(len(target_true), len(target_pred))
                target_true = target_true[:min_length]
                target_pred = target_pred[:min_length]
                
                # Compute metrics for this target (pass y_train for MASE when available)
                target_metrics = self.evaluator.evaluate(target_pred, target_true, y_train=target_train)
                
                # Store metrics for this target
                for metric_name, metric_value in target_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Average metrics across all targets
            averaged_metrics = {}
            for metric_name, metric_values in all_metrics.items():
                averaged_metrics[metric_name] = np.mean(metric_values)
            
            return averaged_metrics
        else:
            # Univariate case - use original logic
            # Handle shape mismatches
            if y_pred.ndim == 2 and y_true.ndim == 1:
                # If predictions are 2D and true values are 1D, flatten predictions
                if y_pred.shape[0] == 1:
                    y_pred = y_pred.flatten()
                elif y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    y_pred = y_pred[0]
            
            # Ensure both arrays have the same length
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            
            # Use evaluator to compute all metrics
            return self.evaluator.evaluate(y_pred, y_true, y_train=y_train)
=======
            timestamps = timestamps

        return timestamps
>>>>>>> f9638005dd4aa8d75af6b035677ec1f02bd24115
