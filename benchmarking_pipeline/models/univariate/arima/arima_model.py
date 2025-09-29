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
from pathlib import Path
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

        # Store the full config for creating variate models
        self.full_config = config

        # Detect if this is anyvariate mode based on config filename
        self.is_anyvariate = False
        if hasattr(self, 'model_config') and 'original_config_path' in config:
            config_filename = Path(config['original_config_path']).name.lower()
            if "univariate" in config_filename:
                self.is_anyvariate = True
                print(f"[ARIMA] Anyvariate mode enabled for config: {config_filename}")

        # Extract ARIMA-specific parameters from the model config section
        # Get the actual arima model config (either from model_config for hyperparameter tuning or from full config)
        if hasattr(self, 'model_config') and isinstance(self.model_config, dict) and 'p' in self.model_config:
            arima_config = self.model_config
        elif 'model' in config and 'arima' in config['model']:
            arima_config = config['model']['arima']
        else:
            arima_config = self.model_config
            
        if "p" not in arima_config:
            raise ValueError("p must be specified in config")
        if "d" not in arima_config:
            raise ValueError("d must be specified in config")
        if "q" not in arima_config:
            raise ValueError("q must be specified in config")
        if "s" not in arima_config:
            raise ValueError("s must be specified in config")

        # Set training loss to one of the configured metrics if mae is not available
        if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'metrics_to_calculate'):
            if 'mae' not in self.evaluator.metrics_to_calculate:
                # Use the first available metric for training loss optimization
                available_metrics = self.evaluator.metrics_to_calculate
                if 'rmse' in available_metrics:
                    self.training_loss = 'rmse'
                elif 'mse' in available_metrics:
                    self.training_loss = 'mse'
                elif len(available_metrics) > 0:
                    self.training_loss = available_metrics[0]

        # Store the arima config for model building
        self.arima_config = arima_config
        
        # Initialize model state
        self.model_ = None
        self.models = []  # For anyvariate mode
        self.is_fitted = False

        # forecast_horizon is inherited from parent class (BaseModel)

    def _train(
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
        if self.arima_config["s"] > 1:
            model = ARIMA(
                endog=endog,
                order=(
                    self.arima_config["p"],
                    self.arima_config["d"],
                    self.arima_config["q"],
                ),
                seasonal_order=(0, 0, 0, self.arima_config["s"]),
                exog=exog,
            )
        else:
            # Non-seasonal ARIMA
            model = ARIMA(
                endog=endog,
                order=(
                    self.arima_config["p"],
                    self.arima_config["d"],
                    self.arima_config["q"],
                ),
                exog=exog,
            )

        self.model_ = model.fit()
        self.is_fitted = True
        return self

    def _predict(
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
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )

        if timestamps_target is None:
            raise ValueError(
                "timestamps_target must be provided to determine forecast horizon for ARIMA."
            )
        forecast_steps = len(timestamps_target)
        if forecast_steps <= 0:
            raise ValueError(
                "Forecast horizon must be positive (timestamps_target must be non-empty)."
            )

        forecast = self.model_.forecast(steps=forecast_steps, exog=None)
        forecast_array = np.asarray(forecast)

        self._last_y_pred = forecast_array.reshape(-1, 1)

        return self._last_y_pred

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ):
        """
        Train the ARIMA model. Supports anyvariate mode for handling multivariate data.
        
        Args:
            y_context: Past target values - training data
            y_target: Future target values (for validation)
            timestamps_context: Timestamps for y_context
            timestamps_target: Timestamps for y_target
            freq: Frequency string
            **kwargs: Additional keyword arguments
            
        Returns:
            self: The fitted model instance
        """
        # Ensure y_context is 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
        if y_target is not None and y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)

        n_variates = y_context.shape[1]

        # For univariate case or non-anyvariate mode, use the original method
        if n_variates == 1 or not self.is_anyvariate:
            return self._train(y_context, y_target, timestamps_context, timestamps_target, freq, **kwargs)

        # For anyvariate mode with multivariate data, train separate ARIMA models for each variate
        self.models = []  # Store individual models for each variate

        for variate_idx in range(n_variates):
            print(f"[INFO] Training ARIMA model for variate {variate_idx + 1}/{n_variates}")

            # Create a new ARIMA model for this variate
            variate_model = ArimaModel(self.full_config)

            # Extract data for this specific variate
            variate_context = y_context[:, variate_idx:variate_idx+1]
            variate_target = y_target[:, variate_idx:variate_idx+1] if y_target is not None else None

            # Train the model for this variate
            variate_model._train(
                y_context=variate_context,
                y_target=variate_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs
            )

            # Store the trained model
            self.models.append(variate_model)

        # Mark the main model as fitted for anyvariate case
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray],
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained ARIMA model. Supports anyvariate mode.
        
        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for y_context
            timestamps_target: Timestamps for the prediction horizon
            freq: Frequency string
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Model predictions
        """
        # Ensure y_context is 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        n_variates = y_context.shape[1]

        # For univariate case or non-anyvariate mode, use the original method
        if n_variates == 1 or not self.is_anyvariate:
            return self._predict(y_context, timestamps_context, timestamps_target, freq, **kwargs)

        # For anyvariate mode with multivariate data, predict using each variate's model
        if not self.models:
            raise ValueError("No variate models found. Train the model first.")

        all_predictions = []
        for variate_idx in range(n_variates):
            print(f"[INFO] Predicting with ARIMA model for variate {variate_idx + 1}/{n_variates}")
            
            # Extract data for this specific variate
            variate_context = y_context[:, variate_idx:variate_idx+1]
            
            # Make predictions for this variate
            variate_pred = self.models[variate_idx]._predict(
                y_context=variate_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs
            )
            
            all_predictions.append(variate_pred)

        # Combine predictions from all variates
        combined_predictions = np.concatenate(all_predictions, axis=1)
        
        # Store for TensorBoard logging (use first variate as representative)
        self._last_y_pred = all_predictions[0]
        
        return combined_predictions

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        loss_function: str = None,
        y_train: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute loss metrics, supporting per-variate metrics in anyvariate mode.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Loss function to use (inherited from parent if None)
            y_train: Training data for metrics like MASE
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        # Ensure inputs are 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_train is not None and y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        n_variates = y_true.shape[1]

        # For univariate case or non-anyvariate mode, use the parent method
        if n_variates == 1 or not self.is_anyvariate:
            return super().compute_loss(y_true, y_pred, loss_function, y_train)

        # For anyvariate mode, compute metrics per variate
        all_metrics = {}
        aggregated_metrics = {}

        for variate_idx in range(n_variates):
            # Extract data for this variate
            y_true_variate = y_true[:, variate_idx:variate_idx+1]
            y_pred_variate = y_pred[:, variate_idx:variate_idx+1]
            y_train_variate = y_train[:, variate_idx:variate_idx+1] if y_train is not None else None

            # Compute metrics for this variate using the evaluator
            variate_metrics = self.evaluator.evaluate(y_pred_variate, y_true_variate, y_train=y_train_variate)

            # Add variate index to metric names for per-variate tracking
            for metric_name, metric_value in variate_metrics.items():
                all_metrics[f"{metric_name}_variate_{variate_idx + 1}"] = metric_value

                # Also accumulate for aggregated metrics
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = []
                aggregated_metrics[metric_name].append(metric_value)

        # Add aggregated metrics (mean across variates) for hyperparameter optimization
        for metric_name, values in aggregated_metrics.items():
            all_metrics[metric_name] = np.mean(values)

        # Store for TensorBoard logging (use first variate as representative)
        self._last_y_true = y_true[:, 0]
        self._last_y_pred = y_pred[:, 0]

        return all_metrics
