"""
SVR model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import io


class SVRModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Support Vector Regression (SVR) model with a given configuration.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor.

        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
            logger: Logger instance for TensorBoard logging
        """
        super().__init__(config)
        self.model_config = self._extract_model_config(config)

        self.scaler = StandardScaler()  # SVR is sensitive to feature scaling

        if "lookback_window" not in self.model_config:
            self.model_config["lookback_window"] = 10
        
        # Add forecast_horizon from task config
        self.model_config["forecast_horizon"] = config["task"]["forecast_horizon"]
        
        # Add training_loss from task config
        self.model_config["training_loss"] = config["task"]["tuning_loss"]

        # Extract SVR-specific parameters for the underlying SVR model
        svr_params = {}
        svr_param_names = ["kernel", "C", "epsilon", "gamma"]
        for param in svr_param_names:
            if param in self.model_config:
                svr_params[param] = self.model_config[param]

        # Store SVR params for later use
        self.svr_params = svr_params

        self._build_model()

    def _build_model(self):
        """
        Build the SVR model instance from the configuration using MultiOutputRegressor for direct multi-output forecasting.
        """
        base_svr = SVR(**self.svr_params)
        self.model = MultiOutputRegressor(base_svr)
        self.is_fitted = False

    def _create_features_targets(self, y_series):
        """
        Create features and multi-step targets for direct multi-output forecasting.
        Each sample uses the previous lookback_window values as features and the next forecast_horizon values as targets.
        Handles (num_series, timesteps) format.
        """
        X, y = [], []
        lookback_window = self.model_config["lookback_window"]
        forecast_horizon = self.model_config["forecast_horizon"]

        # Handle (num_series, timesteps) format
        if y_series.ndim == 1:
            y_series = y_series.reshape(1, -1)
        
        num_series, timesteps = y_series.shape
        
        # Validate data length
        min_required_length = lookback_window + forecast_horizon
        if timesteps < min_required_length:
            raise ValueError(f"Not enough data for SVR. Have {timesteps} observations, need at least {min_required_length}")

        for i in range(timesteps - lookback_window - forecast_horizon + 1):
            # Extract lookback window for all series
            curr_X = y_series[:, i : i + lookback_window].flatten()
            X.append(curr_X)

            # Extract forecast horizon for all series
            curr_y = y_series[:, i + lookback_window : i + lookback_window + forecast_horizon].flatten()
            y.append(curr_y)

        X, y = np.array(X), np.array(y)

        return X, y

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "SvrModel":
        """
        Train the SVR model for direct multi-output forecasting using MultiOutputRegressor.
        """
        if self.is_fitted is None:
            self._build_model()

        # Combine context and target for full training series if y_target is provided
        # Handle (num_series, timesteps) format
        lookback_window = self.model_config["lookback_window"]
        forecast_horizon = self.model_config["forecast_horizon"]
        y_series = np.concatenate([y_context, y_target], axis=1)
        
        print(f"SVR training data shape: {y_series.shape}")
        print(f"Lookback window: {lookback_window}")
        print(f"Forecast horizon: {forecast_horizon}")

        X, y = self._create_features_targets(y_series)

        # Scale features (SVR is sensitive to feature scaling)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """

        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        if y_context.shape[1] < self.model_config["lookback_window"]:
            raise ValueError(
                f"y_context too short: {y_context.shape[1]} < lookback_window {self.model_config['lookback_window']}"
            )

        total_steps = len(timestamps_target)
        num_targets = y_context.shape[0]  # num_series is first dimension
        lookback_window = self.model_config["lookback_window"]
        forecast_horizon = self.model_config["forecast_horizon"]

        # Ensure y_context is (num_series, timesteps)
        if y_context.ndim == 1:
            y_context = y_context.reshape(1, -1)

        preds = []
        context = y_context.copy()

        steps = forecast_horizon
        steps_done = 0

        while steps_done < total_steps:
            # Use last lookback_window timesteps
            current_window = context[:, -lookback_window:]
            
            # Flatten for prediction
            y_flat = np.expand_dims(current_window.flatten(), axis=0)
            pred = self.model.predict(y_flat)
            
            # Reshape prediction to (forecast_horizon, num_targets)
            pred = np.reshape(pred, (forecast_horizon, num_targets))
            
            # Transpose to (num_targets, forecast_horizon) for concatenation
            pred = pred.T
            preds.append(pred)

            # Concatenate along time axis (axis=1)
            context = np.concatenate([context, pred], axis=1)
            steps_done += steps

        # Concatenate all predictions along time axis
        preds = np.concatenate(preds, axis=1)
        
        # Return (num_targets, total_steps) - truncate if needed
        if preds.shape[1] > total_steps:
            preds = preds[:, :total_steps]

        return preds
