"""
SVR model implementation.
"""
import os
import pickle

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ...base_model import BaseModel, validate_inputs


class SvrHyperparams(PydanticBaseModel):
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = Field(default="rbf", description="SVR kernel type")
    C: float = Field(default=1.0, gt=0, description="Regularization parameter")
    epsilon: Optional[float] = Field(default=0.1, ge=0, description="Epsilon parameter for epsilon-SVR")
    gamma: Optional[Literal["scale", "auto"]] = Field(default="scale", description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'")


class SVRModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Support Vector Regression (SVR) model with model-specific parameters.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor.
        """
        super().__init__(params, settings, SvrHyperparams)
        self._scaler = StandardScaler()  # SVR is sensitive to feature scaling
        self._build_model()

    def _build_model(self):
        """
        Build the SVR model instance from the configuration using MultiOutputRegressor for direct multi-output forecasting.
        """
        # Build base SVR from params fields relevant to estimator
        base_svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self._model = MultiOutputRegressor(base_svr)
        self.is_fitted = False

    def _create_features_targets(self, y_series: np.ndarray, forecast_horizon: int, lookback_window: int):
        """
        Create features and multi-step targets for direct multi-output forecasting.
        Each sample uses the previous lookback_window values as features and the next forecast_horizon values as targets.
        Handles (num_series, timesteps) format.
        """
        X, y = [], []
        lookback_window = int(lookback_window)
        forecast_horizon = int(forecast_horizon)

        # Handle (num_steps, num_targets) format
        if y_series.ndim == 1:
            y_series = y_series.reshape(-1, 1)

        num_steps, num_targets = y_series.shape

        # Validate data length
        min_required_length = lookback_window + forecast_horizon
        if num_steps < min_required_length:
            raise ValueError(f"Not enough data for SVR. Have {num_steps} observations, need at least {min_required_length}")

        for i in range(num_steps - lookback_window - forecast_horizon + 1):
            # Extract lookback window for all features
            curr_X = y_series[i : i + lookback_window, :].flatten()
            X.append(curr_X)

            # Extract forecast horizon for all features
            curr_y = y_series[i + lookback_window : i + lookback_window + forecast_horizon, :].flatten()
            y.append(curr_y)

        X, y = np.array(X), np.array(y)

        return X, y

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "SVRModel":
        """
        Train the SVR model for direct multi-output forecasting using MultiOutputRegressor.
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        lookback_window = self.settings["lookback_window"]
        
        if self.is_fitted is None:
            self._build_model()

        # Combine context and target for full training series if y_target is provided
        # Handle (num_steps, num_targets) format
        y_series = np.concatenate([y_context, y_target], axis=0)

        self.logger.debug("SVRModel.train", f"SVR training data shape: {y_series.shape}")
        self.logger.debug("SVRModel.train", f"Lookback window: {lookback_window}")
        self.logger.debug("SVRModel.train", f"Forecast horizon: {forecast_horizon}")

        X, y = self._create_features_targets(
            y_series,
            forecast_horizon=forecast_horizon,
            lookback_window=lookback_window
        )

        # Scale features (SVR is sensitive to feature scaling)
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)
        self._model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        lookback_window = self.settings["lookback_window"]
        
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        if y_context.shape[0] < int(lookback_window):
            raise ValueError(
                f"y_context too short: {y_context.shape[0]} < lookback_window {lookback_window}"
            )

        total_steps = len(timestamps_target)
        num_targets = y_context.shape[1]  # num_targets is second dimension
        lookback_window = int(lookback_window)
        forecast_horizon = int(forecast_horizon)

        # Ensure y_context is (num_steps, num_targets)
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        preds = []
        context = y_context.copy()

        steps = forecast_horizon
        steps_done = 0

        while steps_done < total_steps:
            # Use last lookback_window timesteps
            current_window = context[-lookback_window:, :]

            # Flatten for prediction
            y_flat = np.expand_dims(current_window.flatten(), axis=0)
            pred = self._model.predict(y_flat)

            # Reshape prediction to (forecast_horizon, num_targets)
            pred = np.reshape(pred, (forecast_horizon, num_targets))

            preds.append(pred)

            # Concatenate along time axis (axis=0)
            context = np.concatenate([context, pred], axis=0)
            steps_done += steps

        # Concatenate all predictions along time axis
        preds = np.concatenate(preds, axis=0)

        # Return (total_steps, num_targets) - truncate if needed
        if preds.shape[0] > total_steps:
            preds = preds[:total_steps, :]

        # Note: Do not inverse-transform targets with feature scaler; predictions are in original target scale
        return preds
