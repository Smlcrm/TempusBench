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


class SvrModel(BaseModel):
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

        self.scaler = StandardScaler()  # SVR is sensitive to feature scaling

        if "lookback_window" not in self.model_config:
            self.model_config["lookback_window"] = 10
        # Extract model-level parameters (not SVR-specific)

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
        """
        X, y = [], []
        lookback_window = self.model_config["lookback_window"]
        forecast_horizon = self.model_config["forecast_horizon"]

        for i in range(len(y_series) - lookback_window - forecast_horizon + 1):
            X.append(y_series[i : i + lookback_window])
            y.append(
                y_series[i + lookback_window : i + lookback_window + forecast_horizon]
            )
        return np.array(X), np.array(y)

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
        if not self.is_fitted is None:
            self._build_model()

        # Combine context and target for full training series if y_target is provided
        # Ensure both arrays are 1D before concatenation
        lookback_window = self.model_config["lookback_window"]
        forecast_horizon = self.model_config["forecast_horizon"]
        y_context = y_context.squeeze()
        y_target = y_target.squeeze()
        y_series = np.concatenate([y_context, y_target])

        X, y = self._create_features_targets(y_series)
        print(f"[DEBUG][SVR] Training X shape: {X.shape}, y shape: {y.shape}")

        # Scale features (time index is not used; features are lagged values)
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

        if len(y_context) < self.model_config["lookback_window"]:
            raise ValueError(
                f"y_context too short: {len(y_context)} < lookback_window {self.model_config['lookback_window']}"
            )

        lookback_window = self.model_config["lookback_window"]
        forecast_horizon = self.model_config["forecast_horizon"]

        y_context = y_context

        preds = []
        context = np.array(y_context).flatten().tolist()
        total_steps = len(timestamps_target)
        steps = forecast_horizon
        steps_done = 0

        while steps_done < total_steps:

            y_context = y_context.squeeze()[-lookback_window:]
            y_context_scaled = self.scaler.transform(np.expand_dims(y_context, axis=0))
            y_context_scaled = y_context_scaled
            pred = self.model.predict(y_context_scaled).squeeze()

            preds.extend(pred)

            y_context = np.concatenate([y_context, pred])
            steps_done += steps

        preds = np.array(preds)
        preds = np.expand_dims(preds, axis=1)

        return preds
