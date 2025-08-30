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
                svr_params[param] = self.config[param]

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

    def _create_features_targets(self, y_series, forecast_horizon):
        """
        Create features and multi-step targets for direct multi-output forecasting.
        Each sample uses the previous lookback_window values as features and the next forecast_horizon values as targets.
        """
        X, y = [], []
        lookback_window = self.model_config
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
        y_context = y_context.flatten()

        y_series = np.concatenate([y_context, y_target])

        X, y = self._create_features_targets(
            y_series, self.lookback_window, self.forecast_horizon
        )
        print(f"[DEBUG][SVR] Training X shape: {X.shape}, y shape: {y.shape}")

        # Log training progress to TensorBoard
        self._log_training_progress(X, y)

        # Scale features (time index is not used; features are lagged values)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Make direct multi-output predictions using the trained SVR model.
        If y_target is provided, use rolling_predict to predict the entire length autoregressively.
        Returns a vector of length forecast_horizon or full test length.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None:
            raise ValueError(
                "y_context must be provided to determine prediction context."
            )
        if y_target is None:
            raise ValueError(
                "y_target is required to determine prediction length. No forecast_horizon fallback allowed."
            )

        # Use rolling prediction to cover the full test set
        preds = self.rolling_predict(y_context, y_target)
        print(f"[DEBUG][SVR] Final rolling preds shape: {preds.shape}")
        return preds

    def predict(self, y_context, y_target, **kwargs):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """
        # Debug: Print initial context info
        print(f"[DEBUG][SVR] Initial y_context: {y_context}")
        print(
            f"[DEBUG][SVR] Initial y_context length: {len(y_context)}, lookback_window: {self.lookback_window}"
        )
        if len(y_context) < self.lookback_window:
            raise ValueError(
                f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}"
            )

        preds = []
        context = np.array(y_context).flatten().tolist()
        total_steps = len(y_target)
        steps_done = 0

        while steps_done < total_steps:
            steps = min(self.forecast_horizon, total_steps - steps_done)
            X_pred = np.array(context[-self.lookback_window :]).reshape(1, -1)
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = self.model.predict(X_pred_scaled).flatten()

            print(
                f"[DEBUG][SVR] Rolling predict X_pred shape: {X_pred.shape}, pred shape: {pred.shape}"
            )
            # Only take as many steps as needed
            pred = pred[:steps]
            preds.extend(pred)
            context.extend(pred)
            steps_done += steps

        # Log predictions to TensorBoard if logger is available
        if self.logger is not None and len(preds) > 0:
            try:
                # Convert y_target to array if it's not already
                y_target_array = np.array(y_target).flatten()
                preds_array = np.array(preds)

                # Log prediction plot
                self._log_predictions(
                    y_target_array, preds_array, step=0, tag="rolling_predictions"
                )

                # Log prediction metrics
                if len(y_target_array) == len(preds_array):
                    mae = np.mean(np.abs(y_target_array - preds_array))
                    rmse = np.sqrt(np.mean((y_target_array - preds_array) ** 2))
                    self.logger.log_metrics(
                        {
                            "predictions/mae": mae,
                            "predictions/rmse": rmse,
                            "predictions/num_predictions": len(preds_array),
                        },
                        step=0,
                        model_name="svr",
                    )

            except Exception as e:
                print(f"[WARNING] Failed to log predictions: {e}")

        return np.array(preds)
