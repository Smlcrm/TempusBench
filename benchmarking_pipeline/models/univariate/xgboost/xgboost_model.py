"""
XGBoost model implementation for time series forecasting.
"""

import os
import pickle
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class XgboostModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost model with a given configuration.

        Args:
            config: Configuration dictionary for XGBoostRegressor parameters.
                    e.g., {'n_estimators': 100, 'learning_rate': 0.1, 'lookback_window': 10}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config)

        if "lookback_window" not in self.model_config:
            raise ValueError("lookback_window must be specified in config")

        # forecast_horizon is inherited from parent class (BaseModel)
        self._build_model()

    def _build_model(self):
        """
        Build the XGBRegressor model instance from the configuration.
        """
        # Get hyperparameters from config.
        model_config = self.model_config

        # Extract XGBoost-specific parameters
        xgb_params = {}
        for key in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "random_state",
            "n_jobs",
        ]:
            if key in model_config:
                xgb_params[key] = model_config[key]

        # Ensure random_state for reproducibility if not provided
        if "random_state" not in xgb_params:
            xgb_params["random_state"] = 42

        self.model = XGBRegressor(**xgb_params)
        self.is_fitted = False

    def _create_features(self, y_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create simple time series features for XGBoost.

        Args:
            y_series: Target time series

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """

        n_samples = len(y_series) - self.model_config["lookback_window"]

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {self.model_config['lookback_window'] + 1} samples."
            )

        features = []
        targets = []

        for i in range(n_samples):
            # Create lag features from historical data only
            lag_features = y_series[i : i + self.model_config["lookback_window"]]

            # Create simple rolling statistics
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)

            # Create trend feature (simple difference)
            trend = lag_features[-1] - lag_features[0]

            # Combine features
            sample_features = list(lag_features) + [rolling_mean, rolling_std, trend]

            features.append(sample_features)
            # Target is the next value
            targets.append(y_series[i + self.model_config["lookback_window"]])

        return np.array(features), np.array(targets)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray = None,
        x_context: np.ndarray = None,
        x_target: np.ndarray = None,
        **kwargs,
    ) -> "XgboostModel":
        """
        Train the XGBoost model on given data.

        TECHNIQUE: Simple lag-based regression
        - Creates lag features from historical target values
        - Adds basic rolling statistics (mean, std)
        - Includes simple trend feature
        - Uses XGBoost for regression on engineered features

        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values (optional, for validation)

        Returns:
            self: The fitted model instance.
        """
        if not self.is_fitted:
            self._build_model()

        y_context = np.squeeze(y_context)

        # Create features and targets
        X, y = self._create_features(y_context)

        print(
            f"Training XGBoost model with {X.shape[0]} samples and {X.shape[1]} features..."
        )

        # Train model for single-step prediction
        self.model.fit(X, y)

        self.is_fitted = True
        print("Training complete.")
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
        Make predictions using the trained XGBoost model.

        TECHNIQUE: Iterative single-step prediction
        - Uses last lookback_window values to create features
        - Predicts one step ahead
        - Updates the window with the prediction
        - Repeats until forecast_steps are reached

        Args:
            y_context: Past target values (time series) - used for prediction
            y_target: Future target values - used to determine prediction length

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, 1)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        y_context = np.squeeze(y_context)

        forecast_horizon = len(timestamps_target)
        # Use the last lookback_window values to start
        window = list(y_context[-self.model_config["lookback_window"] :])
        predictions = []

        # Make iterative predictions
        for step in range(forecast_horizon):
            # Create features for the current window
            lag_features = np.array(window)
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)
            trend = lag_features[-1] - lag_features[0]

            sample_features = list(lag_features) + [rolling_mean, rolling_std, trend]
            X_pred = np.array(sample_features).reshape(1, -1)

            # Make single-step prediction
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)

            # Update window for next prediction (remove oldest, add newest)
            window = window[1:] + [pred]

        return np.array(predictions).reshape(1, -1)
