"""
Multivariate XGBoost model implementation for time series forecasting.

This model extends the univariate XGBoost to handle multiple target variables simultaneously.
Uses sklearn's MultiOutputRegressor to handle multiple targets with advanced feature engineering.
"""

import os
import pickle
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class MultivariateXGBoostModel(BaseModel):
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

    def _create_multivariate_features(
        self, y_series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create advanced multivariate time series features for XGBoost.

        Args:
            y_series: Target time series with shape (timesteps, num_targets)
            x_series: Exogenous variables (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets arrays
        """

        num_targets = y_series.shape[1]
        n_samples = (
            len(y_series)
            - self.model_config["lookback_window"]
            - self.model_config["forecast_horizon"]
            + 1
        )

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {self.model_config['lookback_window'] + self.model_config['forecast_horizon']} samples."
            )

        features = []
        targets = []

        for i in range(n_samples):
            sample_features = []

            # Get lookback window for all targets
            lookback_data = y_series[
                i : i + self.model_config["lookback_window"]
            ]  # Shape: (lookback_window, num_targets)

            # 1. Lag features for each target (flattened)
            lag_features = (
                lookback_data.flatten()
            )  # Shape: (lookback_window * num_targets,)
            sample_features.extend(lag_features)

            # 2. Rolling statistics for each target individually
            for target_idx in range(num_targets):
                target_data = lookback_data[:, target_idx]

                # Basic statistics
                rolling_mean = np.mean(target_data)
                rolling_std = np.std(target_data)
                rolling_min = np.min(target_data)
                rolling_max = np.max(target_data)
                rolling_median = np.median(target_data)

                # Trend features
                trend = np.polyfit(
                    range(self.model_config["lookback_window"]), target_data, 1
                )[0]

                # Volatility features
                rolling_range = rolling_max - rolling_min
                rolling_iqr = np.percentile(target_data, 75) - np.percentile(
                    target_data, 25
                )

                sample_features.extend(
                    [
                        rolling_mean,
                        rolling_std,
                        rolling_min,
                        rolling_max,
                        rolling_median,
                        trend,
                        rolling_range,
                        rolling_iqr,
                    ]
                )

            # 3. Cross-correlation features between targets (if multivariate)
            if num_targets > 1:
                for i_target in range(num_targets):
                    for j_target in range(i_target + 1, num_targets):
                        # Correlation between target pairs
                        corr = np.corrcoef(
                            lookback_data[:, i_target], lookback_data[:, j_target]
                        )[0, 1]
                        if np.isnan(corr):
                            corr = 0.0  # Handle constant series
                        sample_features.append(corr)

                        # Ratio features
                        mean_i = np.mean(lookback_data[:, i_target])
                        mean_j = np.mean(lookback_data[:, j_target])
                        if mean_j != 0:
                            ratio = mean_i / mean_j
                        else:
                            ratio = 0.0
                        sample_features.append(ratio)

            # 4. Temporal features (if window is large enough)
            if self.model_config["lookback_window"] >= 7:
                # Weekly patterns (last 7 values for each target)
                recent_data = lookback_data[-7:]
                for target_idx in range(num_targets):
                    recent_mean = np.mean(recent_data[:, target_idx])
                    sample_features.append(recent_mean)

            # # 5. Add exogenous features if available
            # if x_series is not None and len(x_series) > i + self.model_config['lookback_window']:
            #     current_x = x_series[i + self.model_config['lookback_window']]
            #     sample_features.extend(current_x.flatten())

            features.append(sample_features)

            # Create target: next forecast_horizon steps for all targets, flattened
            future_values = y_series[
                i
                + self.model_config["lookback_window"] : i
                + self.model_config["lookback_window"]
                + self.model_config["forecast_horizon"]
            ]
            targets.append(
                future_values.flatten()
            )  # Shape: (forecast_horizon * num_targets,)

        return np.array(features), np.array(targets)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "MultivariateXGBoostModel":
        """
        Train the Multivariate XGBoost model for direct multi-output forecasting.

        TECHNIQUE: Advanced Multivariate Feature Engineering with Gradient Boosting
        - Creates lag features from all target variables
        - Adds rolling statistics per target (mean, std, min, max, median, trend, range, IQR)
        - Includes cross-correlation and ratio features between target pairs
        - Incorporates temporal patterns (weekly means if window ≥ 7)
        - Uses XGBoost's gradient boosting with MultiOutputRegressor for non-linear pattern learning

        Args:
            y_context: Past target values (DataFrame for multivariate)
            y_target: Future target values (optional, for validation)
            y_start_date: The start date timestamp for y_context and y_target
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        if self.model is None:
            self._build_model()

        # Prepare features for training
        X, y = self._create_multivariate_features(y_context)

        # Train the model
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def rolling_predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Autoregressive rolling prediction for Multivariate XGBoost.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.

        Args:
            y_context: Historical context data with shape (timesteps, num_targets)
            y_target: Target data to determine prediction length

        Returns:
            np.ndarray: Predictions with shape (forecast_steps, num_targets)
        """

        preds = []
        # Use the entire context, maintaining the multivariate structure
        context = y_context.copy()
        num_targets = y_context.shape[1]
        total_steps = len(timestamps_target)
        steps_done = 0

        while steps_done < total_steps:
            steps = min(self.model_config["forecast_horizon"], total_steps - steps_done)

            # Take the last lookback_window timesteps for feature creation
            current_window = context[-self.model_config["lookback_window"] :]

            try:
                # Create features using the current window
                sample_features = []

                # 1. Lag features (flattened)
                lag_features = current_window.flatten()
                sample_features.extend(lag_features)

                # 2. Rolling statistics for each target
                for target_idx in range(num_targets):
                    target_data = current_window[:, target_idx]

                    rolling_mean = np.mean(target_data)
                    rolling_std = np.std(target_data)
                    rolling_min = np.min(target_data)
                    rolling_max = np.max(target_data)
                    rolling_median = np.median(target_data)
                    trend = np.polyfit(
                        range(self.model_config["lookback_window"]), target_data, 1
                    )[0]
                    rolling_range = rolling_max - rolling_min
                    rolling_iqr = np.percentile(target_data, 75) - np.percentile(
                        target_data, 25
                    )

                    sample_features.extend(
                        [
                            rolling_mean,
                            rolling_std,
                            rolling_min,
                            rolling_max,
                            rolling_median,
                            trend,
                            rolling_range,
                            rolling_iqr,
                        ]
                    )

                # 3. Cross-correlation features (if multivariate)
                if num_targets > 1:
                    for i_target in range(num_targets):
                        for j_target in range(i_target + 1, num_targets):
                            corr = np.corrcoef(
                                current_window[:, i_target], current_window[:, j_target]
                            )[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            sample_features.append(corr)

                            mean_i = np.mean(current_window[:, i_target])
                            mean_j = np.mean(current_window[:, j_target])
                            ratio = mean_i / mean_j if mean_j != 0 else 0.0
                            sample_features.append(ratio)

                # 4. Temporal features (if applicable)
                if self.model_config["lookback_window"] >= 7:
                    recent_data = current_window[-7:]
                    for target_idx in range(num_targets):
                        recent_mean = np.mean(recent_data[:, target_idx])
                        sample_features.append(recent_mean)

                # 5. No exogenous features (removed as per cleanup)

                # Convert to numpy array and predict
                X_pred = np.array(sample_features).reshape(1, -1)

                # Fail fast on NaN input - don't silently replace
                if np.isnan(X_pred).any():
                    raise ValueError(
                        "Prediction features contain NaN values. This indicates data corruption."
                    )

                pred_flat = self.model.predict(X_pred).flatten()

                # Fail fast on NaN predictions - don't silently replace
                if np.isnan(pred_flat).any():
                    raise ValueError(
                        "Model produced NaN predictions. This indicates a training or data issue."
                    )

                # Reshape predictions back to (forecast_horizon, num_targets)
                pred_reshaped = pred_flat.reshape(
                    self.model_config["forecast_horizon"], num_targets
                )

                print(
                    f"[DEBUG][MultivariateXGBoost] Rolling predict X_pred shape: {X_pred.shape}, pred_reshaped shape: {pred_reshaped.shape}"
                )

                # Only take as many steps as needed
                pred_steps = pred_reshaped[:steps]
                preds.append(pred_steps)

                # Update context with new predictions
                context = np.concatenate([context, pred_steps], axis=0)
                steps_done += steps

            except Exception as e:
                raise RuntimeError(f"Error during prediction step: {str(e)}")

        # Concatenate all predictions
        result = np.concatenate(preds, axis=0)
        return result

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Multivariate XGBoost model.

        TECHNIQUE: Autoregressive Rolling Window Multi-step Forecasting with Advanced Features
        - Uses last lookback_window values to create comprehensive multivariate features
        - Predicts forecast_horizon steps ahead using trained MultiOutputRegressor
        - Uses its own predictions to update the window and predict further steps
        - Repeats until forecast_steps are reached

        Args:
            y_context: Historical context data
            y_target: Target data (used to determine prediction length)

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        # Use rolling prediction to cover the full test set (no exogenous variables)
        preds = self.rolling_predict(
            y_context, timestamps_context, timestamps_target, freq
        )
        print(f"[DEBUG][MultivariateXGBoost] Final rolling preds shape: {preds.shape}")
        return preds
