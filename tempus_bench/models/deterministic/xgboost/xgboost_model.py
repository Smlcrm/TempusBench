"""
Multivariate XGBoost model implementation for time series forecasting.

This model extends the univariate XGBoost to handle multiple target variables simultaneously.
Uses sklearn's MultiOutputRegressor to handle multiple targets with advanced feature engineering.
"""
import os
import pickle

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from ...base_model import BaseModel, validate_inputs


class XgboostHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    n_estimators: int = Field(..., ge=1, description="Number of boosting rounds")
    max_depth: int = Field(..., ge=1, description="Maximum tree depth")
    learning_rate: float = Field(..., gt=0, description="Learning rate")
    # Fixed Hyperparameters - Optional for User to override
    reg_alpha: float = Field(default=0.0, ge=0, description="L1 regularization strength (optional)")
    reg_lambda: float = Field(default=1.0, ge=0, description="L2 regularization strength (optional)")
    subsample: float = Field(default=0.8, gt=0, le=1, description="Subsample ratio of the training instances (optional)")
    colsample_bytree: float = Field(default=0.8, gt=0, le=1, description="Subsample ratio of columns for each tree (optional)")
    min_child_weight: int = Field(default=1, ge=1, description="Minimum sum of instance weight needed in a child (optional)")
    gamma: float = Field(default=0.0, ge=0, description="Minimum loss reduction required to make a further partition (optional)")


class XGBoostModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize XGBoost model with a given configuration.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, XgboostHyperparams)
        self._build_model()

    def _create_multivariate_features(
        self, y_series: np.ndarray, **kwargs: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create advanced multivariate time series features for XGBoost.

        Args:
            y_series: Target time series with shape (num_steps, num_targets)
            x_series: Exogenous variables (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets arrays
        """

        num_steps, num_targets = y_series.shape
        forecast_horizon = kwargs["forecast_horizon"]
        lookback_window = self.lookback_window

        n_samples = (
            num_steps
            - lookback_window
            - forecast_horizon
            + 1
        )

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {lookback_window + forecast_horizon} samples."
            )

        features = []
        targets = []

        # Calculate expected feature dimension for consistency
        lag_features = lookback_window * num_targets
        rolling_stats = num_targets * 8  # 8 stats per target
        cross_corr = (num_targets * (num_targets - 1) // 2) * 2 if num_targets > 1 else 0
        temporal = num_targets if lookback_window >= 7 else 0
        expected_feature_dim = lag_features + rolling_stats + cross_corr + temporal

        # Debug: let's use the actual counts we're getting
        actual_feature_dim = lag_features + rolling_stats + cross_corr + temporal
        self.logger.debug("XGBoostModel._create_features", f"Expected feature dim: {expected_feature_dim} (lag:{lag_features}, stats:{rolling_stats}, cross:{cross_corr}, temp:{temporal})")
        self.logger.debug("XGBoostModel._create_features", f"Actual feature dim: {actual_feature_dim}")

        # Use the actual dimension for consistency
        expected_feature_dim = actual_feature_dim

        for i in range(n_samples):
            sample_features = []

            # Get lookback window for all targets
            lookback_data = y_series[
                i : i + lookback_window, :
            ]  # Shape: (lookback_window, num_targets)

            # 1. Lag features for each target (flattened)
            lag_features = (
                lookback_data.flatten()
            )  # Shape: (lookback_window * num_targets,)
            sample_features.extend(lag_features)

            # 2. Rolling statistics for each target individually
            for target_idx in range(num_targets):
                target_data = lookback_data[:, target_idx]  # Shape: (lookback_window,)

                # Basic statistics
                rolling_mean = np.mean(target_data)
                rolling_std = np.std(target_data)
                rolling_min = np.min(target_data)
                rolling_max = np.max(target_data)
                rolling_median = np.median(target_data)

                # Trend features
                trend = np.polyfit(
                    range(len(target_data)), target_data, 1
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
            cross_corr_count = 0
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
                        cross_corr_count += 1

                        # Ratio features
                        mean_i = np.mean(lookback_data[:, i_target])
                        mean_j = np.mean(lookback_data[:, j_target])
                        if mean_j != 0:
                            ratio = mean_i / mean_j
                        else:
                            ratio = 0.0
                        sample_features.append(ratio)
                        cross_corr_count += 1

            # 4. Temporal features (if window is large enough)
            if lookback_window >= 7:
                # Weekly patterns (last 7 values for each target)
                recent_data = lookback_data[-7:, :]  # Shape: (7, num_targets)
                for target_idx in range(num_targets):
                    recent_mean = np.mean(recent_data[:, target_idx])
                    sample_features.append(recent_mean)

            # # 5. Add exogenous features if available
            # if x_series is not None and len(x_series) > i + self._model_config['lookback_window']:
            #     current_x = x_series[i + self._model_config['lookback_window']]
            #     sample_features.extend(current_x.flatten())

            # Debug feature counts
            lag_count = lookback_window * num_targets
            stats_count = num_targets * 8
            temp_count = num_targets if lookback_window >= 7 else 0
            self.logger.debug("XGBoostModel._create_features", f"Sample {i}: lag={lag_count}, stats={stats_count}, cross={cross_corr_count}, temp={temp_count}, total={len(sample_features)}")

            features.append(sample_features)

            # Create target: next forecast_horizon steps for all targets, flattened
            start_idx = i + lookback_window
            end_idx = start_idx + forecast_horizon
            future_values = y_series[
                start_idx:end_idx, :
            ]  # Shape: (forecast_horizon, num_targets)
            target_flat = future_values.flatten()  # Shape: (forecast_horizon * num_targets,)
            self.logger.debug("XGBoostModel._create_features", f"Sample {i} target shape: {future_values.shape}, flattened: {target_flat.shape}")
            targets.append(target_flat)

        # Ensure all arrays have consistent shapes
        features_array = np.array(features)
        targets_array = np.array(targets)

        self.logger.debug("XGBoostModel._create_features", f"Features shape: {features_array.shape}")
        self.logger.debug("XGBoostModel._create_features", f"Targets shape: {targets_array.shape}")

        return features_array, targets_array

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "XGBoostModel":
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
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]

        # Reference params, settings, device, python_version
        lookback_window = self._model_config["lookback_window"]

        if self._model is None:
            self._build_model()

        # Use full target data if available and has more data than context
        if y_target is not None and y_target.shape[1] > y_context.shape[1]:
            training_data = y_target
        else:
            training_data = y_context

        self.logger.debug("XGBoostModel.train", f"XGBoost training data shape: {training_data.shape}")
        self.logger.debug("XGBoostModel.train", f"Lookback window: {lookback_window}")
        self.logger.debug("XGBoostModel.train", f"Forecast horizon: {forecast_horizon}")

        # Prepare features for training
        X, y = self._create_multivariate_features(training_data, **kwargs)

        # Train the model
        self._model.fit(X, y)
        self.is_fitted = True

        return self

    def rolling_predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Autoregressive rolling prediction for Multivariate XGBoost.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.

        Args:
            y_context: Historical context data with shape (timesteps, num_targets)
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Predictions with shape (num_targets, forecast_steps)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]

        # Reference params, settings, device, python_version
        lookback_window = self._model_config["lookback_window"]

        preds = []
        # Use the entire context, maintaining the multivariate structure
        context = y_context.copy()
        num_targets = y_context.shape[1]  # Fix: num_targets is the second dimension
        total_steps = len(timestamps_target)
        steps_done = 0

        while steps_done < total_steps:
            steps = min(forecast_horizon, total_steps - steps_done)

            # Take the last lookback_window timesteps for feature creation
            current_window = context[-lookback_window:, :]  # Shape: (lookback_window, num_targets)

            try:
                # Create features using the same method as training
                # current_window is already in the correct format: (lookback_window, num_targets)
                sample_features = self._create_single_sample_features(current_window)

                # Convert to numpy array and predict
                X_pred = sample_features.reshape(1, -1)

                # Fail fast on NaN input - don't silently replace
                if np.isnan(X_pred).any():
                    raise ValueError(
                        "Prediction features contain NaN values. This indicates data corruption."
                    )

                pred_flat = self._model.predict(X_pred).flatten()

                # Fail fast on NaN predictions - don't silently replace
                if np.isnan(pred_flat).any():
                    raise ValueError(
                        "Model produced NaN predictions. This indicates a training or data issue."
                    )

                # Reshape predictions back to (num_targets, forecast_horizon)
                pred_reshaped = pred_flat.reshape(
                    num_targets, forecast_horizon
                )

                self.logger.debug(
                    "XGBoostModel.rolling_predict",
                    f"Rolling predict X_pred shape: {X_pred.shape}, pred_reshaped shape: {pred_reshaped.shape}"
                )

                # Only take as many steps as needed
                pred_steps = pred_reshaped[:, :steps]  # Shape: (num_targets, steps)
                preds.append(pred_steps)

                # Update context with new predictions
                # pred_steps needs to be transposed to (steps, num_targets) to match context format
                pred_steps_transposed = pred_steps.T  # Shape: (steps, num_targets)
                context = np.concatenate([context, pred_steps_transposed], axis=0)  # Concatenate along time axis
                steps_done += steps

            except Exception as e:
                raise RuntimeError(f"Error during prediction step: {str(e)}")

        # Concatenate all predictions along time axis
        result = np.concatenate(preds, axis=1)  # Shape: (num_targets, total_steps)
        return result

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
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
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]

        # Reference params, settings, device, python_version
        lookback_window = self._model_config["lookback_window"]

        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        # Use rolling prediction to cover the full test set (no exogenous variables)
        preds = self.rolling_predict(
            y_context, timestamps_context, timestamps_target, **kwargs
        )
        self.logger.debug("XGBoostModel.predict", f"Final rolling preds shape: {preds.shape}")
        return preds

    def _build_model(self):
        """
        Build the XGBRegressor model instance from the configuration.
        """
        all_params = {
            **self.params,
            **self.settings
        }

        self._model = XGBRegressor(**all_params)
        self.is_fitted = False

    def _create_single_sample_features(self, y_series: np.ndarray) -> np.ndarray:
        """
        Create features for a single sample (used by both training and prediction).

        Args:
            y_series: Target time series with shape (num_steps, num_targets)

        Returns:
            np.ndarray: Feature vector for the single sample
        """
        num_targets = y_series.shape[1]
        lookback_window = self.lookback_window

        # Take the last lookback_window timesteps
        current_window = y_series[-lookback_window:, :]  # Shape: (lookback_window, num_targets)

        sample_features = []

        # 1. Lag features (flattened)
        lag_features = current_window.flatten()
        sample_features.extend(lag_features)
        # 2. Rolling statistics for each target
        stats_count = 0
        for target_idx in range(num_targets):
            target_data = current_window[:, target_idx]  # Shape: (lookback_window,)

            rolling_mean = np.mean(target_data)
            rolling_std = np.std(target_data)
            rolling_min = np.min(target_data)
            rolling_max = np.max(target_data)
            rolling_median = np.median(target_data)
            trend = np.polyfit(range(len(target_data)), target_data, 1)[0]
            rolling_range = rolling_max - rolling_min
            rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)

            sample_features.extend([
                rolling_mean, rolling_std, rolling_min, rolling_max,
                rolling_median, trend, rolling_range, rolling_iqr
            ])
            stats_count += 8

        # 3. Cross-correlation features (if multivariate)
        cross_count = 0
        if num_targets > 1:
            for i_target in range(num_targets):
                for j_target in range(i_target + 1, num_targets):
                    corr = np.corrcoef(current_window[:, i_target], current_window[:, j_target])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    sample_features.append(corr)
                    cross_count += 1

                    mean_i = np.mean(current_window[:, i_target])
                    mean_j = np.mean(current_window[:, j_target])
                    ratio = mean_i / mean_j if mean_j != 0 else 0.0
                    sample_features.append(ratio)
                    cross_count += 1

        # 4. Temporal features (if applicable)
        temp_count = 0
        if lookback_window >= 7:
            recent_data = current_window[-7:, :]  # Shape: (7, num_targets)
            for target_idx in range(num_targets):
                recent_mean = np.mean(recent_data[:, target_idx])
                sample_features.append(recent_mean)
                temp_count += 1

        return np.array(sample_features)
