"""
Random Forest model implementation for time series forecasting
"""

import numpy as np
import pandas as pd

from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import BaseModel as PydanticBaseModel, Field
from sklearn.ensemble import RandomForestRegressor

from tempus_bench.models.base_model import BaseModel, validate_inputs


class RandomForestHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    n_estimators: int = Field(..., ge=1, description="Number of trees in the forest")
    max_depth: int = Field(..., ge=1, description="Maximum depth of trees")
    # Fixed Hyperparameters - Optional for User to override
    min_samples_split: int = Field(
        default=2, ge=2, description="Minimum samples to split a node"
    )
    min_samples_leaf: int = Field(
        default=1, ge=1, description="Minimum samples in a leaf"
    )
    max_features: Literal["sqrt", "log2", "auto"] = Field(
        default="sqrt", description="Number of features to consider for splits"
    )


class RandomForestModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Random Forest model with model-specific parameters.
        """
        super().__init__(params, settings, RandomForestHyperparams)

        self._build_model(params, settings)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> "RandomForestModel":
        """
        Train the Random Forest model on given data.

        TECHNIQUE: Single Model with Timestamp Features
        - Creates lag features from historical target values
        - Adds rolling statistics (mean, std, min, max)
        - Includes trend features using linear regression
        - Incorporates timestamp features for time-aware splits
        - Uses a single model to predict all forecast horizon steps

        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values (optional, for validation)
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance.
        """

        # Calculate forecast_horizon from y_target if not provided in kwargs
        forecast_horizon = kwargs.get("forecast_horizon", y_target.shape[0])

        # Concatenate along time axis (axis=0) for our (num_steps, num_targets) format
        full_y_data = np.concatenate([y_context, y_target], axis=0)
        full_timestamps = np.concatenate(
            [timestamps_context, timestamps_target], axis=0
        )
        full_timestamps = np.squeeze(full_timestamps)

        X, y, effective_lookback = self._create_features(
            full_y_data, full_timestamps, forecast_horizon=forecast_horizon, **kwargs
        )

        # Store effective_lookback used during training for consistency in prediction
        self._trained_lookback_window = effective_lookback
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.

        TECHNIQUE: Single Model with Timestamp Features
        - Uses last lookback_window values to create features
        - Incorporates timestamp features for time-aware prediction
        - Predicts all forecast horizon steps with a single model

        Args:
            y_context: Past target values (time series) - used for prediction
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        full_timestamps = np.concatenate(
            [timestamps_context, timestamps_target], axis=0
        )
        full_timestamps = np.squeeze(full_timestamps)
        num_targets = y_context.shape[1]
        dummy_future = np.zeros((forecast_horizon, num_targets))
        full_y_data = np.concatenate([y_context, dummy_future], axis=0)

        feature_row, _, _ = self._create_features(
            full_y_data, full_timestamps, forecast_horizon=forecast_horizon, **kwargs
        )

        X_last = feature_row[-1:].reshape(1, -1)

        # Predict all steps at once
        preds = self._model.predict(X_last)

        # Reshape predictions back to (forecast_horizon, num_targets)
        # The model predicts forecast_horizon * num_targets values
        preds_reshaped = preds.reshape(forecast_horizon, num_targets)

        # Inverse transform predictions if scaler is available
        if hasattr(self, "_scaler") and self._scaler is not None:
            preds_reshaped = self._scaler.inverse_transform(preds_reshaped)

        return preds_reshaped

    def _build_model(self, params, settings):
        """
        Build the RandomForestRegressor model instance from the configuration.
        """
        # Get hyperparameters from params, excluding non-estimator parameters
        # Convert Pydantic model to dict if needed
        all_params = {**params, **settings}

        # Filter out keys that are not valid for RandomForestRegressor
        valid_keys = set(RandomForestRegressor().get_params().keys())
        filtered_params = {k: v for k, v in all_params.items() if k in valid_keys}

        self._model = RandomForestRegressor(**filtered_params)

        self.is_fitted = False

    def _create_features(
        self, y_series: np.ndarray, timestamps: np.ndarray, **kwargs: dict
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Create time series features for Random Forest with timestamp features.

        Args:
            y_series: Target time series
            x_series: Exogenous variables (optional)
            timestamps: Timestamp features (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Features, targets, and effective_lookback
        """
        num_steps, num_targets = y_series.shape
        forecast_horizon = kwargs.get("forecast_horizon", None)
        if forecast_horizon is None:
            raise ValueError(
                "forecast_horizon must be provided in kwargs or as a parameter"
            )

        # Use stored lookback_window from training if available (for prediction consistency)
        # Otherwise, adapt lookback_window to available data
        if hasattr(self, "_trained_lookback_window"):
            effective_lookback = self._trained_lookback_window
        else:
            # Adapt lookback_window to available data if needed
            # Ensure we have at least forecast_horizon + 1 steps for creating features
            max_lookback = num_steps - forecast_horizon
            effective_lookback = (
                min(self.lookback_window, max_lookback) if max_lookback > 0 else 1
            )

        num_samples = num_steps - effective_lookback - forecast_horizon + 1

        if num_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {effective_lookback + forecast_horizon} samples, "
                f"but only have {num_steps} steps (lookback_window={self.lookback_window})."
            )

        features = []
        targets = []

        for i in range(num_samples):
            # Create lag features for all targets
            lag_features = y_series[i : i + effective_lookback].flatten()

            # Create rolling statistics for each target
            sample_features = []
            sample_features.extend(lag_features)

            for target_idx in range(num_targets):
                target_data = y_series[i : i + effective_lookback, target_idx]

                rolling_mean = np.mean(target_data)
                rolling_std = np.std(target_data)
                rolling_min = np.min(target_data)
                rolling_max = np.max(target_data)

                # Create trend features
                trend = np.polyfit(range(len(target_data)), target_data, 1)[0]

                # Create volatility features
                rolling_range = rolling_max - rolling_min
                rolling_iqr = np.percentile(target_data, 75) - np.percentile(
                    target_data, 25
                )

                # Add features for this target
                sample_features.extend(
                    [
                        rolling_mean,
                        rolling_std,
                        rolling_min,
                        rolling_max,
                        trend,
                        rolling_range,
                        rolling_iqr,
                    ]
                )

            # Add timestamp features if available
            if (
                timestamps is not None
                and len(timestamps) >= i + effective_lookback + forecast_horizon
            ):
                # Add current timestamp and future timestamps as features
                current_timestamp = timestamps[i + effective_lookback - 1]
                future_timestamps = timestamps[
                    i + effective_lookback : i + effective_lookback + forecast_horizon
                ]

                # Convert timestamps to numerical features
                # Convert numpy.datetime64 to pandas Timestamp
                if isinstance(current_timestamp, np.datetime64):
                    current_timestamp = pd.to_datetime(current_timestamp)
                if isinstance(current_timestamp, pd.Timestamp):
                    current_time_features = [
                        current_timestamp.year,
                        current_timestamp.month,
                        current_timestamp.day,
                        current_timestamp.hour,
                        current_timestamp.dayofweek,
                        current_timestamp.dayofyear,
                    ]
                else:
                    # If timestamps are already numerical, use as is
                    current_time_features = [current_timestamp]

                # Add future timestamp features
                future_time_features = []
                for ts in future_timestamps:
                    if isinstance(ts, np.datetime64):
                        ts = pd.to_datetime(ts)
                    if isinstance(ts, pd.Timestamp):
                        future_time_features.extend(
                            [
                                ts.year,
                                ts.month,
                                ts.day,
                                ts.hour,
                                ts.dayofweek,
                                ts.dayofyear,
                            ]
                        )
                    else:
                        future_time_features.append(ts)

                sample_features.extend(current_time_features + future_time_features)

            features.append(sample_features)
            # Multi-output: target is a vector of length forecast_horizon * num_targets
            target_values = []
            for step in range(forecast_horizon):
                for target_idx in range(num_targets):
                    target_values.append(
                        y_series[i + effective_lookback + step, target_idx]
                    )
            targets.append(target_values)

        return np.array(features), np.array(targets), effective_lookback
