"""
Random Forest model implementation for time series forecasting.
TODO-COULD work multivariate
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List, Optional
import json
import os
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Random Forest model with given configuration.

        Args:
            config: Configuration dictionary for RandomForestRegressor parameters.
                    e.g., {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'lookback_window': 10}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config)
        if "lookback_window" not in self.model_config:
            raise ValueError("lookback_window must be specified in config")

        self._build_model()

    def _build_model(self):
        """
        Build the RandomForestRegressor model instance from the configuration.
        """

        print("building model")
        # Get hyperparameters from config, excluding model-level parameters
        model_params = {}
        for key, value in self.model_config.items():
            if key not in ["lookback_window"]:
                model_params[key] = value

        # Filter out keys that are not valid for RandomForestRegressor
        valid_keys = set(RandomForestRegressor().get_params().keys())
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}

        if "random_state" not in filtered_params:
            filtered_params["random_state"] = 42

        if "n_jobs" not in filtered_params:
            filtered_params["n_jobs"] = 1

        self.model = RandomForestRegressor(**filtered_params)
        self.is_fitted = False

    def _create_features(
        self, y_series: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series features for Random Forest with timestamp features.

        Args:
            y_series: Target time series
            x_series: Exogenous variables (optional)
            timestamps: Timestamp features (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        forecast_horizon = self.model_config["forecast_horizon"]

        n_samples = (
            len(y_series) - self.model_config["lookback_window"] - forecast_horizon + 1
        )

        print("len y_series:", len(y_series))
        print("len timestamps:", len(timestamps))
        print("forecast_horizon:", forecast_horizon)

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {self.model_config['lookback_window'] + forecast_horizon} samples."
            )

        features = []
        targets = []

        for i in range(n_samples):
            # Create lag features
            lag_features = y_series[i : i + self.model_config["lookback_window"]]

            # Create rolling statistics
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)
            rolling_min = np.min(lag_features)
            rolling_max = np.max(lag_features)

            # Create trend features
            trend = np.polyfit(
                range(self.model_config["lookback_window"]), lag_features, 1
            )[0]

            # Combine all features
            sample_features = list(lag_features) + [
                rolling_mean,
                rolling_std,
                rolling_min,
                rolling_max,
                trend,
            ]

            # Add timestamp features if available
            if (
                timestamps is not None
                and len(timestamps)
                >= i + self.model_config["lookback_window"] + forecast_horizon
            ):
                # Add current timestamp and future timestamps as features
                current_timestamp = timestamps[
                    i + self.model_config["lookback_window"] - 1
                ]
                future_timestamps = timestamps[
                    i
                    + self.model_config["lookback_window"] : i
                    + self.model_config["lookback_window"]
                    + forecast_horizon
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
            # Multi-output: target is a vector of length forecast_horizon
            targets.append(
                [
                    y_series[i + self.model_config["lookback_window"] + step]
                    for step in range(forecast_horizon)
                ]
            )

        return np.array(features), np.array(targets)

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
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
            y_start_date: The start date timestamp for y_context and y_target in string form
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)

        Returns:
            self: The fitted model instance.
        """
        if not self.is_fitted is None:
            self._build_model()

        # Combine context and target data for training
        forecast_horizon = timestamps_target.shape[0]

        full_y_data = np.concatenate([y_context, y_target], axis=0)
        full_y_data = np.squeeze(full_y_data)
        # Combine timestamps if available

        full_timestamps = np.concatenate(
            [timestamps_context, timestamps_target], axis=0
        )
        full_timestamps = np.squeeze(full_timestamps)

        print("Creating features")
        # Create features and targets (no exogenous variables)
        X, y = self._create_features(full_y_data, full_timestamps)
        print("Started training random forest")
        # y is shape (n_samples, forecast_horizon)
        self.model.fit(X, y)
        print("Ended training random forest")
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
        Make predictions using the trained Random Forest model.

        TECHNIQUE: Single Model with Timestamp Features
        - Uses last lookback_window values to create features
        - Incorporates timestamp features for time-aware prediction
        - Predicts all forecast horizon steps with a single model

        Args:
            y_context: Past target values (time series) - used for prediction
            y_target: Future target values - used to determine prediction length
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            freq: Frequency string (ignored for random_forest, kept for compatibility)

        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        forecast_horizon = self.model_config["forecast_horizon"]

        y_context = np.squeeze(y_context)

        # Combine timestamps if available

        full_timestamps = np.concatenate(
            [timestamps_context, timestamps_target], axis=0
        )
        full_timestamps = np.squeeze(full_timestamps)

        # Use the last available context to create a single feature row
        # For prediction, we want to predict forecast_horizon ahead
        # So we need to create a feature row for the current context and the next forecast_horizon timestamps

        # Use the last lookback_window context and the actual target timestamps
        context_timestamps = timestamps_context[-self.model_config["lookback_window"] :]
        feature_row, _ = self._create_features(
            np.concatenate([y_context, np.zeros(forecast_horizon)]), full_timestamps
        )

        X_last = feature_row[-1:].reshape(1, -1)

        print("X_last shape:", X_last.shape)

        # Predict all steps at once
        preds = self.model.predict(X_last)
        return preds

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate full-length predictions for the test set using a rolling window approach.
        This method repeatedly calls the model's predict method, each time advancing the context by forecast_horizon steps,
        until the total number of predictions matches the length of y_target.

        Args:
            y_context: Initial context (typically last lookback_window values from train set)
            y_target: The full test set (used to determine total prediction length)
            x_context, x_target: Exogenous variables (optional)
            timestamps_context, timestamps_target: Timestamps (optional)
            freq: Frequency string (ignored for random_forest, kept for compatibility)
            **kwargs: Additional keyword arguments
        Returns:
            np.ndarray: Full-length predictions matching the length of y_target
        """

        # Initialize context with the last lookback_window values from y_context
        # Ensure context is 1D for consistent handling

        y_context = y_context[-self.model_config["lookback_window"] :, 0]
        timestamps_context = timestamps_context[
            -self.model_config["lookback_window"] :, 0
        ]
        timestamps_target = timestamps_target[:, 0]

        preds = []
        steps = self.model_config["forecast_horizon"]
        steps_remaining = len(timestamps_target)

        while steps_remaining > 0:
            # Determine how many steps to predict in this iteration

            # Create a dummy target array for this prediction step
            # The actual target values don't matter for prediction, we just need the right shape
            target_for_prediction = np.zeros(steps)

            # Get the current target timestamps for this prediction window

            current_target_start = len(preds)
            current_target_end = current_target_start + steps
            current_target_timestamps = timestamps_target[
                current_target_start:current_target_end
            ]

            # Make prediction for this window
            pred = self._predict(
                y_context=y_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs,
            )

            # Extract the predictions for this window and ensure they're 1D
            pred = pred.flatten()[:steps]
            preds.extend(pred)

            # Update context by removing oldest values and adding new predictions
            # This maintains a fixed-size context equal to lookback_window
            # Ensure pred is 1D before concatenating
            y_context = np.concatenate([y_context, pred])[
                -self.model_config["lookback_window"] :
            ]

            # Update context timestamps if available
            timestamps_context = np.concatenate(
                [timestamps_context, timestamps_target]
            )[-self.model_config["lookback_window"] :]

            # Update remaining steps
            steps_remaining -= steps

        preds = np.array(preds)
        preds = np.expand_dims(preds, axis=1)

        return preds
