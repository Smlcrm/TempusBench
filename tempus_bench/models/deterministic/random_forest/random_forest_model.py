"""
Random Forest model implementation for time series forecasting
"""
import numpy as np
import pandas as pd

from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import BaseModel as PydanticBaseModel, Field
from sklearn.ensemble import RandomForestRegressor

from ...base_model import BaseModel, validate_inputs


class RandomForestHyperparams(PydanticBaseModel):
    n_estimators: int = Field(default=100, ge=1, description="Number of trees in the forest")
    max_depth: int = Field(default=None, ge=1, description="Maximum depth of trees")
    min_samples_split: Optional[int] = Field(default=2, ge=2, description="Minimum samples to split a node")
    min_samples_leaf: Optional[int] = Field(default=1, ge=1, description="Minimum samples in a leaf")
    max_features: Optional[Literal["sqrt", "log2", "auto"]] = Field(default="sqrt", description="Number of features to consider for splits")


class RandomForestModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Random Forest model with model-specific parameters.
        """
        super().__init__(params, settings, RandomForestHyperparams)
        self._build_model()

    def _build_model(self):
        """
        Build the RandomForestRegressor model instance from the configuration.
        """

        self.logger.debug("RandomForestModel._build_model", "building model")
        # Get hyperparameters from params, excluding non-estimator parameters
        model_params = {}
        for key, value in self.dict().items():
            if key not in ["lookback_window"]:
                model_params[key] = value

        # Filter out keys that are not valid for RandomForestRegressor
        valid_keys = set(RandomForestRegressor().get_params().keys())
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}

        if "random_state" not in filtered_params:
            filtered_params["random_state"] = self.settings["random_state"]

        if "n_jobs" not in filtered_params:
            filtered_params["n_jobs"] = self.settings["n_jobs"]

        self._model = RandomForestRegressor(**filtered_params)
        self.is_fitted = False

    def _create_features(
        self, y_series: np.ndarray, timestamps: np.ndarray, **kwargs: dict
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
        # Handle multivariate series for feature generation
        if y_series.ndim == 1:
            y_series = y_series.reshape(-1, 1)
        num_targets = y_series.shape[1]
        if timestamps is not None:
            timestamps = np.squeeze(timestamps)

        forecast_horizon = kwargs["forecast_horizon"]

        n_samples = (
            len(y_series) - self.settings["lookback_window"] - forecast_horizon + 1
        )

        self.logger.debug("RandomForestModel._create_features", f"len y_series: {len(y_series)}")
        self.logger.debug("RandomForestModel._create_features", f"len timestamps: {len(timestamps)}")
        self.logger.debug("RandomForestModel._create_features", f"forecast_horizon: {forecast_horizon}")

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data. Need at least {self.settings["lookback_window"] + forecast_horizon} samples."
            )

        features = []
        targets = []

        for i in range(n_samples):
            # Create lag features for all targets
            lag_features = y_series[i : i + self.settings["lookback_window"]].flatten()

            # Create rolling statistics for each target
            sample_features = []
            sample_features.extend(lag_features)

            for target_idx in range(num_targets):
                target_data = y_series[i : i + self.settings["lookback_window"], target_idx]

                rolling_mean = np.mean(target_data)
                rolling_std = np.std(target_data)
                rolling_min = np.min(target_data)
                rolling_max = np.max(target_data)

                # Create trend features
                trend = np.polyfit(range(len(target_data)), target_data, 1)[0]

                # Create volatility features
                rolling_range = rolling_max - rolling_min
                rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)

                # Add features for this target
                sample_features.extend([
                    rolling_mean,
                    rolling_std,
                    rolling_min,
                    rolling_max,
                    trend,
                    rolling_range,
                    rolling_iqr,
                ])

            # Add timestamp features if available
            if (
                timestamps is not None
                and len(timestamps)
                >= i + self.settings["lookback_window"] + forecast_horizon
            ):
                # Add current timestamp and future timestamps as features
                current_timestamp = timestamps[
                    i + self.settings["lookback_window"] - 1
                ]
                future_timestamps = timestamps[
                    i
                    + self.settings["lookback_window"] : i
                    + self.settings["lookback_window"]
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
            # Multi-output: target is a vector of length forecast_horizon * num_targets
            target_values = []
            for step in range(forecast_horizon):
                for target_idx in range(num_targets):
                    target_values.append(
                        y_series[i + self.settings["lookback_window"] + step, target_idx]
                    )
            targets.append(target_values)

        return np.array(features), np.array(targets)

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
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
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        lookback_window = self.settings["lookback_window"]
        
        if not self.is_fitted is None:
            self._build_model()

        # Combine context and target data for training
        # Train the model to predict the configured forecast horizon
        forecast_horizon = int(len(timestamps_target)) if timestamps_target is not None else int(forecast_horizon)

        # Concatenate along time axis (axis=0) for our (num_steps, num_targets) format
        full_y_data = np.concatenate([y_context, y_target], axis=0)

        # Combine timestamps if available
        full_timestamps = np.concatenate(
            [timestamps_context, timestamps_target], axis=0
        )
        full_timestamps = np.squeeze(full_timestamps)

        self.logger.debug("RandomForestModel._train", "Creating features")
        # Create features and targets (no exogenous variables)
        X, y = self._create_features(full_y_data, full_timestamps, **kwargs)
        self.logger.info("RandomForestModel._train", "Started training random forest")
        # y is shape (n_samples, forecast_horizon)
        self._model.fit(X, y)
        self.logger.info("RandomForestModel._train", "Ended training random forest")
        self.is_fitted = True
        return self

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "RandomForestModel":
        return self._train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            **kwargs,
        )

    @validate_inputs
    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
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
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        lookback_window = self.settings["lookback_window"]
        
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        forecast_horizon = int(len(timestamps_target)) if timestamps_target is not None else int(forecast_horizon)

        # Ensure y_context is in (num_steps, num_targets) format
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        # Combine timestamps if available
        full_timestamps = np.concatenate(
            [timestamps_context, timestamps_target], axis=0
        )
        full_timestamps = np.squeeze(full_timestamps)

        # Use the last available context to create a single feature row
        # For prediction, we want to predict forecast_horizon ahead
        # So we need to create a feature row for the current context and the next forecast_horizon timestamps

        # Create dummy future data for feature creation
        num_targets = y_context.shape[1]
        dummy_future = np.zeros((forecast_horizon, num_targets))
        full_y_data = np.concatenate([y_context, dummy_future], axis=0)

        feature_row, _ = self._create_features(full_y_data, full_timestamps, **kwargs)

        X_last = feature_row[-1:].reshape(1, -1)

        self.logger.debug("RandomForestModel._predict", f"X_last shape: {X_last.shape}")

        # Predict all steps at once
        preds = self._model.predict(X_last)

        # Reshape predictions back to (forecast_horizon, num_targets)
        # The model predicts forecast_horizon * num_targets values
        preds_reshaped = preds.reshape(forecast_horizon, num_targets)

        # Inverse transform predictions if scaler is available
        if self._scaler is not None:
            preds_reshaped = self._scaler.inverse_transform(preds_reshaped)

        return preds_reshaped

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Generate full-length predictions for the test set using a rolling window approach.
        This method repeatedly calls the model's predict method, each time advancing the context by forecast_horizon steps,
        until the total number of predictions matches the length of y_target.

        Args:
            y_context: Initial context (typically last lookback_window values from train set)
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments
        Returns:
            np.ndarray: Full-length predictions matching the length of y_target
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        lookback_window = self.settings["lookback_window"]

        # Initialize context with the last lookback_window values from y_context
        # Ensure context is 1D for consistent handling

        # Ensure y_context is in (num_steps, num_targets) format
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        num_targets = y_context.shape[1]
        forecast_horizon = len(timestamps_target)

        # Use the last lookback_window timesteps as context
        y_context = y_context[-lookback_window:, :]
        timestamps_context = timestamps_context[-lookback_window:]

        # Make prediction for the full forecast horizon
        pred = self._predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            **kwargs,
        )  # pred shape (num_series, forecast_horizon)

        return pred
