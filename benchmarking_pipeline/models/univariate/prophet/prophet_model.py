"""
Prophet model implementation.
TODO-COULD work multivariate
"""

import os
import json
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from benchmarking_pipeline.models.base_model import BaseModel


class ProphetModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Prophet model with a given configuration.

        Args:
            config: Configuration dictionary for Prophet parameters.
                    e.g., {'model_params': {'seasonality_mode': 'multiplicative'}}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config)
        self._build_model()

    def _build_model(self):
        self.model = Prophet(**self.model_config)
        self.is_fitted = False

    @staticmethod
    def ensure_series_with_datetimeindex(y, start_date, freq):
        """
        Ensure the input series has a proper datetime index.

        Args:
            y: Input series (can be numpy array, pandas series, or already indexed)
            start_date: Start date to use for the index
            freq: Frequency from CSV data - MUST be provided

        Returns:
            pd.Series: Series with proper datetime index

        Raises:
            ValueError: If freq is None or empty
        """
        if freq is None or freq == "":
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )

        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            return y
        return pd.Series(
            y.values if hasattr(y, "values") else y,
            index=pd.date_range(start_date, periods=len(y), freq=freq),
        )

    def convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                # Convert from nanoseconds to datetime
                if timestamps[0] > 1e18:  # Likely nanoseconds
                    timestamps = pd.to_datetime(timestamps, unit="ns")
                elif timestamps[0] > 1e15:  # Likely microseconds
                    timestamps = pd.to_datetime(timestamps, unit="us")
                elif timestamps[0] > 1e12:  # Likely milliseconds
                    timestamps = pd.to_datetime(timestamps, unit="ms")
                else:  # Likely seconds
                    timestamps = pd.to_datetime(timestamps, unit="s")
            else:
                timestamps = pd.to_datetime(timestamps)
        else:
            timestamps = timestamps

        return timestamps

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ):

        if not self.is_fitted:
            self.model = Prophet(**self.model_config)
        # Use the provided timestamps to create a DatetimeIndex for y_context
        # Ensure 1D series indexed by provided timestamps
        y_context = y_context.squeeze()
        y_target = y_target.squeeze()

        timestamps_context = self.convert_to_datetimeindex(timestamps_context)
        timestamps_target = self.convert_to_datetimeindex(timestamps_target)

        train_df = pd.DataFrame({"ds": timestamps_context, "y": y_context})

        self.model.fit(train_df)
        # Store training statistics for fallback predictions
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained Prophet model.

        Args:
            y_context: Recent/past target values
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            y_context_timestamps: Timestamps for context data
            y_target: Target values for evaluation (optional)
            y_target_timestamps: Timestamps for target data (optional)
            freq: Frequency string from CSV data - MUST be provided

        Returns:
            np.ndarray: Model predictions

        Raises:
            ValueError: If freq is None or if required data is missing
        """

        # Default forecast horizon if neither y_target nor y_target_timestamps provided

        # Create future dataframe with the correct timestamps
        future_df = pd.DataFrame(
            {"ds": self.convert_to_datetimeindex(timestamps_target)}
        )

        # Make predictions
        forecast = self.model.predict(future_df)

        forecast = np.asarray(forecast["yhat"])

        if len(forecast.shape) == 1:
            forecast = np.expand_dims(forecast, axis=1)
        # Return the predicted values

        return forecast
