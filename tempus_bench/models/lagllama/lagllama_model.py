import os
import subprocess
import sys
import warnings

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from pydantic import BaseModel as PydanticBaseModel, Field
import torch
from tempus_bench.models.base_model import BaseModel, validate_inputs

# Add the lagllama directory to the Python path for absolute imports
lagllama_dir = os.path.dirname(os.path.abspath(__file__))
if lagllama_dir not in sys.path:
    sys.path.insert(0, lagllama_dir)

from lag_llama.gluon.estimator import LagLlamaEstimator


class LagllamaHyperparams(PydanticBaseModel):
    pass


# Try to import lag_llama, install if not available
class LagllamaModel(BaseModel):
    """
    Lag-Llama model implementation that inherits from BaseModel.
    Works seamlessly like TimesFM with automatic setup.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Lag-Llama model with BaseModel interface.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        # Initialize base model
        super().__init__(params, settings, LagllamaHyperparams)

    def _create_predictor_for_horizon(
        self, forecast_horizon: int, input_size: int = 1, num_samples: int = 10
    ):
        """Create a predictor for a specific forecast horizon.

        Args:
            forecast_horizon: Number of future time steps to predict
            input_size: Number of features (for multivariate). Default is 1 for univariate.

        Returns:
            Predictor instance
        """

        # Reference params, settings, device, python_version
        context_length = self.context_length
        batch_size = self.batch_size

        # Create the estimator with the specified horizon and input_size
        # Use lags_seq that avoids deprecated frequencies that cause issues with GluonTS
        # GluonTS's get_lags_for_frequency has compatibility issues with pandas new format
        # Only W (weekly) and D (daily) work reliably with current GluonTS version
        # Using minimal lags_seq to avoid errors - model will still work but with fewer lag features
        compatible_lags_seq = ["W", "D"]  # Only frequencies that work with current GluonTS
        
        estimator = LagLlamaEstimator(
            prediction_length=forecast_horizon,
            context_length=context_length,
            input_size=input_size,
            batch_size=batch_size,
            num_parallel_samples=num_samples,
            device=torch.device(self.device),
            lags_seq=compatible_lags_seq,  # Use compatible frequencies
        )

        # Create predictor from estimator
        transformation = estimator.create_transformation()
        lightning_module = estimator.create_lightning_module()
        predictor = estimator.create_predictor(transformation, lightning_module)

        return predictor

    def convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                min_ts = np.min(timestamps)
                max_ts = np.max(timestamps)

                # Pandas datetime bounds for 64-bit ns: 1677-09-21 to 2262-04-11
                # 1677-09-21T00:12:43.145224192Z = -9223372036854775808 ns
                # 2262-04-11T23:47:16.854775807Z = 9223372036854775807 ns
                NS_LOWER = -9223372036854775808
                NS_UPPER = 9223372036854775807
                US_LOWER = NS_LOWER // 1000
                US_UPPER = NS_UPPER // 1000
                MS_LOWER = NS_LOWER // 1_000_000
                MS_UPPER = NS_UPPER // 1_000_000
                S_LOWER = NS_LOWER // 1_000_000_000
                S_UPPER = NS_UPPER // 1_000_000_000

                def in_bounds(val, lower, upper):
                    return lower <= val <= upper

                # Try to classify the likely unit and check bounds
                unit = None
                if isinstance(min_ts, (int, np.integer)):
                    # Try nanoseconds
                    if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(
                        max_ts, NS_LOWER, NS_UPPER
                    ):
                        unit = "ns"
                    # Try microseconds
                    elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(
                        max_ts, US_LOWER, US_UPPER
                    ):
                        unit = "us"
                    # Try milliseconds
                    elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(
                        max_ts, MS_LOWER, MS_UPPER
                    ):
                        unit = "ms"
                    # Try seconds
                    elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(
                        max_ts, S_LOWER, S_UPPER
                    ):
                        unit = "s"
                    else:
                        raise ValueError(
                            f"Timestamps are out of bounds for pandas datetime64[ns] (min={min_ts}, max={max_ts})."
                        )
                    timestamps = pd.to_datetime(timestamps, unit=unit)
                else:
                    timestamps = pd.to_datetime(timestamps)

        return timestamps


    @validate_inputs
    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Lag-Llama model.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model prediction samples with shape (num_samples, forecast_horizon, num_targets)

        Raises:
            ValueError: If freq is None or empty - frequency must always be read from CSV data
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Convert pandas new frequency format to old format for GluonTS compatibility
        # GluonTS expects old format (Q, M, A) but preprocessor uses new format (QE, ME, YE)
        gluonts_freq_mapping = {
            "QE": "Q",  # Quarter End -> Quarterly
            "ME": "M",  # Month End -> Monthly
            "YE": "A",  # Year End -> Annual
        }
        freq_for_gluonts = gluonts_freq_mapping.get(freq, freq)  # Use original if not in mapping
        num_samples = kwargs["num_samples"]

        # Reference params, settings, device, python_version
        context_length = self.context_length
        batch_size = self.batch_size

        forecast_horizon = timestamps_target.shape[0]

        # _predict only handles univariate time series
        # Ensure y_context is univariate (1D or 2D with single feature)
        if y_context.ndim == 1:
            # Univariate: ensure 2D shape (num_steps, 1)
            y_context = y_context[:, np.newaxis]
        elif y_context.ndim == 2:
            # Check if it's truly univariate
            if y_context.shape[1] != 1:
                raise ValueError(
                    f"_predict only handles univariate data. Got y_context with shape {y_context.shape}. "
                    "Use predict() for multivariate data."
                )
        else:
            raise ValueError(f"y_context must be 1D or 2D, got shape {y_context.shape}")

        # Convert input to DataFrame format
        start_time = self.convert_to_datetimeindex(timestamps_context)[0]
        periods = y_context.shape[0]  # Use num_steps
        timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Create GluonTS dataset (always univariate)
        context_df = pd.DataFrame(
            {
                "ds": timestamps,
                "target": y_context[:, 0],
                "unique_id": "test_series",
            }
        )
        context_df["target"] = context_df["target"].astype("float32")
        dataset = PandasDataset.from_long_dataframe(
            context_df,
            target="target",
            timestamp="ds",
            item_id="unique_id",
            freq=freq_for_gluonts,  # Use GluonTS-compatible frequency
        )

        # Generate forecasts
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=self.predictor,
            num_samples=kwargs["num_samples"],
        )

        forecasts = list(forecast_it)

        # Process results to get samples
        forecast = forecasts[0]  # Single forecast for univariate

        samples = np.asarray(forecast.samples)  # Shape: (num_samples, forecast_horizon)

        samples = np.expand_dims(samples, axis=2)

        return samples

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "LagllamaModel":
        """
        Anyvariate wrapper: for multivariate, no separate fitting is needed; we keep separate handles.
            """
        if not hasattr(self, "predictor"):
            self.predictor = self._create_predictor_for_horizon(
                y_target.shape[0], input_size=1, num_samples=kwargs["num_samples"]
            )
            # Lag-Llama is pre-trained, so we just mark as fitted
            self.is_fitted = True

        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Lag-Llama model.

        Args:
            y_context: Past target values (time series data)
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (num_samples, forecast_horizon, num_targets)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        num_samples = kwargs["num_samples"]

        # Handle multivariate by iterating over features
        if y_context.ndim > 1 and y_context.shape[1] > 1:
            # Multivariate: treat each feature (column) as an independent univariate series
            preds = []
            num_targets = y_context.shape[1]
            for k in range(num_targets):
                pred = self._predict(
                    y_context=y_context[:, k : k + 1],
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                    num_samples=num_samples,
                )
                preds.append(pred)
            # Concatenate along the feature dimension
            # Each pred has shape (num_samples, forecast_horizon, 1)
            # Result should be (num_samples, forecast_horizon, num_targets)
            return np.concatenate(preds, axis=2)
        else:
            # Univariate: pass directly to _predict
            return self._predict(
                y_context=y_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                num_samples=num_samples,
            )
