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
    context_length: Optional[int] = Field(
        default=2048, ge=1, description="Context window size"
    )
    num_samples: Optional[int] = Field(
        default=10, ge=1, description="Number of probabilistic samples"
    )
    batch_size: Optional[int] = Field(default=1, ge=1, description="Batch size")


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

        # Model-specific attributes
        self._model = None

    def _create_predictor_for_horizon(self, forecast_horizon: int):
        """Create a predictor for a specific forecast horizon."""

        # Reference params, settings, device, python_version
        context_length = self.context_length
        batch_size = self.batch_size
        num_samples = self.num_samples

        # Create the estimator with the specified horizon
        estimator = LagLlamaEstimator(
            prediction_length=forecast_horizon,
            context_length=context_length,
            batch_size=batch_size,
            num_parallel_samples=num_samples,
            device=torch.device(self.device),
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
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "LagllamaModel":
        """
        Train/fine-tune the Lag-Llama model on given data.
        Lag-Llama is pre-trained, so this method just validates inputs and sets fitted status.

        Args:
            y_context: Past target values
            y_target: Future target values (not used for pre-trained model)
            y_start_date: Start date timestamp (not used for pre-trained model)

        Returns:
            self: The fitted model instance
        """

        # Lag-Llama is pre-trained, so we just mark as fitted
        self.is_fitted = True
        return self

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
        num_samples = kwargs["num_samples"]

        # Reference params, settings, device, python_version
        context_length = self.context_length
        batch_size = self.batch_size

        forecast_horizon = timestamps_target.shape[0]
        # Create predictor for this horizon
        predictor = self._create_predictor_for_horizon(forecast_horizon)

        # Convert input to DataFrame format
        start_time = self.convert_to_datetimeindex(timestamps_context)[0]
        periods = y_context.shape[0]  # Use num_steps, not num_targets
        timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Create series DataFrame
        # Ensure y_context is 1D for single series prediction
        if y_context.ndim > 1:
            y_context_1d = (
                y_context[:, 0] if y_context.shape[1] > 0 else y_context.flatten()
            )
        else:
            y_context_1d = y_context

        context_df = pd.DataFrame(
            {
                "ds": timestamps,
                "target": y_context_1d,
                "unique_id": "test_series",
            }
        )

        context_df["target"] = context_df["target"].astype("float32")

        # Create GluonTS dataset
        context_df = PandasDataset.from_long_dataframe(
            context_df, target="target", timestamp="ds", item_id="unique_id", freq=freq
        )

        # Generate forecasts
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=context_df,
            predictor=predictor,
            num_samples=self.num_samples,
        )

        forecasts = list(forecast_it)

        # Process results to get samples
        samples_list = []
        for forecast in forecasts:
            # Get samples from forecast object
            if hasattr(forecast, "samples"):
                samples = forecast.samples  # Shape: (num_samples, forecast_horizon)
            else:
                # Fallback: generate samples from mean and std if available
                mean = forecast.mean
                std = getattr(forecast, "std", None)
                if std is not None:
                    samples = np.random.normal(mean, std, (self.num_samples, len(mean)))
                else:
                    # If no std, create samples by adding small noise to mean
                    samples = np.tile(mean, (self.num_samples, 1))
                    samples += np.random.normal(0, 0.01, samples.shape)

            samples_list.append(samples)

        # Combine samples from all forecasts
        if samples_list:
            # For single series, samples_list has one element
            samples = samples_list[0]  # Shape: (num_samples, forecast_horizon)

            # Ensure correct shape: (num_samples, forecast_horizon, num_targets)
            if samples.ndim == 2:
                samples = samples[:, :, np.newaxis]  # Add target dimension

            return samples
        else:
            # Fallback: return zeros with correct shape
            return np.zeros((num_samples, forecast_horizon, 1))

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
        if y_context.ndim > 1 and y_context.shape[1] > 1:
            # Treat each feature (column) as an independent series
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
            return np.concatenate(preds, axis=1)
        else:
            return self._predict(
                y_context=y_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                num_samples=num_samples,
            )

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Generate quantile forecasts.

        Args:
            df: Historical time series data
            forecast_horizon: Number of future steps to predict
            quantile_levels: List of quantile levels to compute

        Returns:
            Nested dict with series names and quantile forecasts
        """
        sample_results = self._predict_internal(
            df, forecast_horizon, return_samples=True
        )

        quantile_results = {}
        for series_name, forecasts in sample_results.items():
            if "samples" in forecasts:
                samples = np.array(forecasts["samples"])
                quantiles = {}

                for q in quantile_levels:
                    quantiles[f"q{int(q*100)}"] = np.percentile(
                        samples, q * 100, axis=0
                    ).tolist()

                quantile_results[series_name] = quantiles
            else:
                quantile_results[series_name] = {
                    f"q{int(q*100)}": forecasts.get("mean", [0.0] * forecast_horizon)
                    for q in quantile_levels
                }

        return quantile_results
