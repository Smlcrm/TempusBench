import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import warnings
import os
import subprocess
import sys
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from tempus_bench.models.base_model import BaseModel

# Add the lagllama directory to the Python path for absolute imports
lagllama_dir = os.path.dirname(os.path.abspath(__file__))
if lagllama_dir not in sys.path:
    sys.path.insert(0, lagllama_dir)

from lag_llama.gluon.estimator import LagLlamaEstimator

# Try to import lag_llama, install if not available


class LagllamaModel(BaseModel):
    """
    Lag-Llama model implementation that inherits from BaseModel.
    Works seamlessly like TimesFM with automatic setup.
    """

    def __init__(self, config: Dict[str, Any], logs_dir: str):
        """
        Initialize Lag-Llama model with BaseModel interface.

        Args:
            config: Configuration dictionary containing:
                - checkpoint_path: str, path to checkpoint (default: "lag-llama.ckpt")
                - context_length: int, context window size (default: 128)
                - prediction_length: int, number of time series elements to predict (30)
                - num_samples: int, number of probabilistic samples (default: 5)
                - device: str, device to use (default: "auto")
            logs_dir: Directory for storing log files (optional)
        """

        # Initialize base model
        super().__init__(config, logs_dir)
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model-specific attributes
        self.model_config["context_length"] = 32
        self.model_config["num_samples"] = 10
        self.model_config["batch_size"] = 1
        self.model_config["batch_size"] = 1

        self.model = None

        print(
            f"🦙 Lag-Llama initialized - Device: {self.device}, Context: {self.model_config['context_length']}"
        )

    def _create_predictor_for_horizon(self, forecast_horizon: int):
        """Create a predictor for a specific forecast horizon."""

        # Create the estimator with the specified horizon
        estimator = LagLlamaEstimator(
            prediction_length=forecast_horizon,
            context_length=self.model_config["context_length"],
            batch_size=self.model_config["batch_size"],
            num_parallel_samples=self.model_config["num_samples"],
            device=self.device,
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
                    if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(max_ts, NS_LOWER, NS_UPPER):
                        unit = "ns"
                    # Try microseconds
                    elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(max_ts, US_LOWER, US_UPPER):
                        unit = "us"
                    # Try milliseconds
                    elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(max_ts, MS_LOWER, MS_UPPER):
                        unit = "ms"
                    # Try seconds
                    elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(max_ts, S_LOWER, S_UPPER):
                        unit = "s"
                    else:
                        raise ValueError(
                            f"Timestamps are out of bounds for pandas datetime64[ns] (min={min_ts}, max={max_ts})."
                        )
                    timestamps = pd.to_datetime(timestamps, unit=unit)
                else:
                    timestamps = pd.to_datetime(timestamps)

        return timestamps

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
        print("✅ Lag-Llama ready (pre-trained)")

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
        Make predictions using the trained Lag-Llama model.

        Args:
            y_context: Recent/past target values
            y_target: Future target values (used to determine forecast horizon if not provided)
            y_context_timestamps: Timestamps for context data (not used)
            y_target_timestamps: Timestamps for target data (not used)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            **kwargs: Additional arguments (ignored)

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon,)
        """

        forecast_horizon = timestamps_target.shape[0]
        # Create predictor for this horizon
        predictor = self._create_predictor_for_horizon(forecast_horizon)

        # Convert input to DataFrame format
        # df = pd.DataFrame(y_context)

        # Use the internal prediction method
        # results = self._predict_internal(df, forecast_horizon)
        start_time = self.convert_to_datetimeindex(timestamps_context)[0]
        periods = y_context.shape[0]  # Use num_steps, not num_targets
        timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Create series DataFrame
        # Ensure y_context is 1D for single series prediction
        if y_context.ndim > 1:
            y_context_1d = y_context[:, 0] if y_context.shape[1] > 0 else y_context.flatten()
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
            num_samples=self.model_config["num_samples"],
        )

        forecasts = list(forecast_it)

        # Process results
        results = {}
        for forecast in forecasts:
            # series_name = getattr(forecast, "item_id", "unknown")

            # if return_samples:
            #     results = {
            #         "mean": forecast.mean.tolist(),
            #         "median": forecast.quantile(0.5).tolist(),
            #         "q10": forecast.quantile(0.1).tolist(),
            #         "q90": forecast.quantile(0.9).tolist(),
            #         "samples": forecast.samples.tolist(),
            #     }
            # else:
            results = forecast.mean.tolist()
            results = np.asarray(results)
            if len(results.shape) == 1:
                results = np.expand_dims(results, axis=1)

        return results

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "LagllamaModel":
        """
        Anyvariate wrapper: for multivariate, no separate fitting is needed; we keep separate handles.
        """
        if y_context.ndim > 1 and y_context.shape[1] > 1:
            # Treat each feature (column) as an independent series
            self.models = []
            num_targets = y_context.shape[1]
            for k in range(num_targets):
                m = LagllamaModel(self.config, logs_dir=self.logs_dir)
                yc = y_context[:, k]
                yt = y_target[:, k] if (y_target is not None and y_target.ndim > 1 and y_target.shape[1] > k) else y_target
                m._train(y_context=yc, y_target=yt,
                         timestamps_context=timestamps_context, timestamps_target=timestamps_target, freq=freq)
                self.models.append(m)
            self.is_fitted = True
            return self
        return self._train(y_context, y_target, timestamps_context, timestamps_target, freq)

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        if hasattr(self, "models") and self.models:
            preds = []
            for k, m in enumerate(self.models):
                yc = y_context[:, k] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(y_context=yc, timestamps_context=timestamps_context,
                                timestamps_target=timestamps_target, freq=freq, **kwargs)
                preds.append(pk.reshape(-1, 1) if pk.ndim == 1 else pk)
            return np.concatenate(preds, axis=1)
        return self._predict(y_context, timestamps_context, timestamps_target, freq, **kwargs)

    # def _predict_internal(
    #     self,
    #     df: pd.DataFrame,
    #     prediction_length: int,
    #     freq: str,
    #     return_samples: bool = False,
    # ) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
    #     """Internal prediction method - similar to standalone forecaster"""

    #     # Use existing predictor or create new one if needed

    #     predictor = self._create_predictor_for_horizon(prediction_length)

    #     # Create timestamps
    #     end_date = datetime.now()
    #     start_date = end_date - timedelta(days=len(series_data) - 1)
    #     timestamps = pd.date_range(
    #         start=start_date, periods=len(series_data), freq=freq
    #     )

    #     # Create series DataFrame
    #     series_df = pd.DataFrame(
    #         {
    #             "ds": timestamps,
    #             "target": series_data.values,
    #             "unique_id": series_name,
    #         }
    #     )

    #     all_series_data.append(series_df)
    #     series_names.append(series_name)

    #     if not all_series_data:
    #         return {}

    #     # Combine all series
    #     combined_df = pd.concat(all_series_data, ignore_index=True)

    #     # Ensure target column is float32 to match model dtype
    #     combined_df["target"] = combined_df["target"].astype(np.float32)

    #     # Create GluonTS dataset
    #     dataset = PandasDataset.from_long_dataframe(
    #         combined_df, target="target", item_id="unique_id", timestamp="ds", freq=freq
    #     )

    #     # Generate forecasts
    #     forecast_it, ts_it = make_evaluation_predictions(
    #         dataset=dataset,
    #         predictor=predictor,
    #         num_samples=self.model_config["num_samples"],
    #     )

    #     forecasts = list(forecast_it)

    #     # Process results
    #     results = {}
    #     for forecast in forecasts:
    #         series_name = getattr(forecast, "item_id", "unknown")

    #         if return_samples:
    #             results[series_name] = {
    #                 "mean": forecast.mean.tolist(),
    #                 "median": forecast.quantile(0.5).tolist(),
    #                 "q10": forecast.quantile(0.1).tolist(),
    #                 "q90": forecast.quantile(0.9).tolist(),
    #                 "samples": forecast.samples.tolist(),
    #             }
    #         else:
    #             results[series_name] = forecast.mean.tolist()

    #     return results

    # TimesFM-style convenience methods
    def predict_df(
        self, df: pd.DataFrame, forecast_horizon: int, return_samples: bool = False
    ) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        """
        TimesFM-style prediction on DataFrame.

        Args:
            df: DataFrame with time series columns
            forecast_horizon: Number of steps to forecast
            return_samples: Whether to return probabilistic samples

        Returns:
            Dictionary with forecasts for each series
        """
        return self._predict_internal(
            df, forecast_horizon, return_samples=return_samples
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


# Convenience wrapper for standalone usage (like TimesFM)
class LagLlamaForecaster:
    """
    Standalone forecaster wrapper for easy usage (mirrors TimesFM interface)
    """

    def __init__(self, checkpoint_path: str = "lag-llama.ckpt", logs_dir: str = None, **kwargs):
        """Initialize with TimesFM-like interface"""
        config = {"checkpoint_path": checkpoint_path}
        config.update(kwargs)
        self.model = LagllamaModel(config, logs_dir=logs_dir)

    def predict(self, df: pd.DataFrame, forecast_horizon: int, **kwargs):
        """TimesFM-style predict method"""
        return self.model.predict_df(df, forecast_horizon, **kwargs)

    def predict_quantiles(self, df: pd.DataFrame, forecast_horizon: int, **kwargs):
        """TimesFM-style quantile prediction"""
        return self.model.predict_quantiles(df, forecast_horizon, **kwargs)
