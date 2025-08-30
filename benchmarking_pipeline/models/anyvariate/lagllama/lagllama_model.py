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
from benchmarking_pipeline.models.foundation_model import FoundationModel

from .lag_llama.gluon.estimator import LagLlamaEstimator

# Try to import lag_llama, install if not available


class LagllamaModel(FoundationModel):
    """
    Lag-Llama model implementation that inherits from BaseModel.
    Works seamlessly like TimesFM with automatic setup.
    """

    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Lag-Llama model with BaseModel interface.

        Args:
            config: Configuration dictionary containing:
                - checkpoint_path: str, path to checkpoint (default: "lag-llama.ckpt")
                - context_length: int, context window size (default: 128)
                - prediction_length: int, number of time series elements to predict (30)
                - num_samples: int, number of probabilistic samples (default: 5)
                - device: str, device to use (default: "auto")
            config_file: Path to JSON config file
        """

        # Initialize base model
        super().__init__(config, config_file)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model-specific attributes
        self.context_length = self.config.get("context_length", 32)
        self.num_samples = self.config.get("num_samples", 5)
        self.batch_size = 1

        # forecast_horizon is inherited from parent class (FoundationModel)
        self.model = None
        self.predictor = None

        print(
            f"🦙 Lag-Llama initialized - Device: {self.device}, Context: {self.context_length}"
        )

    def _create_predictor_for_horizon(self, forecast_horizon: int):
        """Create a predictor for a specific forecast horizon."""

        # Create the estimator with the specified horizon
        estimator = LagLlamaEstimator(
            prediction_length=forecast_horizon,
            context_length=self.context_length,
            batch_size=self.batch_size,
            num_parallel_samples=self.num_samples,
            device=self.device,
        )

        # Create predictor from estimator
        transformation = estimator.create_transformation()
        lightning_module = estimator.create_lightning_module()
        predictor = estimator.create_predictor(transformation, lightning_module)

        return predictor

    def train(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]],
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        y_start_date: Optional[str] = None,
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
        if y_context is None:
            raise ValueError("y_context is required for Lag-Llama")

        # Convert to DataFrame format
        if isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame({"series": y_context})
            else:
                df = pd.DataFrame(y_context)
        else:
            df = y_context

        print("SHAPE", df.shape)

        print(
            f"TRAIN🦙 Lag-Llama is pre-trained, setting up for forecast_horizon={self.forecast_horizon}"
        )

        # Create predictor for this horizon
        self.predictor = self._create_predictor_for_horizon(self.forecast_horizon)

        # Lag-Llama is pre-trained, so we just mark as fitted
        self.is_fitted = True

        print("✅ Lag-Llama ready (pre-trained)")

        return self

    def fit(self, df: pd.DataFrame, forecast_horizon: int = None, verbose: bool = True):
        """
        Lag-Llama is pre-trained, so this method just validates inputs and sets fitted status.

        Args:
            df: DataFrame with time series data
            forecast_horizon: Forecast horizon (uses model config if not specified)
            verbose: Whether to print progress information
        """
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon

        if verbose:
            print(
                f"🦙 Lag-Llama is pre-trained, setting up for forecast_horizon={forecast_horizon}"
            )

        # Create predictor for this horizon
        self.predictor = self._create_predictor_for_horizon(forecast_horizon)

        # Lag-Llama is pre-trained, so we just mark as fitted
        self.is_fitted = True

        if verbose:
            print("✅ Lag-Llama ready (pre-trained)")

        return self

    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        y_context_timestamps: Optional[np.ndarray] = None,
        y_target_timestamps: Optional[np.ndarray] = None,
        forecast_horizon: Optional[int] = None,
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

        if y_context is None:
            raise ValueError("y_context is required for prediction")

        # Determine forecast horizon from y_target if not provided
        if forecast_horizon is None:
            if y_target is not None:
                horizon = len(y_target)
            else:
                horizon = self.forecast_horizon
        else:
            horizon = forecast_horizon

        # Convert input to DataFrame format
        if isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame({"series": y_context})
            else:
                df = pd.DataFrame(y_context)
        else:
            df = y_context
        # Ensure we have a fitted predictor
        if not self.is_fitted:
            self.train(y_context)

        # Use the internal prediction method
        results = self._predict_internal(df, horizon)

        # return results["mean"]
        # Return as numpy array for single series
        if "series" in results:
            return np.array(results["series"])
        elif len(results) == 1:
            # Return first series as numpy array
            first_series = list(results.values())[0]
            return np.array(first_series)
        else:
            # Fallback: return zeros
            return np.zeros(horizon)

    def _predict_internal(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        freq: str = "D",
        return_samples: bool = False,
    ) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        """Internal prediction method - similar to standalone forecaster"""

        # Use existing predictor or create new one if needed
        if self.predictor is None:
            predictor = self._create_predictor_for_horizon(prediction_length)
        else:
            predictor = self.predictor

        # Prepare data for each series
        all_series_data = []
        series_names = []

        for series_name in df.columns:
            series_data = df[series_name].dropna()

            if len(series_data) == 0:
                warnings.warn(f"Series '{series_name}' has no valid data. Skipping.")
                continue

            # Create timestamps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(series_data) - 1)
            timestamps = pd.date_range(
                start=start_date, periods=len(series_data), freq=freq
            )

            # Create series DataFrame
            series_df = pd.DataFrame(
                {
                    "ds": timestamps,
                    "target": series_data.values,
                    "unique_id": series_name,
                }
            )

            all_series_data.append(series_df)
            series_names.append(series_name)

        if not all_series_data:
            return {}

        # Combine all series
        combined_df = pd.concat(all_series_data, ignore_index=True)

        # Ensure target column is float32 to match model dtype
        combined_df["target"] = combined_df["target"].astype(np.float32)

        # Create GluonTS dataset
        dataset = PandasDataset.from_long_dataframe(
            combined_df, target="target", item_id="unique_id", timestamp="ds", freq=freq
        )

        # Generate forecasts
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset, predictor=predictor, num_samples=self.num_samples
        )

        forecasts = list(forecast_it)

        # Process results
        results = {}
        for forecast in forecasts:
            series_name = getattr(forecast, "item_id", "unknown")

            if return_samples:
                results[series_name] = {
                    "mean": forecast.mean.tolist(),
                    "median": forecast.quantile(0.5).tolist(),
                    "q10": forecast.quantile(0.1).tolist(),
                    "q90": forecast.quantile(0.9).tolist(),
                    "samples": forecast.samples.tolist(),
                }
            else:
                results[series_name] = forecast.mean.tolist()

        return results

    def get_params(self) -> Dict[str, Any]:
        return {
            "context_length": self.context_length,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "forecast_horizon": self.forecast_horizon,
        }

    def set_params(self, **params: Dict[str, Any]) -> "LagllamaModel":
        """Set model parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in self.config:
                self.config[key] = value
        return self

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

    def __init__(self, checkpoint_path: str = "lag-llama.ckpt", **kwargs):
        """Initialize with TimesFM-like interface"""
        config = {"checkpoint_path": checkpoint_path}
        config.update(kwargs)
        self.model = LagllamaModel(config)

    def predict(self, df: pd.DataFrame, forecast_horizon: int, **kwargs):
        """TimesFM-style predict method"""
        return self.model.predict_df(df, forecast_horizon, **kwargs)

    def predict_quantiles(self, df: pd.DataFrame, forecast_horizon: int, **kwargs):
        """TimesFM-style quantile prediction"""
        return self.model.predict_quantiles(df, forecast_horizon, **kwargs)
