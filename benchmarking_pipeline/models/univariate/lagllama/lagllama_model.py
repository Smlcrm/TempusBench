import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import warnings
import os
import subprocess
import sys
from pathlib import Path
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from benchmarking_pipeline.models.base_model import BaseModel

from .lag_llama.gluon.estimator import LagLlamaEstimator

# Try to import lag_llama, install if not available


class LagllamaModel(BaseModel):
    """
    Lag-Llama model implementation that inherits from BaseModel.
    Works seamlessly like TimesFM with automatic setup.
    """

    def __init__(self, config: Dict[str, Any]):
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
        super().__init__(config)
        
        # Store the full config for creating variate models
        self.full_config = config

        # Detect if this is anyvariate mode based on config filename
        self.is_anyvariate = False
        if hasattr(self, 'model_config') and 'original_config_path' in config:
            config_filename = Path(config['original_config_path']).name.lower()
            if "univariate" in config_filename:
                self.is_anyvariate = True
                print(f"[LAGLLAMA] Anyvariate mode enabled for config: {config_filename}")

        # Set training loss to one of the configured metrics if mae is not available
        if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'metrics_to_calculate'):
            if 'mae' not in self.evaluator.metrics_to_calculate:
                # Use the first available metric for training loss optimization
                available_metrics = self.evaluator.metrics_to_calculate
                if 'rmse' in available_metrics:
                    self.training_loss = 'rmse'
                elif 'mse' in available_metrics:
                    self.training_loss = 'mse'
                elif len(available_metrics) > 0:
                    self.training_loss = available_metrics[0]

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model-specific attributes
        self.model_config["context_length"] = 32
        self.model_config["num_samples"] = 10
        self.model_config["batch_size"] = 1
        self.model_config["batch_size"] = 1

        self.model = None
        self.models = []  # For anyvariate mode

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
        periods = y_context.shape[0]
        timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Create series DataFrame
        context_df = pd.DataFrame(
            {
                "ds": timestamps,
                "target": y_context[:, 0],
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

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ):
        """
        Train the LagLlama model. Supports anyvariate mode for handling multivariate data.
        
        Args:
            y_context: Past target values - training data
            y_target: Future target values (for validation)
            timestamps_context: Timestamps for y_context
            timestamps_target: Timestamps for y_target
            freq: Frequency string
            **kwargs: Additional keyword arguments
            
        Returns:
            self: The fitted model instance
        """
        # Ensure y_context is 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
        if y_target is not None and y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)

        n_variates = y_context.shape[1]

        # For univariate case or non-anyvariate mode, use the original method
        if n_variates == 1 or not self.is_anyvariate:
            return self._train(y_context, y_target, timestamps_context, timestamps_target, freq, **kwargs)

        # For anyvariate mode with multivariate data, train separate LagLlama models for each variate
        self.models = []  # Store individual models for each variate

        for variate_idx in range(n_variates):
            print(f"[INFO] Training LagLlama model for variate {variate_idx + 1}/{n_variates}")

            # Create a new LagLlama model for this variate
            variate_model = LagllamaModel(self.full_config)

            # Extract data for this specific variate
            variate_context = y_context[:, variate_idx:variate_idx+1]
            variate_target = y_target[:, variate_idx:variate_idx+1] if y_target is not None else None

            # Train the model for this variate
            variate_model._train(
                y_context=variate_context,
                y_target=variate_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs
            )

            # Store the trained model
            self.models.append(variate_model)

        # Mark the main model as fitted for anyvariate case
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray],
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained LagLlama model. Supports anyvariate mode.
        
        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for y_context
            timestamps_target: Timestamps for the prediction horizon
            freq: Frequency string
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Model predictions
        """
        # Ensure y_context is 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        n_variates = y_context.shape[1]

        # For univariate case or non-anyvariate mode, use the original method
        if n_variates == 1 or not self.is_anyvariate:
            return self._predict(y_context, timestamps_context, timestamps_target, freq, **kwargs)

        # For anyvariate mode with multivariate data, predict using each variate's model
        if not self.models:
            raise ValueError("No variate models found. Train the model first.")

        all_predictions = []
        for variate_idx in range(n_variates):
            print(f"[INFO] Predicting with LagLlama model for variate {variate_idx + 1}/{n_variates}")
            
            # Extract data for this specific variate
            variate_context = y_context[:, variate_idx:variate_idx+1]
            
            # Make predictions for this variate
            variate_pred = self.models[variate_idx]._predict(
                y_context=variate_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs
            )
            
            all_predictions.append(variate_pred)

        # Combine predictions from all variates
        combined_predictions = np.concatenate(all_predictions, axis=1)
        
        # Store for TensorBoard logging (use first variate as representative)
        self._last_y_pred = all_predictions[0]
        
        return combined_predictions

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        loss_function: str = None,
        y_train: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute loss metrics, supporting per-variate metrics in anyvariate mode.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Loss function to use (inherited from parent if None)
            y_train: Training data for metrics like MASE
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        # Ensure inputs are 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_train is not None and y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        n_variates = y_true.shape[1]

        # For univariate case or non-anyvariate mode, use the parent method
        if n_variates == 1 or not self.is_anyvariate:
            return super().compute_loss(y_true, y_pred, loss_function, y_train)

        # For anyvariate mode, compute metrics per variate
        all_metrics = {}
        aggregated_metrics = {}

        for variate_idx in range(n_variates):
            # Extract data for this variate
            y_true_variate = y_true[:, variate_idx:variate_idx+1]
            y_pred_variate = y_pred[:, variate_idx:variate_idx+1]
            y_train_variate = y_train[:, variate_idx:variate_idx+1] if y_train is not None else None

            # Compute metrics for this variate using the evaluator
            variate_metrics = self.evaluator.evaluate(y_pred_variate, y_true_variate, y_train=y_train_variate)

            # Add variate index to metric names for per-variate tracking
            for metric_name, metric_value in variate_metrics.items():
                all_metrics[f"{metric_name}_variate_{variate_idx + 1}"] = metric_value

                # Also accumulate for aggregated metrics
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = []
                aggregated_metrics[metric_name].append(metric_value)

        # Add aggregated metrics (mean across variates) for hyperparameter optimization
        for metric_name, values in aggregated_metrics.items():
            all_metrics[metric_name] = np.mean(values)

        # Store for TensorBoard logging (use first variate as representative)
        self._last_y_true = y_true[:, 0]
        self._last_y_pred = y_pred[:, 0]

        return all_metrics


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
