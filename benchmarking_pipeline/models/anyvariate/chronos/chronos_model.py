"""
Chronos foundation model implementation for time series forecasting.

This module provides a wrapper around the Amazon Chronos foundation model for time series
forecasting. Chronos is a large language model specifically designed for time series
forecasting tasks and can handle both univariate and multivariate data.

The model supports multiple sizes (tiny, mini, small, base, large) and can be configured
with different context lengths and sampling strategies.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Union, Tuple, List, Optional
from benchmarking_pipeline.models.base_model import BaseModel
from chronos import ChronosPipeline as BaseChronosPipeline
from einops import rearrange


class ChronosModel(BaseModel):
    """
    Chronos foundation model wrapper for time series forecasting.

    This class provides a unified interface for the Amazon Chronos model, which is
    a large language model specifically designed for time series forecasting.

    Attributes:
        model_size: Size of the Chronos model ('tiny', 'mini', 'small', 'base', 'large')
        context_length: Number of past time steps used as context
        num_samples: Number of predictive samples to generate
        pipeline: The underlying Chronos pipeline instance
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Chronos model wrapper.

        Args:
            config: Configuration dictionary containing model parameters
                - model_size: str, size of the Chronos model (default: 'small')
                - context_length: int, number of past time steps for context (default: 8)
                - num_samples: int, number of predictive samples (default: 5)
            config_file: Path to a JSON configuration file
        """
        super().__init__(config)

        self.model_config["model_size"] = (
            "tiny"  # Valid model sizes = {'tiny', 'mini', 'small', 'base', 'large'}
        )
        self.model_config["context_length"] = 512
        self.model_config["num_samples"] = 10

        # Initialize model state
        self.is_fitted = False

        # forecast_horizon is inherited from parent class (FoundationModel)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ChronosModel":
        """
        Initialize the Chronos model (no training required for foundation models).

        Args:
            y_context: Past target values (not used for training, for compatibility)
            y_target: Future target values (not used for training, for compatibility)
            y_start_date: Start date for y_context (not used)

        Returns:
            self: The model instance

        Note:
            Chronos is a pre-trained foundation model that doesn't require training.
            This method just marks the model as ready for inference.
        """
        # For foundation models, we don't need to load the model here
        # It will be loaded fresh for each prediction (like it was in the working version)
        # Load the Chronos model fresh for each prediction (like the working version)
        hf_model_name = f"amazon/chronos-t5-{self.model_config['model_size']}"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        self.model = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Chronos model loaded successfully!")

        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray = None,
        timestamps_context: np.ndarray = None,
        timestamps_target: np.ndarray = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Chronos model.

        Args:
            y_context: Recent target values for context
            y_target: Target values to predict (used to determine forecast length)
            y_context_timestamps: Timestamps for context data
            y_target_timestamps: Timestamps for target data
            forecast_horizon: Number of steps to forecast (overrides y_target length if provided)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions

        Raises:
            ValueError: If model is not fitted or required data is missing
        """

        forecast_horizon = timestamps_target.shape[0]

        padding_length = self.model_config["context_length"] - y_context.shape[0]
        if padding_length <= 0:
            # Use the most recent context_length data points
            y_context = y_context[-self.model_config["context_length"] :, :]
        else:
            # If not enough data, pad with the last available value
            y_context = np.pad(
                y_context,
                ((padding_length, 0), (0, 0)),
                mode="constant",
                constant_values=torch.nan,
            )

        y_context = torch.tensor(y_context.T)
        # Generate forecasts
        forecasts = self.model.predict(
            context=y_context,
            prediction_length=forecast_horizon,
            num_samples=self.model_config["num_samples"],
        )
        forecasts = np.squeeze(np.asarray(forecasts))
        forecasts = np.mean(forecasts, axis=0, keepdims=True).T

        return forecasts

        # y_context =
        # # Use the working approach: load model fresh and convert to proper format
        # results = self._sub_predict(df, prediction_length)

        # # Process results based on data dimensionality
        # if len(list(results.keys())) == 1:
        #     # Univariate result - always expect '1' as per working commit 434d3b0e
        #     series_vals = np.array(results["1"])  # shape (pred_len,)

        #     if series_vals.ndim > 1:
        #         series_vals = series_vals.squeeze()
        #     if series_vals.shape[0] > prediction_length:
        #         series_vals = series_vals[:prediction_length]
        #     return series_vals
        # else:
        #     # Multivariate result
        #     multivariate_values = []
        #     for key in results.keys():
        #         vals = np.array(results[key])
        #         if vals.ndim > 1:
        #             vals = vals.squeeze()
        #         if vals.shape[0] > prediction_length:
        #             vals = vals[:prediction_length]
        #         multivariate_values.append(vals)
        #     preds = np.array(multivariate_values)  # shape (num_targets, pred_len)
        #     # Ensure exact horizon length
        #     if preds.shape[1] > prediction_length:
        #         preds = preds[:, :prediction_length]
        # return preds

    # def _sub_predict(
    #     self, df: pd.DataFrame, prediction_length: int
    # ) -> Dict[str, List[float]]:
    #     """
    #     Generates forecasts for future time steps based on the most recent data.

    #     This method uses the last `context_length` data points from each time series
    #     in the DataFrame to predict the next `prediction_length` steps.

    #     Args:
    #         df: DataFrame containing time series data with timestamps as index
    #         prediction_length: Number of future time steps to predict

    #     Returns:
    #         Dict[str, List[float]]: A dictionary where keys are time series names (column headers)
    #                                 and values are the list of forecasted points.
    #     """

    #     # Create one context window for each time series
    #     all_contexts = []
    #     for series_name in df.columns:
    #         series_data = df[series_name].values

    #         # Intelligent context selection: use more recent data for better predictions
    #         if len(series_data) >= self.model_config["context_length"]:
    #             # Use the most recent context_length data points
    #             context_data = series_data[-self.model_config["context_length"] :]
    #         else:
    #             # If not enough data, pad with the last available value
    #             context_data = np.full(
    #                 self.model_config["context_length"], series_data[-1]
    #             )
    #             context_data[-len(series_data) :] = series_data

    #         # Ensure data is properly formatted for Chronos
    #         context_data = np.asarray(context_data, dtype=np.float32)

    #         # Handle any NaN values
    #         if np.any(np.isnan(context_data)):
    #             context_data = np.nan_to_num(context_data, nan=0.0)

    #         all_contexts.append(torch.tensor(context_data, dtype=torch.float32))

    #     # Generate forecasts
    #     all_forecasts = pipeline.predict(
    #         context=all_contexts,
    #         prediction_length=prediction_length,
    #         num_samples=self.model_config["num_samples"],
    #     )

    #     # Process results
    #     results = {}
    #     for i, series_name in enumerate(df.columns):
    #         # For each series, aggregate the prediction samples intelligently
    #         forecasts = all_forecasts[i]  # shape: (num_samples, prediction_length)

    #         # Convert PyTorch tensor to numpy array
    #         if hasattr(forecasts, "cpu"):
    #             forecasts = forecasts.cpu().numpy()
    #         elif hasattr(forecasts, "numpy"):
    #             forecasts = forecasts.numpy()
    #         else:
    #             forecasts = np.array(forecasts)

    #         if self.model_config["num_samples"] > 1:
    #             # Use weighted average: give more weight to more recent predictions
    #             weights = np.linspace(0.5, 1.0, self.model_config["num_samples"])
    #             weights = weights / np.sum(weights)

    #             # Weighted average across samples
    #             weighted_forecast = np.average(forecasts, axis=0, weights=weights)

    #             # Also compute median as fallback
    #             median_forecast = np.median(forecasts, axis=0)

    #             # Use the better of the two (lower variance)
    #             # Calculate variance manually to avoid numpy version issues
    #             forecast_variance = np.mean(
    #                 (forecasts - np.mean(forecasts, axis=0)) ** 2, axis=0
    #             )
    #             if np.mean(forecast_variance) < 0.1:  # Low variance = use weighted avg
    #                 final_forecast = weighted_forecast
    #             else:  # High variance = use median (more robust)
    #                 final_forecast = median_forecast
    #         else:
    #             final_forecast = forecasts[0]  # Single sample

    #         results[series_name] = final_forecast.tolist()

    #     # DEBUG: Print what we're returning
    #     print(f"[CHRONOS DEBUG] _sub_predict returning keys: {list(results.keys())}")
    #     print(f"[CHRONOS DEBUG] _sub_predict returning content: {results}")

    #     return results

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Chronos model's properties.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            "model_type": "Chronos",
            "model_size": self.model_config["model_size"],
            "context_length": self.model_config["context_length"],
            "num_samples": self.model_config["num_samples"],
            "forecast_horizon": self.forecast_horizon,
            "is_fitted": self.is_fitted,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
