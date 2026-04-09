import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Vendored Datadog ``toto`` package lives in this directory as ``toto/``. Its modules use
# ``from toto....`` absolute imports. Those must resolve to the vendored tree, not
# ``tempus_bench.models.toto`` (this file's package), so the wrapper dir must precede
# normal imports of vendored code.
_TOTO_WRAPPER_DIR = Path(__file__).resolve().parent
if str(_TOTO_WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(_TOTO_WRAPPER_DIR))

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel as PydanticBaseModel, Field

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

from tempus_bench.models.base_model import BaseModel
from tempus_bench.models.toto import freq_seconds as _toto_freq_seconds


class TotoHyperparams(PydanticBaseModel):
    # Foundation model with minimal parameters
    pass


class TotoModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TOTO model with configuration.

        Args:
            config: JobConfig instance containing model and task configuration
            logs_path: Directory for storing log files (required)
        """
        super().__init__(params, settings, TotoHyperparams)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.CUBLAS_WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(self.use_deterministic_algorithms)
        torch.device(self.device)
        toto = Toto.from_pretrained(self.hf_model_name)
        toto.to(self.device)
        try:
            toto.compile()
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Toto torch.compile() skipped (inference still works): {exc}",
                UserWarning,
                stacklevel=1,
            )
        self._model = TotoForecaster(toto.model)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "TotoModel":
        """
        Train/fine-tune the foundation model on given data.
        For TOTO, this is a pre-trained model that doesn't require additional training.
        This method sets the fitted flag and returns the model.

        Args:
            y_context: Past target values
            y_target: Future target values (not used for pre-trained model)
            y_start_date: Start date timestamp

        Returns:
            self: The fitted model instance
        """
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Make predictions using the trained TOTO model.

        Args:
            y_context: Recent/past target values
            y_target: Future target values (used to determine the forecast length)
            y_context_timestamps: Timestamps for the context data
            y_target_timestamps: Timestamps for the target data

        Returns:
            np.ndarray: Model prediction samples with shape (num_samples, forecast_horizon, num_targets)
        """

        freq = kwargs["freq"]
        num_samples = kwargs["num_samples"]
        
        # y_context shape: (num_steps, num_variates)
        # timestamps_context shape: (num_steps,)
        # timestamps_target shape: (forecast_horizon,)
        # x_context shape: (num_steps, num_covariates) if provided
        # x_target shape: (forecast_horizon, num_covariates) if provided
        num_steps, num_variates = y_context.shape
        forecast_horizon = len(timestamps_target)

        timestamps_context = timestamps_context / 1e9  # Convert nanoseconds to seconds
        time_diff = _toto_freq_seconds.freq_to_seconds(freq)

        # Build series: [target | covariates]. Exogenous MUST be at the end.
        y_context_tensor = torch.tensor(y_context.T, dtype=torch.float)  # (num_variates, num_steps)
        future_exogenous = None
        num_exogenous = 0

        # TOTO supports past-only, future-only, or both (optional, independent)
        if x_context is not None:
            x_context_tensor = torch.tensor(x_context.T, dtype=torch.float)
            series = torch.cat([y_context_tensor, x_context_tensor], dim=0)
            num_exogenous = x_context.shape[1]
        else:
            series = y_context_tensor
            num_exogenous = 0

        if x_target is not None:
            x_target_trimmed = x_target[:forecast_horizon]
            future_exogenous = torch.tensor(
                x_target_trimmed.T, dtype=torch.float
            ).unsqueeze(0)
        else:
            future_exogenous = None

        series = series.unsqueeze(0)  # (1, num_variates [+ num_covariates], num_steps)
        num_channels = series.shape[1]

        timestamps_context_tensor = torch.tensor(timestamps_context, dtype=torch.float)
        timestamps_context_tensor = timestamps_context_tensor.unsqueeze(0).unsqueeze(0)
        timestamps_context_tensor = timestamps_context_tensor.expand(1, num_channels, -1)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.full_like(series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(series[:, :, :1], dtype=torch.float),
            timestamp_seconds=timestamps_context_tensor,
            time_interval_seconds=torch.full((1, num_channels), time_diff, dtype=torch.float),
            num_exogenous_variables=num_exogenous,
        )
        inputs = inputs.to(self.device)
        if future_exogenous is not None:
            future_exogenous = future_exogenous.to(self.device)

        forecast = self._model.forecast(
            inputs,
            prediction_length=forecast_horizon,
            num_samples=num_samples,
            # Keep generation micro-batches fixed; coupling this to num_samples
            # can create attention-mask shape mismatches in Toto internals.
            samples_per_batch=1,
            future_exogenous_variables=future_exogenous,
        )

        # forecast.samples already excludes exogenous (only target variates)
        forecast_samples = forecast.samples  # (1, num_variates, forecast_horizon, num_samples)
        forecast_samples = forecast_samples.squeeze(0)  # (num_variates, forecast_horizon, num_samples)
        # Transpose to (num_samples, forecast_horizon, num_variates)
        forecast_samples = forecast_samples.permute(2, 1, 0)  # (num_samples, forecast_horizon, num_variates)
        
        # Convert to numpy
        forecast_samples = np.asarray(forecast_samples.cpu())

        return forecast_samples

    def freq_to_seconds(self, freq: Union[str, float, int]) -> float:
        """Delegate to :func:`~tempus_bench.models.toto.freq_seconds.freq_to_seconds`."""
        return _toto_freq_seconds.freq_to_seconds(freq)

    # def _sub_predict(self, input_series: torch.Tensor, time_interval_sec: int = 900) -> dict:
    #     """
    #     Args:
    #         input_series (torch.Tensor): Shape (num_series, time_steps)
    #         time_interval_sec (int): Interval between timesteps, default is 900s (15min)

    #     Returns:
    #         dict with keys: 'median', 'samples', 'quantile_0.1', 'quantile_0.9'
    #     """
    #     num_series, time_steps = input_series.shape
    #     input_series = input_series.to(self.device)

    #     # Dummy timestamp-related info for compatibility
    #     timestamp_seconds = torch.zeros_like(input_series).to(self.device)
    #     time_interval_seconds = torch.full((num_series,), time_interval_sec).to(self.device)

    #     # Construct MaskedTimeseries as expected by TOTO forecaster
    #     inputs = MaskedTimeseries(
    #         series=input_series,
    #         padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
    #         id_mask=torch.zeros_like(input_series),
    #         timestamp_seconds=timestamp_seconds,
    #         time_interval_seconds=time_interval_seconds,
    #     )

    #     # Generate forecasts using the forecaster
    #     forecasts = self.forecaster.forecast(
    #         inputs,
    #         prediction_length=self.forecast_horizon,
    #         num_samples=self.m

    #     # Convert forecasts to the expected format
    #     if hasattr(forecasts, 'median'):
    #         median_forecast = np.array(forecasts.median)
    #     else:
    #         # If forecasts is a tensor/array of samples, compute median
    #         median_forecast = np.median(forecasts, axis=0)

    #     # Ensure we have the correct shape for univariate forecasting
    #     # The expected output should be (prediction_length,) for univariate
    #     if median_forecast.ndim > 1:
    #         # If we have multiple dimensions, take the first series
    #         if median_forecast.shape[0] == 1:
    #             # If first dimension is 1, squeeze it
    #             median_forecast = median_forecast.squeeze(0)
    #         else:
    #             # Take the first series
    #             median_forecast = median_forecast[0]

    #     # Ensure the output has the correct length
    #     if len(median_forecast) != self.forecast_horizon:
    #         print(f"Warning: Expected forecast length {self.forecast_horizon}, got {len(median_forecast)}")
    #         # Truncate or pad if necessary
    #         if len(median_forecast) > self.forecast_horizon:
    #             median_forecast = median_forecast[:self.forecast_horizon]
    #         else:
    #             # Pad with the last value if too short
    #             last_val = median_forecast[-1] if len(median_forecast) > 0 else 0
    #             median_forecast = np.pad(median_forecast, (0, self.forecast_horizon - len(median_forecast)), mode='constant', constant_values=last_val)

    #     # Clean up tensors to free memory
    #     del inputs, forecasts
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     return median_forecast
