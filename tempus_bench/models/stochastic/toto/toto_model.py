import os
import torch
import numpy as np
import pandas as pd
import re

from .toto.model.toto import Toto
from .toto.data.util.dataset import MaskedTimeseries
from .toto.inference.forecaster import TotoForecaster
from benchmarking_pipeline.models.base_model import BaseModel
from typing import Optional, Union, Dict, Any


class TotoModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TOTO model with configuration.

        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
        toto.to(self.device)

        # JIT compilation for faster inference
        toto.compile()
        self.model = TotoForecaster(toto.model)

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
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
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained TOTO model.

        Args:
            y_context: Recent/past target values
            y_target: Future target values (used to determine the forecast length)
            y_context_timestamps: Timestamps for the context data
            y_target_timestamps: Timestamps for the target data

        Returns:
            np.ndarray: Model predictions with shape (prediction_length,)
        """

        forecast_horizon, num_variates = timestamps_target.shape

        timestamps_context = timestamps_context / 1e9  # Convert nanoseconds to seconds

        y_context = torch.tensor(y_context.T, dtype=torch.float)
        timestamps_context = torch.tensor(timestamps_context.T, dtype=torch.float)
        time_diff = self.freq_to_seconds(freq)

        # Create a MaskedTimeseries object
        inputs = MaskedTimeseries(
            series=y_context,
            padding_mask=torch.full_like(y_context, True, dtype=torch.bool),
            id_mask=torch.zeros_like(y_context, dtype=torch.float),
            timestamp_seconds=timestamps_context,
            time_interval_seconds=torch.full((num_variates,), time_diff, dtype=torch.float),
        )

        # Generate forecasts for the next 336 timesteps
        forecast = self.model.forecast(
            inputs,
            prediction_length=forecast_horizon,
            num_samples=50,  # Number of samples for probabilistic forecasting
            samples_per_batch=50,  # Control memory usage during inference
        )

        forecast_samples = np.squeeze(np.asarray(forecast.samples), axis=0)

        forecast = np.mean(forecast_samples, axis=-1).T

        return forecast

    def freq_to_seconds(self, freq: Union[str, float, int]) -> float:
        """
        Convert a frequency string with an increment to seconds.

        Accepts forms like:
        - '15m', '30min', '45sec', '2h', '1.5h'
        - '4w', '12mth', '1y', '250ms', '10us', '100ns'
        - pandas-style short forms are fine: '2H', '15MIN', '30S'
        - week anchors like 'W-MON' are treated as a week
        Returns:
        float seconds
        """
        if isinstance(freq, (int, float)):
            # Assume already seconds if a number is given
            return float(freq)

        if not isinstance(freq, str) or not freq.strip():
            raise ValueError(f"Unsupported frequency: {freq!r}")

        s = freq.strip().lower().replace("µs", "us")

        # Handle pandas week anchors like 'w-mon', 'w-fri' → treat as 1 week
        if s.startswith("w-"):
            return 7 * 24 * 3600.0

        # Extract numeric value + unit (e.g., '15m', '30min', '1.5h', '250ms')
        m = re.fullmatch(r"\s*(?P<val>[+-]?\d*\.?\d+)\s*(?P<unit>[a-z\-]+)\s*", s)
        if not m:
            # Also allow pure units like 'h' (implied 1)
            m = re.fullmatch(r"\s*(?P<unit>[a-z\-]+)\s*", s)
            if m:
                val = 1.0
                unit = m.group("unit")
            else:
                raise ValueError(f"Could not parse frequency string: {freq!r}")
        else:
            val = float(m.group("val"))
            unit = m.group("unit")

        # Canonicalize common aliases
        aliases = {
            # sub-second
            "ns": "ns",
            "nanosecond": "ns",
            "nanoseconds": "ns",
            "us": "us",
            "microsecond": "us",
            "microseconds": "us",
            "ms": "ms",
            "millisecond": "ms",
            "milliseconds": "ms",
            # seconds
            "s": "s",
            "sec": "s",
            "secs": "s",
            "second": "s",
            "seconds": "s",
            # minutes
            "m": "min",
            "min": "min",
            "mins": "min",
            "t": "min",
            "minute": "min",
            "minutes": "min",
            # hours
            "h": "h",
            "hr": "h",
            "hrs": "h",
            "hour": "h",
            "hours": "h",
            # days
            "d": "d",
            "day": "d",
            "days": "d",
            # weeks
            "w": "w",
            "wk": "w",
            "wks": "w",
            "week": "w",
            "weeks": "w",
            # months (calendar-average)
            "mth": "mon",
            "mths": "mon",
            "mo": "mon",
            "mon": "mon",
            "month": "mon",
            "months": "mon",
            # years (calendar-average)
            "y": "y",
            "yr": "y",
            "yrs": "y",
            "year": "y",
            "years": "y",
            # explicit words sometimes seen
            "minu": "min",
            "mins.": "min",
            "sec.": "s",
        }

        unit = aliases.get(unit, unit)  # fold alias

        # Seconds per unit (months/years use astronomical averages)
        SECS = {
            "ns": 1e-9,
            "us": 1e-6,
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
            "d": 86400.0,
            "w": 7 * 86400.0,
            # Averages: 365.25 days/year, 12 months/year → ~30.44 days/month
            "mon": 365.25 / 12 * 86400.0,  # ≈ 2_629_746 seconds
            "y": 365.25 * 86400.0,  # ≈ 31_557_600 seconds
        }

        # Also accept pandas-like unit spellings directly:
        # 'sec', 'second', 'minutes', etc. handled via aliases above.
        if unit not in SECS:
            # Try a few pandas-like quirks: 'w-mon' already handled; 'qs', 'a' not supported as they’re not fixed.
            # If someone passes 'w-mon', we caught it at the top. Anything else unknown → error.
            raise ValueError(f"Unsupported or non-fixed frequency unit: {unit!r} from {freq!r}")

        return float(val) * SECS[unit]

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
