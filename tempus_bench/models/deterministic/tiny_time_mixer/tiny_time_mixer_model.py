import numpy as np
import pandas as pd

from typing import Dict, Any, Union, Tuple, Optional
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class TinyTimeMixerParams(PydanticBaseModel):
    # Foundation model with minimal parameters
    pass


class TinyTimeMixerModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TinyTimeMixer model.
        
        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, TinyTimeMixerParams)

        # forecast_horizon is inherited from parent class (BaseModel)
        self._model = None

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

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "TinyTimeMixerModel":
        """
        Train/fine-tune the foundation model on given data.

        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        # TinyTimeMixer is a foundation model with minimal params
        
        # TinyTimeMixer is a zero-shot model, so training is not needed
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ):

        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        # TinyTimeMixer is a foundation model with minimal params

        forecast_horizon = timestamps_target.shape[0]

        # Handle (num_steps, num_targets) format
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
            
        num_steps, num_targets = y_context.shape

        # Construct DataFrame - Tiny Time Mixer expects (timesteps, num_targets)
        columns = list(range(num_targets))
        timestamps_context = self.convert_to_datetimeindex(timestamps_context)

        # Data is already in (timesteps, num_targets) format for Tiny Time Mixer
        df = pd.DataFrame(y_context, index=timestamps_context, columns=columns)

        results = self._sub_predict(df, forecast_horizon)
        results = np.asarray(results)
        
        # Convert to (forecast_horizon, num_targets) format
        if results.ndim == 1:
            # Univariate case
            results = results.reshape(-1, 1)
        elif results.ndim == 2 and results.shape[1] == forecast_horizon:
            # Results are (num_targets, forecast_horizon), transpose to (forecast_horizon, num_targets)
            results = results.T
        
        return results

    def _sub_predict(self, dataframe: pd.DataFrame, forecast_horizon):
        """
        We assume dataframe is in the following format:
        Its index column is a bunch of date timestamps.
        We assume each of the rest of its columns are a different time series.
        The header of these columns should be some identifying mark distinguishing
        the time series from each other. The actual name chosen does not matter.
        """

        forecaster = TinyTimeMixerForecaster()

        # Fit the model
        forecaster.fit(dataframe, fh=list(range(1, forecast_horizon + 1)))
        forecast = forecaster.predict()

        # all_time_series_names = dataframe.columns.values

        # results_dict = dict()
        # for time_series_name in all_time_series_names:
        #     results_dict[time_series_name] = []

        # forecaster.fit(dataframe, fh=list(range(1, 901)))

        # forecast = forecaster.predict()

        # forecast = forecast[all_time_series_names[0]].values

        # if len(forecast.shape) == 1:
        #     results_dict[all_time_series_names[0]].extend(forecast)

        # else:
        #     # The forecast is originally of shape [time series step, time series].
        #     # I want to change it to shape [time series, time series step]
        #     forecast = np.reshape(forecast, (forecast.shape[1], -1))

        #     # This is just an index that tracks which forecast we want to add to which mapping in our results dict.
        #     current_forecasted_timeseries_idx = 0
        #     while current_forecasted_timeseries_idx < len(all_time_series_names):
        #         current_time_series_name = all_time_series_names[
        #             current_forecasted_timeseries_idx
        #         ]
        #         current_forecast = forecast[current_forecasted_timeseries_idx]

        #         results_dict[current_time_series_name].extend(current_forecast)
        #         current_forecasted_timeseries_idx += 1

        # print(y_pred)
        # print("SHAPE", y_pred.shape)
        # exit()
        return forecast
