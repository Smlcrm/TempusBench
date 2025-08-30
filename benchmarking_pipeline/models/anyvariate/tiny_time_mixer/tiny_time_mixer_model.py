import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class TinyTimeMixerModel(BaseModel):

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
          prediction length: any positive integer that shows many steps to forecast
        """

        super().__init__(config)

        # forecast_horizon is inherited from parent class (FoundationModel)
        self.model = None

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
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "TinyTimeMixerModel":
        """
        Train/fine-tune the foundation model on given data.

        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            x_context: Past exogenous variables - used during tuning and testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            x_target: Future exogenous variables - if provided, can be used with x_context for training
            y_start_date: The start date timestamp for y_context and y_target in string form
            x_start_date: The start date timestamp for x_context and x_target in string form

        Returns:
            self: The fitted model instance
        """
        # TinyTimeMixer is a zero-shot model, so training is not needed
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):

        forecast_horizon = timestamps_target.shape[0]

        # Construct DataFrame
        columns = list(range(y_context.shape[1]))

        timestamps_context = self.convert_to_datetimeindex(timestamps_context)

        df = pd.DataFrame(y_context, index=timestamps_context, columns=columns)

        results = self._sub_predict(df, forecast_horizon)
        results = np.asarray(results)
        # if len(list(results.keys())) == 1:
        #     return np.array(results["1"])
        # else:
        #     multivariate_values = []
        #     for key in results.keys():
        #         multivariate_values.append(results[key])
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
