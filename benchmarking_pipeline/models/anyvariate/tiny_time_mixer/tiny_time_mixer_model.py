import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.foundation_model import FoundationModel
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class TinyTimeMixerModel(FoundationModel):

  def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
    """
    Args:
      prediction length: any positive integer that shows many steps to forecast
    """
    
    super().__init__(config, config_file)
    if 'model_name' not in self.config:
        raise ValueError("model_name must be specified in config")
    
    self.model_name = self.config['model_name']
    # forecast_horizon is inherited from parent class (FoundationModel)
    self.model = None
  
  def set_params(self, **params: Dict[str, Any]) -> 'TinyTimeMixerModel':
    for key, value in params.items():
      if hasattr(self, key):
        setattr(self, key, value)
    return self
  
  def train(self, 
            y_context: Optional[Union[pd.Series, np.ndarray]], 
            x_context: Optional[Union[pd.Series, np.ndarray]] = None, 
            y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
            x_target: Optional[Union[pd.Series, np.ndarray]] = None,
            y_start_date: Optional[str] = None,
            x_start_date: Optional[str] = None
  ) -> 'TinyTimeMixerModel':
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
  
  def get_params(self) -> Dict[str, Any]:
    """
    Get the current model parameters.
    
    Returns:
        Dict[str, Any]: Dictionary of current model parameters
    """
    return {
        'model_name': self.model_name,
        'forecast_horizon': self.forecast_horizon,
        'is_fitted': self.is_fitted
    }
         
  def predict(self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_context_timestamps = None,
        y_target_timestamps = None,
        **kwargs):
    #print("HUH")
    #print(y_target)
    #print("YUHUHU?")
    #print(y_target_timestamps[0].strftime('%Y-%m-%d %X'))
    #raise Exception("UNgas")
    #timestamp_strings = [ts.strftime('%Y-%m-%d %X') for ts in y_target_timestamps]
    
    # Construct DataFrame
    if len(y_context.shape) == 1:
      columns = ['1']
    else:
      columns = list(range(y_context.shape[0])) 
    df = pd.DataFrame(y_context, index=y_context_timestamps, columns=columns)
    self.ctx = len(df)
    results = self._sub_predict(df)
    if len(list(results.keys())) == 1:
      return np.array(results["1"])
    else:
      multivariate_values = []
      for key in results.keys():
        multivariate_values.append(results[key])
      return np.array(multivariate_values)
  

      
  
  def _sub_predict(self, dataframe: pd.DataFrame):
    """
    We assume dataframe is in the following format:
    Its index column is a bunch of date timestamps.
    We assume each of the rest of its columns are a different time series.
    The header of these columns should be some identifying mark distinguishing 
    the time series from each other. The actual name chosen does not matter.
    """

    forecaster = TinyTimeMixerForecaster() 

    # Fit the model
    forecaster.fit(dataframe, fh=list(range(1, self.forecast_horizon + 1)))
    y_pred = forecaster.predict() 

    all_time_series_names = dataframe.columns.values

    results_dict = dict()
    for time_series_name in all_time_series_names:
        results_dict[time_series_name] = []

    forecaster.fit(dataframe, fh=list(range(1,901))) 

    forecast = forecaster.predict() 
    
    forecast = forecast[all_time_series_names[0]].values
        
    if len(forecast.shape) == 1:
      results_dict[all_time_series_names[0]].extend(forecast)

    else:
      # The forecast is originally of shape [time series step, time series].
      # I want to change it to shape [time series, time series step]
      forecast = np.reshape(forecast, (forecast.shape[1], -1))

      # This is just an index that tracks which forecast we want to add to which mapping in our results dict.
      current_forecasted_timeseries_idx = 0
      while current_forecasted_timeseries_idx < len(all_time_series_names):
        current_time_series_name = all_time_series_names[current_forecasted_timeseries_idx]
        current_forecast = forecast[current_forecasted_timeseries_idx]

        results_dict[current_time_series_name].extend(current_forecast)
        current_forecasted_timeseries_idx += 1
    
    
    return results_dict

    