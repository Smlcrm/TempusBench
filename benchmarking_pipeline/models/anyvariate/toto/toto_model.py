import os
import torch
import numpy as np
import pandas as pd

from .toto.model.toto import Toto
from .toto.data.util.dataset import MaskedTimeseries
from .toto.inference.forecaster import TotoForecaster
from benchmarking_pipeline.models.foundation_model import FoundationModel
from typing import Optional, Union, Dict, Any


class TotoModel(FoundationModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize TOTO model with configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
            config_file: Path to configuration file
        """
        super().__init__(config, config_file)
        
        # Extract model-specific config
        model_config = self._extract_model_config(self.config)
        
        # Required parameters - no defaults
        if 'num_samples' not in model_config:
            raise ValueError("num_samples must be specified in config")
        if 'samples_per_batch' not in model_config:
            raise ValueError("samples_per_batch must be specified in config")
        

        self.num_samples = model_config['num_samples']
        self.samples_per_batch = model_config['samples_per_batch']
        # forecast_horizon is inherited from parent class (FoundationModel)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(self.device)

        # JIT compilation for faster inference
        self.model.compile()  
        self.forecaster = TotoForecaster(self.model.model)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'num_samples': self.num_samples,
            'samples_per_batch': self.samples_per_batch,
            'forecast_horizon': self.forecast_horizon
        }
    
    def set_params(self, **params: Dict[str, Any]) -> 'TotoModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray]], 
              y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_start_date: Optional[str] = None
    ) -> 'TotoModel':
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
    
    def predict(self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None,
        y_context_timestamps = None,
        y_target_timestamps = None,
        **kwargs) -> np.ndarray:
        """
        Make predictions using the trained TOTO model.
        
        Args:
            y_context: Recent/past target values
            y_target: Future target values (used to determine the forecast length)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            y_context_timestamps: Timestamps for the context data
            y_target_timestamps: Timestamps for the target data
            
        Returns:
            np.ndarray: Model predictions with shape (prediction_length,)
        """
        # Determine prediction length
        if y_target is not None:
            # Use target length as prediction length if provided
            if isinstance(y_target, pd.Series):
                prediction_length = len(y_target)
            else:
                # For time series, we always want the number of time steps (first dimension)
                prediction_length = y_target.shape[0]
        elif forecast_horizon is not None:
            prediction_length = forecast_horizon
        else:
            prediction_length = self.forecast_horizon
        
        # Update forecast_horizon temporarily
        original_horizon = self.forecast_horizon
        self.forecast_horizon = prediction_length
        
        # Convert to numpy if needed
        if isinstance(y_context, pd.Series):
            y_context = y_context.values
        
        if len(y_context.shape) == 1:
            y_context = y_context.reshape(1, -1)
        
        input_tensor = torch.from_numpy(y_context).float()
        
        # Handle timestamps if available
        if y_context_timestamps is not None and len(y_context_timestamps) > 1:
            # Handle different timestamp formats
            if hasattr(y_context_timestamps[0], 'total_seconds'):
                # Timestamps are datetime/timedelta objects
                time_series_delta = int((y_context_timestamps[1] - y_context_timestamps[0]).total_seconds())
            else:
                # Timestamps might be numeric (assume they represent time intervals in seconds)
                time_series_delta = int(y_context_timestamps[1] - y_context_timestamps[0])
                # If the delta is very large, it might be in different units, use default
                if time_series_delta > 86400:  # More than a day 
                    time_series_delta = 900  # Default 15 minutes
        else:
            time_series_delta = 900  # Default 15 minutes
        
        result = self._sub_predict(input_tensor, time_series_delta)
        
        # Restore original forecast_horizon
        self.forecast_horizon = original_horizon
            
        return result

    def _sub_predict(self, input_series: torch.Tensor, time_interval_sec: int = 900) -> dict:
        """
        Args:
            input_series (torch.Tensor): Shape (num_series, time_steps)
            time_interval_sec (int): Interval between timesteps, default is 900s (15min)

        Returns:
            dict with keys: 'median', 'samples', 'quantile_0.1', 'quantile_0.9'
        """
        num_series, time_steps = input_series.shape
        input_series = input_series.to(self.device)

        # Dummy timestamp-related info for compatibility
        timestamp_seconds = torch.zeros_like(input_series).to(self.device)
        time_interval_seconds = torch.full((num_series,), time_interval_sec).to(self.device)

        # Construct MaskedTimeseries as expected by TOTO forecaster
        inputs = MaskedTimeseries(
            series=input_series,
            padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(input_series),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )
        
        # Generate forecasts using the forecaster
        forecasts = self.forecaster.forecast(
            inputs,
            prediction_length=self.forecast_horizon,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch
        )

        # Convert forecasts to the expected format
        if hasattr(forecasts, 'median'):
            median_forecast = np.array(forecasts.median)
        else:
            # If forecasts is a tensor/array of samples, compute median
            median_forecast = np.median(forecasts, axis=0)
        
        # For univariate case, return only the first series forecast
        if median_forecast.ndim > 1:
            median_forecast = median_forecast[0]
            
        return median_forecast