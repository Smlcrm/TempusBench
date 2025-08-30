import os
import torch
import numpy as np
import pandas as pd

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
        
        if 'num_samples' not in model_config:
            raise ValueError("num_samples must be specified in config")
        if 'samples_per_batch' not in model_config:
            raise ValueError("samples_per_batch must be specified in config")
        
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.num_samples = model_config['num_samples']
        self.samples_per_batch = model_config['samples_per_batch']
        
        
        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(self.device)

        # JIT compilation for faster inference
        self.model.compile()  
        self.forecaster = TotoForecaster(self.model.model)
    
    def train(self, 
            y_context: Optional[np.ndarray],
            y_target: Optional[np.ndarray] = None,
            timestamps_context: Optional[np.ndarray] = None,
            timestamps_target: Optional[np.ndarray] = None,
            freq: str = None,
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
            y_context: Optional[np.ndarray] = None,
            timestamps_context: Optional[np.ndarray] = None,
            timestamps_target: Optional[np.ndarray] = None,
            freq: str = None,
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

        forecast_horizon = timestamps_target.shape[0]
        
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
        
        # Ensure we have the correct shape for univariate forecasting
        # The expected output should be (prediction_length,) for univariate
        if median_forecast.ndim > 1:
            # If we have multiple dimensions, take the first series
            if median_forecast.shape[0] == 1:
                # If first dimension is 1, squeeze it
                median_forecast = median_forecast.squeeze(0)
            else:
                # Take the first series
                median_forecast = median_forecast[0]
        
        # Ensure the output has the correct length
        if len(median_forecast) != self.forecast_horizon:
            print(f"Warning: Expected forecast length {self.forecast_horizon}, got {len(median_forecast)}")
            # Truncate or pad if necessary
            if len(median_forecast) > self.forecast_horizon:
                median_forecast = median_forecast[:self.forecast_horizon]
            else:
                # Pad with the last value if too short
                last_val = median_forecast[-1] if len(median_forecast) > 0 else 0
                median_forecast = np.pad(median_forecast, (0, self.forecast_horizon - len(median_forecast)), mode='constant', constant_values=last_val)
        
        # Clean up tensors to free memory
        del inputs, forecasts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return median_forecast
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> Dict[str, float]:
        """
        Compute loss metrics for the model predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            y_train: Training data (used for MASE calculation)
            
        Returns:
            Dict[str, float]: Dictionary containing loss metrics
        """
        from benchmarking_pipeline.metrics.mae import MAE
        from benchmarking_pipeline.metrics.rmse import RMSE
        from benchmarking_pipeline.metrics.mase import MASE
        
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Initialize metric instances
        mae_calculator = MAE()
        rmse_calculator = RMSE()
        mase_calculator = MASE()
        
        # Calculate MAE
        mae_val = mae_calculator(y_true, y_pred)
        
        # Calculate RMSE
        rmse_val = rmse_calculator(y_true, y_pred)
        
        # Calculate MASE if training data is provided
        if y_train is not None:
            mase_val = mase_calculator(y_true, y_pred, y_train=y_train)
        else:
            mase_val = np.nan
        
        return {
            'mae': mae_val,
            'rmse': rmse_val,
            'mase': mase_val
        }