"""
Exponential Smoothing model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from benchmarking_pipeline.models.base_model import BaseModel


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Exponential Smoothing model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    e.g., {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12, ...}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        if 'trend' not in self.config:
            raise ValueError("trend must be specified in config")
        if 'seasonal' not in self.config:
            raise ValueError("seasonal must be specified in config")
        if 'seasonal_periods' not in self.config:
            raise ValueError("seasonal_periods must be specified in config")
        if 'damped_trend' not in self.config:
            raise ValueError("damped_trend must be specified in config")
        # forecast_horizon is inherited from parent class (FoundationModel)
        
        # Get training loss from config
        if 'training_loss' not in self.config:
            raise ValueError("training_loss must be specified in config")
        self.training_loss = self.config['training_loss']
        
        def _cast_param(key, value):
            if key == 'seasonal_periods':
                return int(value) if value is not None else None
            if key == 'damped_trend':
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            if key == 'forecast_horizon':
                return int(value) if value is not None else 1
            if key in ['trend', 'seasonal']:
                if isinstance(value, str) and value.lower() == 'none':
                    return None
                return value
            return value
            
        self.trend = _cast_param('trend', self.config['trend'])
        self.seasonal = _cast_param('seasonal', self.config['seasonal'])
        self.seasonal_periods = _cast_param('seasonal_periods', self.config['seasonal_periods'])
        self.damped_trend = _cast_param('damped_trend', self.config['damped_trend'])
        self.forecast_horizon = _cast_param('forecast_horizon', self.config['forecast_horizon'])
        
        self.model_ = None
        self.is_fitted = False
        
        # Adapt for training_loss refactoring


    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None,
              x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'ExponentialSmoothingModel':
        
        print(f"[ExponentialSmoothing train] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        
        # Ensure correct types for model parameters
        trend = self.trend
        seasonal = self.seasonal
        if isinstance(trend, str) and trend.lower() == 'none':
            trend = None
        if isinstance(seasonal, str) and seasonal.lower() == 'none':
            seasonal = None
        seasonal_periods = int(self.seasonal_periods) if self.seasonal_periods is not None else None
        
        damped_trend = bool(self.damped_trend)
        if isinstance(damped_trend, str):
            damped_trend = damped_trend.lower() == 'true'
        # Only allow damped_trend if trend is not None
        if trend is None:
            damped_trend = None
            
        # Handle input data - ensure we have the right format
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        elif isinstance(y_context, pd.DataFrame):
            endog = y_context.values.flatten()
        else:
            endog = y_context
            
        # Ensure endog is 1D
        if endog.ndim > 1:
            endog = endog.flatten()
            
        print(f"[ExponentialSmoothing train] endog shape: {endog.shape}, first 5 values: {endog[:5]}")
        print(f"[ExponentialSmoothing train] parameters: trend={trend}, seasonal={seasonal}, seasonal_periods={seasonal_periods}, damped_trend={damped_trend}")
        
        try:
            self.model_ = ExponentialSmoothing(
                endog,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            ).fit()
            self.is_fitted = True
            print(f"[ExponentialSmoothing train] Model fitted successfully")
        except Exception as e:
            print(f"[ExponentialSmoothing train] Error fitting model: {e}")
            raise
            
        return self

    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None,
                x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, **kwargs) -> np.ndarray:
        
        print(f"[ExponentialSmoothing predict] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        print(f"[ExponentialSmoothing predict] y_target type: {type(y_target)}, shape: {getattr(y_target, 'shape', 'N/A')}")
        
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        
        forecast_steps = len(y_target)
        print(f"[ExponentialSmoothing predict] Forecasting {forecast_steps} steps")
        
        try:
            forecast = self.model_.forecast(steps=forecast_steps)
            print(f"[ExponentialSmoothing predict] forecast type: {type(forecast)}, shape: {getattr(forecast, 'shape', 'N/A')}")
            print(f"[ExponentialSmoothing predict] forecast first 5 values: {forecast[:5] if hasattr(forecast, '__getitem__') else forecast}")
            
            # Convert to numpy array and reshape
            if isinstance(forecast, pd.Series):
                forecast_array = forecast.values
            else:
                forecast_array = np.asarray(forecast)
                
            result = forecast_array.reshape(1, -1)
            print(f"[ExponentialSmoothing predict] result shape: {result.shape}")
            return result
            
        except Exception as e:
            print(f"[ExponentialSmoothing predict] Error during forecast: {e}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration and instance attributes.
        """
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon
        }

    def set_params(self, **params: Dict[str, Any]) -> 'ExponentialSmoothingModel':
        """
        Set model parameters by updating the configuration and instance attributes.
        The model will be rebuilt with these new parameters on the next .train() call.
        """
        def _cast_param(key, value):
            if key == 'seasonal_periods':
                return int(value) if value is not None else None
            if key == 'damped_trend':
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            if key == 'forecast_horizon':
                return int(value) if value is not None else 1
            if key in ['trend', 'seasonal']:
                if isinstance(value, str) and value.lower() == 'none':
                    return None
                return value
            return value
            
        for key, value in params.items():
            casted_value = _cast_param(key, value)
            setattr(self, key, casted_value)
            self.config[key] = casted_value
        self.is_fitted = False
        self.model_ = None
        return self

    def save(self, path: str) -> None:
        """
        Save the trained model to a pickle file.
        """
        model_data = {
            'model_': self.model_,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> 'ExponentialSmoothingModel':
        """
        Load a trained model from a pickle file.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_ = model_data['model_']
        self.is_fitted = model_data['is_fitted']
        self.config = model_data['config']
        self.trend = model_data['trend']
        self.seasonal = model_data['seasonal']
        self.seasonal_periods = model_data['seasonal_periods']
        self.damped_trend = model_data['damped_trend']
        if 'training_loss' not in model_data:
            raise ValueError("training_loss must be specified in model_data")
        self.training_loss = model_data['training_loss']
        self.forecast_horizon = model_data['forecast_horizon']
        return self