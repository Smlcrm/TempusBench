"""
Prophet model implementation.
TODO-COULD work multivariate
"""

import os
import json
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from benchmarking_pipeline.models.base_model import BaseModel

class ProphetModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Prophet model with a given configuration.
        
        Args:
            config: Configuration dictionary for Prophet parameters.
                    e.g., {'model_params': {'seasonality_mode': 'multiplicative'}}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        self.model = Prophet(**self.config['model_params'])
        self.is_fitted = False

    @staticmethod
    def ensure_series_with_datetimeindex(y, start_date, freq):
        """
        Ensure the input series has a proper datetime index.
        
        Args:
            y: Input series (can be numpy array, pandas series, or already indexed)
            start_date: Start date to use for the index
            freq: Frequency from CSV data - MUST be provided
            
        Returns:
            pd.Series: Series with proper datetime index
            
        Raises:
            ValueError: If freq is None or empty
        """
        if freq is None or freq == "":
            raise ValueError("Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks.")
        
        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            return y
        return pd.Series(
            y.values if hasattr(y, 'values') else y,
            index=pd.date_range(start_date, periods=len(y), freq=freq)
        )

    def train(self, y_context, x_context=None, y_context_timestamps=None, **kwargs):
        if y_context_timestamps is None:
            raise ValueError("y_context_timestamps must be provided for Prophet training.")
        # Create a new Prophet model instance for each training
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        self.model = Prophet(**self.config['model_params'])
        # Use the provided timestamps to create a DatetimeIndex for y_context
        # Ensure 1D series indexed by provided timestamps
        y_vals = y_context.values if hasattr(y_context, 'values') else y_context
        if isinstance(y_vals, np.ndarray) and y_vals.ndim > 1:
            y_vals = y_vals.flatten()
        
        # Convert timestamps to datetime if they're not already
        if not isinstance(y_context_timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(y_context_timestamps[0], (int, np.integer)):
                # Convert from nanoseconds to datetime
                if y_context_timestamps[0] > 1e18:  # Likely nanoseconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='ns')
                elif y_context_timestamps[0] > 1e15:  # Likely microseconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='us')
                elif y_context_timestamps[0] > 1e12:  # Likely milliseconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='ms')
                else:  # Likely seconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='s')
            else:
                timestamps = pd.to_datetime(y_context_timestamps)
        else:
            timestamps = y_context_timestamps
            
        y_context = pd.Series(y_vals, index=timestamps)
        train_df = pd.DataFrame({'ds': y_context.index, 'y': y_context.values})
        
        if x_context is not None:
            if not isinstance(x_context, pd.DataFrame):
                x_context = pd.DataFrame(x_context, index=y_context.index)
            for col in x_context.columns:
                self.model.add_regressor(col)
            train_df = train_df.join(x_context)
        
        self.model.fit(train_df)
        # Store training statistics for fallback predictions
        self.training_mean = train_df['y'].mean()
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None,
        y_context_timestamps: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target_timestamps: Optional[Union[pd.Series, np.ndarray]] = None,
        freq: str = None
    ) -> np.ndarray:
        """
        Make predictions using the trained Prophet model.
        
        Args:
            y_context: Recent/past target values
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            y_context_timestamps: Timestamps for context data
            y_target: Target values for evaluation (optional)
            y_target_timestamps: Timestamps for target data (optional)
            freq: Frequency string from CSV data - MUST be provided
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            ValueError: If freq is None or if required data is missing
        """
        if freq is None or freq == "":
            raise ValueError("Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks.")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Determine forecast horizon
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon
        
        if y_target is not None:
            # If y_target is provided, use its length as the forecast horizon
            forecast_horizon = len(y_target)
            
            # If no timestamps provided, we need to infer them from the context
            if y_context_timestamps is not None and len(y_context_timestamps) > 0:
                last_timestamp = pd.to_datetime(y_context_timestamps[-1])
                # Create future timestamps starting from the next point after the last training data
                future_index = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), 
                                          periods=len(y_target), freq=freq)
            else:
                # For plotting purposes, we need to create timestamps that make sense
                # Use the last training timestamp if available, otherwise use a reasonable default
                if y_context_timestamps is not None and len(y_context_timestamps) > 0:
                    last_timestamp = pd.to_datetime(y_context_timestamps[-1])
                    future_index = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), 
                                              periods=len(y_target), freq=freq)
                else:
                    # If no context timestamps, we can't make meaningful predictions
                    raise ValueError("Cannot make predictions without timestamps. Frequency from CSV data is required.")
        else:
            # Default forecast horizon if neither y_target nor y_target_timestamps provided
            forecast_horizon = 300  # Default to 300 steps
            if y_context_timestamps is not None and len(y_context_timestamps) > 0:
                last_timestamp = pd.to_datetime(y_context_timestamps[-1])
                future_index = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), 
                                          periods=forecast_horizon, freq=freq)
            else:
                # If no context timestamps, we can't make meaningful predictions
                raise ValueError("Cannot make predictions without timestamps. Frequency from CSV data is required.")
        
        # Create future dataframe with the correct timestamps
        future_df = pd.DataFrame({'ds': future_index})
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        # Return the predicted values
        return forecast['yhat'].values

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration.
        """
        # Return Prophet model parameters
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        return self.config['model_params']

    def set_params(self, **params: Dict[str, Any]) -> 'ProphetModel':
        """
        Set model parameters. This will rebuild the Prophet model instance with the new parameters.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        
        # Filter out non-Prophet parameters
        valid_prophet_params = {}
        
        for key, value in params.items():
            if key in ['forecast_horizon', 'dataset']:
                # Skip these parameters - they're not for Prophet constructor
                continue
            elif key in ['seasonality_mode', 'changepoint_prior_scale', 'seasonality_prior_scale', 
                        'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
                        'holidays', 'holidays_prior_scale', 'changepoint_range']:
                # These are valid Prophet parameters
                valid_prophet_params[key] = value
            else:
                # Skip unknown parameters instead of raising error to allow for future Prophet versions
                print(f"[WARNING] Skipping unknown Prophet parameter: {key}")
                continue
        
        self.config['model_params'].update(valid_prophet_params)
        
        # Re-build the model with the valid parameters only
        self.model = Prophet(**self.config['model_params'])
        self.is_fitted = False
        return self

    def save(self, path: str) -> None:
        """
        Save the trained Prophet model to disk using its native JSON serialization.
        
        Args:
            path: Path to save the model file. The '.json' extension is recommended.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, 'w') as f:
            json.dump(model_to_json(self.model), f)
            
    def load(self, path: str) -> 'ProphetModel':
        """
        Load a trained Prophet model from a JSON file.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'r') as f:
            self.model = model_from_json(json.load(f))
        self.is_fitted = True
        return self