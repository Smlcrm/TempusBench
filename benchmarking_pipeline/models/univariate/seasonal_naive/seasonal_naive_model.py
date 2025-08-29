"""
Seasonal Naive model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster
from benchmarking_pipeline.models.base_model import BaseModel

class SeasonalNaiveModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Seasonal Naive model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    e.g., {'model_params': {'sp': 7}} for weekly seasonality in daily data.
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        model_params = self.config['model_params']
        if 'sp' not in model_params:
            raise ValueError("sp must be specified in model_params")
        sp = model_params['sp']
        self.sp = sp
        self.is_fitted = False

    def _build_model(self):
        """
        Build the NaiveForecaster model instance from the configuration.
        """
        # Get hyperparameters from config
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        model_params = self.config['model_params']
        if 'sp' not in model_params:
            raise ValueError("sp must be specified in model_params")
        sp = model_params['sp']
        
        self.model = NaiveForecaster(strategy="last", sp=sp)
        self.is_fitted = False

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'SeasonalNaiveModel':
        """
        Train the Seasonal Naive model on given data. For this model, "training"
        simply means storing the historical data for future lookups.
        
        Args:
            y_context: Past target values (pd.Series or np.ndarray).
            y_target, x_context, x_target: Not used by this model, but included for compatibility.
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
        
        # Handle both 1D and 2D input data
        if isinstance(y_context, np.ndarray) and y_context.ndim == 2:
            # Extract the single column from 2D array
            y_context = y_context[:, 0]
        elif hasattr(y_context, 'values') and hasattr(y_context.values, 'ndim') and y_context.values.ndim == 2:
            # Handle pandas DataFrame or similar
            y_context = y_context.values[:, 0]
        
        if not isinstance(y_context, pd.Series):
            # works best with a proper index
            y_context = pd.Series(y_context)
        
        self.model.fit(y=y_context, X=None)
        self.is_fitted = True
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None, **kwargs):
        """
        Make predictions using the trained Seasonal Naive model.
        
        Args:
            y_context: Context time series values (pd.Series or np.ndarray).
            y_target: Used to determine the number of steps to forecast.
            x_context, x_target: Not used by this model, but included for compatibility.
            **kwargs: Additional keyword arguments.
        
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon).
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        forecast_horizon = len(y_target)
        fh = np.arange(1, forecast_horizon + 1)
        predictions = self.model.predict(fh=fh)
        return predictions.values.reshape(1, -1)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying sktime model.
        """
        if self.model:
            return self.model.get_params()
        # Return config params if model is not yet instantiated
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        return self.config['model_params']

    def set_params(self, **params: Dict[str, Any]) -> 'SeasonalNaiveModel':
        """
        Set model parameters. This will rebuild the sktime model instance.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(params)
        
        # Re-build the model with the new parameters
        self._build_model()
        return self

    def save(self, path: str) -> None:
        """
        Save the trained sktime model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
        
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str) -> None:
        """
        Load a trained sktime model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self