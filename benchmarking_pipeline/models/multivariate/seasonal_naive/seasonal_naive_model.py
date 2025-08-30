"""
Seasonal Naive model implementation.
TODO-(SHOULD work multivariate)
TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
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
        self._build_model()
        
    def _build_model(self):
        """
        Build the NaiveForecaster model instance from the configuration.
        """
        # Get hyperparameters from config
        model_params = self.config.get('model_params', {})
        sp = model_params.get('sp', 1)
        
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
        if not isinstance(y_context, pd.DataFrame):
            # works best with a proper index
            y_context = pd.DataFrame(y_context)
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
        forecast_horizon = len(y_target.shape[0])
        fh = np.arange(1, forecast_horizon + 1)
        predictions = self.model.predict(fh=fh)
        return predictions.values

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying sktime model.
        """
        if self.model:
            return self.model.get_params()
        # Return config params if model is not yet instantiated
        return self.config.get('model_params', {})

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
            
    def load(self, path: str) -> 'SeasonalNaiveModel':
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