"""
Theta model implementation.

This model implements the Theta method for time series forecasting using sktime's ThetaForecaster.
"""

import os
import pickle
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sktime.forecasting.theta import ThetaForecaster
from benchmarking_pipeline.models.base_model import BaseModel

class ThetaModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the Theta model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    Example Format:
                    {'sp': 12, 'forecast_horizon': 10} - for monthly data with yearly seasonality.   
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        
        # Extract model-specific config
        model_config = self._extract_model_config(self.config)
        if 'sp' not in model_config:
            raise ValueError("sp must be specified in config")
        self.sp = model_config['sp']
        self.model = None
        self.is_fitted = False
        
    def _build_model(self):
        """
        Build the ThetaForecaster model instance from the configuration.
        """
        # Only pass supported arguments to ThetaForecaster
        # Note: For normalized data with values near zero, multiplicative seasonality
        # can fail. In such cases, we disable deseasonalization.
        model_params = {"sp": self.sp}
        
        # Try with default settings first, fallback to no deseasonalization if needed
        try:
            self.model = ThetaForecaster(**model_params)
        except:
            # If model creation fails, disable deseasonalization
            model_params["deseasonalize"] = False
            self.model = ThetaForecaster(**model_params)
        
        self.is_fitted = False

    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              y_target: Union[pd.Series, np.ndarray] = None, 
              **kwargs
    ) -> 'ThetaModel':
        """
        Train the Theta model on given data. For this model, "training" involves
        decomposing the series and fitting exponential smoothing.
        
        Args:
            y_context: Historical target time series values (pd.Series or np.ndarray).
            y_target: Target values for validation (ignored during training).
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
            
        if not isinstance(y_context, pd.Series):
            # Handle both 1D and 2D input data
            if isinstance(y_context, np.ndarray) and y_context.ndim == 2:
                # Extract the single column from 2D array
                y_context = y_context[:, 0]
            elif hasattr(y_context, 'values') and hasattr(y_context.values, 'ndim') and y_context.values.ndim == 2:
                # Handle pandas DataFrame or similar
                y_context = y_context.values[:, 0]
            # sktime works best with Pandas Series with a proper index
            y_context = pd.Series(y_context)
            
        # For normalized data with values near zero, add small epsilon to avoid 
        # multiplicative seasonality issues while preserving the relative patterns
        if self.sp > 1 and np.any(y_context <= 0.001):
            epsilon = 1e-6
            y_context = y_context + epsilon
            print(f"[INFO] Added epsilon ({epsilon}) to avoid multiplicative seasonality issues with normalized data")
            
        # Theta is a univariate method, so we only use y_context
        try:
            self.model.fit(y=y_context)  # No exogenous variables
            self.is_fitted = True
        except ValueError as e:
            if "Multiplicative seasonality is not appropriate" in str(e):
                # For normalized data, try different strategies to maintain forecasting capability
                print(f"[WARNING] Multiplicative seasonality failed with sp={self.sp}")
                
                # Strategy 1: Try with no seasonality (sp=1) but keep deseasonalization enabled
                if self.sp > 1:
                    print(f"[INFO] Trying fallback with sp=1 (no seasonality)")
                    model_params = {"sp": 1, "deseasonalize": True}
                    self.model = ThetaForecaster(**model_params)
                    try:
                        self.model.fit(y=y_context)
                        self.is_fitted = True
                        return self
                    except ValueError:
                        pass  # Continue to next strategy
                
                # Strategy 2: Disable deseasonalization as last resort
                print(f"[INFO] Using fallback: disabling deseasonalization")
                model_params = {"sp": self.sp, "deseasonalize": False}
                self.model = ThetaForecaster(**model_params)
                self.model.fit(y=y_context)
                self.is_fitted = True
            else:
                raise e
        return self
        
    def predict(self, 
                y_context: Union[pd.Series, np.ndarray], 
                y_target: Union[pd.Series, np.ndarray] = None,
                forecast_horizon: Optional[int] = None,
                **kwargs) -> np.ndarray:
        """
        Make predictions using the trained Theta model.
        
        Args:
            y_context: Recent/past target values (ignored by Theta - uses training data).
            y_target: Used to determine the number of steps to forecast (ignored by Theta).
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided).
            **kwargs: Additional keyword arguments for compatibility.
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon).
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Determine forecast horizon from y_target
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")
        
        forecast_steps = len(y_target)
        fh = np.arange(1, forecast_steps + 1)
        
        # The sktime predict method uses the forecasting horizon (fh)
        predictions = self.model.predict(fh=fh)
        
        return predictions.values

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration.
        """
        return {
            'sp': self.sp,
            'forecast_horizon': self.forecast_horizon
        }

    def set_params(self, **params: Dict[str, Any]) -> 'ThetaModel':
        """
        Set model parameters. This will rebuild the sktime model instance.
        """
        model_params_changed = False
        for key, value in params.items():
            if key == 'dataset':
                # Skip dataset parameter - it's not a Theta model parameter
                continue
            elif hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['sp'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed and self.is_fitted:
            self.model = None
            self.is_fitted = False
            
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
            
        # Save both the model and the configuration
        model_state = {
            'model': self.model,
            'config': self.config,
            'sp': self.sp,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }
            
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'ThetaModel':
        """
        Load a trained sktime model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Restore model state
        self.model = model_state['model']
        self.config = model_state.get('config', self.config)
        self.sp = model_state.get('sp', self.sp)
        self.forecast_horizon = model_state.get('forecast_horizon', self.forecast_horizon)
        self.is_fitted = model_state.get('is_fitted', False)
        
        return self