"""
Chronos foundation model implementation for time series forecasting.

This module provides a wrapper around the Amazon Chronos foundation model for time series
forecasting. Chronos is a large language model specifically designed for time series
forecasting tasks and can handle both univariate and multivariate data.

The model supports multiple sizes (tiny, mini, small, base, large) and can be configured
with different context lengths and sampling strategies.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Union, Tuple, List, Optional
from benchmarking_pipeline.models.foundation_model import FoundationModel
from chronos import ChronosPipeline as BaseChronosPipeline
from einops import rearrange


class ChronosModel(FoundationModel):
    """
    Chronos foundation model wrapper for time series forecasting.
    
    This class provides a unified interface for the Amazon Chronos model, which is
    a large language model specifically designed for time series forecasting.
    
    Attributes:
        model_size: Size of the Chronos model ('tiny', 'mini', 'small', 'base', 'large')
        context_length: Number of past time steps used as context
        num_samples: Number of predictive samples to generate
        pipeline: The underlying Chronos pipeline instance
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the Chronos model wrapper.

        Args:
            config: Configuration dictionary containing model parameters
                - model_size: str, size of the Chronos model (default: 'small')
                - context_length: int, number of past time steps for context (default: 8)
                - num_samples: int, number of predictive samples (default: 5)
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        if 'model_size' not in self.config:
            raise ValueError("model_size must be specified in config")
        if 'context_length' not in self.config:
            raise ValueError("context_length must be specified in config")
        if 'num_samples' not in self.config:
            raise ValueError("num_samples must be specified in config")
        
        self.model_size = self.config['model_size']
        self.context_length = self.config['context_length']
        self.num_samples = self.config['num_samples']
        
        # Validate model size
        valid_sizes = {'tiny', 'mini', 'small', 'base', 'large'}
        if self.model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}")
        
        # Initialize model state
        self.is_fitted = False
        
        # forecast_horizon is inherited from parent class (FoundationModel)
    
    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray]], 
              y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_start_date: Optional[str] = None
    ) -> 'ChronosModel':
        """
        Initialize the Chronos model (no training required for foundation models).
        
        Args:
            y_context: Past target values (not used for training, for compatibility)
            y_target: Future target values (not used for training, for compatibility)
            y_start_date: Start date for y_context (not used)
            
        Returns:
            self: The model instance
            
        Note:
            Chronos is a pre-trained foundation model that doesn't require training.
            This method just marks the model as ready for inference.
        """
        # For foundation models, we don't need to load the model here
        # It will be loaded fresh for each prediction (like it was in the working version)
        self.is_fitted = True
        return self
    
    def predict(self,
                y_context: Optional[Union[pd.Series, np.ndarray]] = None,
                y_target: Union[pd.Series, np.ndarray] = None,
                y_context_timestamps: Optional[Union[pd.DatetimeIndex, List]] = None,
                y_target_timestamps: Optional[Union[pd.DatetimeIndex, List]] = None,
                forecast_horizon: Optional[int] = None,
                **kwargs
    ) -> np.ndarray:
        """
        Make predictions using the trained Chronos model.
        
        Args:
            y_context: Recent target values for context
            y_target: Target values to predict (used to determine forecast length)
            y_context_timestamps: Timestamps for context data
            y_target_timestamps: Timestamps for target data
            forecast_horizon: Number of steps to forecast (overrides y_target length if provided)
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            ValueError: If model is not fitted or required data is missing
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        if y_context is None:
            raise ValueError("y_context must be provided for predictions.")
        
        if y_context_timestamps is None:
            raise ValueError("y_context_timestamps must be provided for predictions.")
        
        # Determine forecast horizon from y_target
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")
        
        # Accept Series/DataFrame/ndarray
        if isinstance(y_target, pd.Series):
            prediction_length = len(y_target)
        elif isinstance(y_target, pd.DataFrame):
            # rows=time, columns=targets
            prediction_length = y_target.shape[0]
        else:
            y_target_arr = np.asarray(y_target)
            if y_target_arr.ndim == 1:
                prediction_length = y_target_arr.shape[0]
            elif y_target_arr.ndim == 2:
                prediction_length = y_target_arr.shape[0]
            else:
                raise ValueError("Unsupported y_target dimensionality for prediction length inference.")
        
        # Simplified univariate vs multivariate handling
        if len(y_context.shape) == 1 or (y_context.ndim == 2 and y_context.shape[1] == 1):
            # Univariate: shape (N,) or (N,1)
            columns = ['1']
            y_context_reshaped = y_context.reshape(-1, 1) if len(y_context.shape) == 1 else y_context
        else:
            # Multivariate: shape (N, num_targets)
            columns = list(range(y_context.shape[1]))
            y_context_reshaped = y_context
        
        # Create DataFrame with proper timestamps
        df = pd.DataFrame(y_context_reshaped, index=y_context_timestamps, columns=columns)
        
        # DEBUG: Print DataFrame info
        print(f"[CHRONOS DEBUG] DataFrame shape: {df.shape}")
        print(f"[CHRONOS DEBUG] DataFrame columns: {list(df.columns)}")
        print(f"[CHRONOS DEBUG] DataFrame first few values: {df.iloc[:3, :].values}")
        
        # Use the working approach: load model fresh and convert to proper format
        results = self._sub_predict(df, prediction_length)
        
        # DEBUG: Print what results actually contains
        print(f"[CHRONOS DEBUG] Results keys: {list(results.keys())}")
        print(f"[CHRONOS DEBUG] Results type: {type(results)}")
        print(f"[CHRONOS DEBUG] Results content: {results}")
        
        # Process results based on data dimensionality
        if len(list(results.keys())) == 1:
            # Univariate result - always expect '1' as per working commit 434d3b0e
            series_vals = np.array(results["1"])  # shape (pred_len,)
            
            if series_vals.ndim > 1:
                series_vals = series_vals.squeeze()
            if series_vals.shape[0] > prediction_length:
                series_vals = series_vals[:prediction_length]
            return series_vals
        else:
            # Multivariate result
            multivariate_values = []
            for key in results.keys():
                vals = np.array(results[key])
                if vals.ndim > 1:
                    vals = vals.squeeze()
                if vals.shape[0] > prediction_length:
                    vals = vals[:prediction_length]
                multivariate_values.append(vals)
            preds = np.array(multivariate_values)  # shape (num_targets, pred_len)
            # Ensure exact horizon length
            if preds.shape[1] > prediction_length:
                preds = preds[:, :prediction_length]
            return preds
    
    def _sub_predict(self, df: pd.DataFrame, prediction_length: int) -> Dict[str, List[float]]:
        """
        Generates forecasts for future time steps based on the most recent data.
        
        This method uses the last `context_length` data points from each time series
        in the DataFrame to predict the next `prediction_length` steps.
        
        Args:
            df: DataFrame containing time series data with timestamps as index
            prediction_length: Number of future time steps to predict
            
        Returns:
            Dict[str, List[float]]: A dictionary where keys are time series names (column headers)
                                    and values are the list of forecasted points.
        """
        # Load the Chronos model fresh for each prediction (like the working version)
        hf_model_name = f"amazon/chronos-t5-{self.model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        pipeline = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Chronos model loaded successfully!")
        
        # Create one context window for each time series
        all_contexts = []
        for series_name in df.columns:
            series_data = df[series_name].values
            
            # Intelligent context selection: use more recent data for better predictions
            if len(series_data) >= self.context_length:
                # Use the most recent context_length data points
                context_data = series_data[-self.context_length:]
            else:
                # If not enough data, pad with the last available value
                context_data = np.full(self.context_length, series_data[-1])
                context_data[-len(series_data):] = series_data
            
            # Ensure data is properly formatted for Chronos
            context_data = np.asarray(context_data, dtype=np.float32)
            
            # Handle any NaN values
            if np.any(np.isnan(context_data)):
                context_data = np.nan_to_num(context_data, nan=0.0)
            
            all_contexts.append(torch.tensor(context_data, dtype=torch.float32))
        
        # Generate forecasts
        all_forecasts = pipeline.predict(
            context=all_contexts,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
        )
        
        # Process results
        results = {}
        for i, series_name in enumerate(df.columns):
            # For each series, aggregate the prediction samples intelligently
            forecasts = all_forecasts[i]  # shape: (num_samples, prediction_length)
            
            # Convert PyTorch tensor to numpy array
            if hasattr(forecasts, 'cpu'):
                forecasts = forecasts.cpu().numpy()
            elif hasattr(forecasts, 'numpy'):
                forecasts = forecasts.numpy()
            else:
                forecasts = np.array(forecasts)
            
            if self.num_samples > 1:
                # Use weighted average: give more weight to more recent predictions
                weights = np.linspace(0.5, 1.0, self.num_samples)
                weights = weights / np.sum(weights)
                
                # Weighted average across samples
                weighted_forecast = np.average(forecasts, axis=0, weights=weights)
                
                # Also compute median as fallback
                median_forecast = np.median(forecasts, axis=0)
                
                # Use the better of the two (lower variance)
                # Calculate variance manually to avoid numpy version issues
                forecast_variance = np.mean((forecasts - np.mean(forecasts, axis=0))**2, axis=0)
                if np.mean(forecast_variance) < 0.1:  # Low variance = use weighted avg
                    final_forecast = weighted_forecast
                else:  # High variance = use median (more robust)
                    final_forecast = median_forecast
            else:
                final_forecast = forecasts[0]  # Single sample
            
            results[series_name] = final_forecast.tolist()
            
        # DEBUG: Print what we're returning
        print(f"[CHRONOS DEBUG] _sub_predict returning keys: {list(results.keys())}")
        print(f"[CHRONOS DEBUG] _sub_predict returning content: {results}")
            
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'model_size': self.model_size,
            'context_length': self.context_length,
            'num_samples': self.num_samples,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params: Dict[str, Any]) -> 'ChronosModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        # Track if model parameters that affect model loading are changed
        model_params_changed = False
        
        for key, value in params.items():
            if hasattr(self, key):
                # Check if this is a model parameter that requires reloading
                if key in ['model_size'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed and self.is_fitted:
            self.is_fitted = False
            
        return self
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Chronos model's properties.
        
        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            'model_type': 'Chronos',
            'model_size': self.model_size,
            'context_length': self.context_length,
            'num_samples': self.num_samples,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }