"""
XGBoost model implementation for time series forecasting.
"""
import os
import pickle
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class XgboostModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize XGBoost model with a given configuration.
        
        Args:
            config: Configuration dictionary for XGBoostRegressor parameters.
                    e.g., {'n_estimators': 100, 'learning_rate': 0.1, 'lookback_window': 10}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        
        # Extract model-specific configuration
        model_config = self._extract_model_config(self.config)
        
        if 'lookback_window' not in model_config:
            raise ValueError("lookback_window must be specified in config")
        
        self.lookback_window = model_config['lookback_window']
        # forecast_horizon is inherited from parent class (BaseModel)
        self._build_model()
        
    def _extract_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model-specific configuration from nested config structure.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Dict[str, Any]: Model-specific configuration
        """
        # If config has a 'model' section, look for the specific model type
        if 'model' in config:
            model_section = config['model']
            # Find the first model configuration (e.g., 'arima', 'lstm', etc.)
            for model_name, model_config in model_section.items():
                if isinstance(model_config, dict):
                    return model_config
        
        # If no nested structure, return the config as-is
        return config
        
    def _build_model(self):
        """
        Build the XGBRegressor model instance from the configuration.
        """
        # Get hyperparameters from config.
        model_config = self._extract_model_config(self.config)
        
        # Extract XGBoost-specific parameters
        xgb_params = {}
        for key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'random_state', 'n_jobs']:
            if key in model_config:
                xgb_params[key] = model_config[key]
        
        # Ensure random_state for reproducibility if not provided
        if 'random_state' not in xgb_params:
            xgb_params['random_state'] = 42
            
        self.model = XGBRegressor(**xgb_params)
        self.is_fitted = False

    def _create_features(self, y_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create simple time series features for XGBoost.
        
        Args:
            y_series: Target time series
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        n_samples = len(y_series) - self.lookback_window
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.lookback_window + 1} samples.")
        
        features = []
        targets = []
        
        for i in range(n_samples):
            # Create lag features from historical data only
            lag_features = y_series[i:i + self.lookback_window]
            
            # Create simple rolling statistics
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)
            
            # Create trend feature (simple difference)
            trend = lag_features[-1] - lag_features[0]
            
            # Combine features
            sample_features = list(lag_features) + [rolling_mean, rolling_std, trend]
            
            features.append(sample_features)
            # Target is the next value
            targets.append(y_series[i + self.lookback_window])
        
        return np.array(features), np.array(targets)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, y_start_date: pd.Timestamp = None, **kwargs) -> 'XgboostModel':
        """
        Train the XGBoost model on given data.
        
        TECHNIQUE: Simple lag-based regression
        - Creates lag features from historical target values
        - Adds basic rolling statistics (mean, std)
        - Includes simple trend feature
        - Uses XGBoost for regression on engineered features
        
        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values (optional, for validation)
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
        
        # Convert inputs to numpy arrays
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_data = y_context.values
        else:
            y_data = y_context
            
        # Ensure y_data is 1D
        if y_data.ndim > 1:
            y_data = y_data.flatten()
        
        # Create features and targets
        X, y = self._create_features(y_data)
        
        print(f"Training XGBoost model with {X.shape[0]} samples and {X.shape[1]} features...")
        
        # Train model for single-step prediction
        self.model.fit(X, y)
        
        self.is_fitted = True
        print("Training complete.")
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        
        TECHNIQUE: Iterative single-step prediction
        - Uses last lookback_window values to create features
        - Predicts one step ahead
        - Updates the window with the prediction
        - Repeats until forecast_steps are reached
        
        Args:
            y_context: Past target values (time series) - used for prediction
            y_target: Future target values - used to determine prediction length
        
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_steps)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        
        forecast_steps = len(y_target)
        
        # Convert inputs to numpy arrays
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_data = y_context.values
        else:
            y_data = y_context
        if y_data.ndim > 1:
            y_data = y_data.flatten()
        
        # Use the last lookback_window values to start
        window = list(y_data[-self.lookback_window:])
        predictions = []
        
        # Make iterative predictions
        for step in range(forecast_steps):
            # Create features for the current window
            lag_features = np.array(window)
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)
            trend = lag_features[-1] - lag_features[0]
            
            sample_features = list(lag_features) + [rolling_mean, rolling_std, trend]
            X_pred = np.array(sample_features).reshape(1, -1)
            
            # Make single-step prediction
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Update window for next prediction (remove oldest, add newest)
            window = window[1:] + [pred]
        
        return np.array(predictions).reshape(1, -1)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying xgboost model.
        """
        if self.model:
            return self.model.get_params()
        model_config = self._extract_model_config(self.config)
        return model_config

    def set_params(self, **params: Dict[str, Any]) -> 'XGBoostModel':
        """
        Set model parameters. This will rebuild the model instance.
        """
        model_config = self._extract_model_config(self.config)
        model_config.update(params)
        
        if self.model:
            self.model.set_params(**params)
        
        return self

    def save(self, path: str) -> None:
        """
        Save the trained xgboost model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'model': self.model
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'XGBoostModel':
        """
        Load a trained xgboost model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.lookback_window = model_state['lookback_window']
        self.forecast_horizon = model_state['forecast_horizon']
        self.model = model_state['model']
        
        return self