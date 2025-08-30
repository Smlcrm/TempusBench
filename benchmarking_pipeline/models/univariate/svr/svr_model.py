"""
SVR model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import io


class SvrModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None, logger=None):
        """
        Initialize Support Vector Regression (SVR) model with a given configuration.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor.
        
        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
            logger: Logger instance for TensorBoard logging
        """
        super().__init__(config, config_file)
        self.scaler = StandardScaler() # SVR is sensitive to feature scaling
        
        # Debug: Print what config we received
        print(f"[DEBUG][SVR] Received config: {config}")
        print(f"[DEBUG][SVR] Config keys: {list(config.keys()) if config else 'None'}")
        if config and 'log_dir' in config:
            print(f"[DEBUG][SVR] log_dir found: {config['log_dir']}")
        else:
            print(f"[DEBUG][SVR] log_dir not found in config")
        
        # Set up logging - use provided logger or create our own
        if logger is not None:
            self.logger = logger
            print(f"[INFO] SVR model using provided logger")
        else:
            # Try to create our own logger if config has logging info
            try:
                from benchmarking_pipeline.pipeline.logger import Logger
                log_config = {
                    'log_dir': config.get('log_dir', 'logs/tensorboard') if config else 'logs/tensorboard',
                    'run_name': 'svr_model',
                    'verbose': True
                }
                print(f"[DEBUG][SVR] Creating logger with config: {log_config}")
                self.logger = Logger(log_config)
                print(f"[INFO] SVR model created its own Logger for TensorBoard logging")
            except Exception as e:
                print(f"[WARNING] Failed to create Logger for SVR model: {e}")
                self.logger = None
        
        # Extract model-level parameters (not SVR-specific)
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        
        # Extract SVR-specific parameters for the underlying SVR model
        svr_params = {}
        svr_param_names = ['kernel', 'C', 'epsilon', 'gamma']
        for param in svr_param_names:
            if param in self.config:
                svr_params[param] = self.config[param]
        
        # Store SVR params for later use
        self.svr_params = svr_params
        
        self._build_model()

    def _build_model(self):
        """
        Build the SVR model instance from the configuration using MultiOutputRegressor for direct multi-output forecasting.
        """
        base_svr = SVR(**self.svr_params)
        self.model = MultiOutputRegressor(base_svr)
        self.is_fitted = False

    def _create_features_targets(self, y_series, lookback_window, forecast_horizon):
        """
        Create features and multi-step targets for direct multi-output forecasting.
        Each sample uses the previous lookback_window values as features and the next forecast_horizon values as targets.
        """
        X, y = [], []
        for i in range(len(y_series) - lookback_window - forecast_horizon + 1):
            X.append(y_series[i:i+lookback_window])
            y.append(y_series[i+lookback_window:i+lookback_window+forecast_horizon])
        return np.array(X), np.array(y)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'SvrModel':
        """
        Train the SVR model for direct multi-output forecasting using MultiOutputRegressor.
        """
        if self.model is None:
            self._build_model()
        if y_context is None:
            raise ValueError("y_context (target series) must be provided for training.")
        # Combine context and target for full training series if y_target is provided
        # Ensure both arrays are 1D before concatenation
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values.flatten()
        elif isinstance(y_context, np.ndarray) and y_context.ndim > 1:
            y_context = y_context.flatten()
            
        if y_target is not None:
            if isinstance(y_target, (pd.Series, pd.DataFrame)):
                y_target = y_target.values.flatten()
            elif isinstance(y_target, np.ndarray) and y_target.ndim > 1:
                y_target = y_target.flatten()
            y_series = np.concatenate([y_context, y_target])
        else:
            y_series = y_context
        X, y = self._create_features_targets(y_series, self.lookback_window, self.forecast_horizon)
        print(f"[DEBUG][SVR] Training X shape: {X.shape}, y shape: {y.shape}")
        
        # Log training progress to TensorBoard
        self._log_training_progress(X, y)
        
        # Scale features (time index is not used; features are lagged values)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self

    def rolling_predict(self, y_context, y_target, **kwargs):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """
        # Debug: Print initial context info
        print(f"[DEBUG][SVR] Initial y_context: {y_context}")
        print(f"[DEBUG][SVR] Initial y_context length: {len(y_context)}, lookback_window: {self.lookback_window}")
        if len(y_context) < self.lookback_window:
            raise ValueError(f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}")

        preds = []
        context = np.array(y_context).flatten().tolist()
        total_steps = len(y_target)
        steps_done = 0
        
        while steps_done < total_steps:
            steps = min(self.forecast_horizon, total_steps - steps_done)
            X_pred = np.array(context[-self.lookback_window:]).reshape(1, -1)              
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = self.model.predict(X_pred_scaled).flatten()
            
            print(f"[DEBUG][SVR] Rolling predict X_pred shape: {X_pred.shape}, pred shape: {pred.shape}")
            # Only take as many steps as needed
            pred = pred[:steps]
            preds.extend(pred)
            context.extend(pred)
            steps_done += steps
        
        # Log predictions to TensorBoard if logger is available
        if self.logger is not None and len(preds) > 0:
            try:
                # Convert y_target to array if it's not already
                y_target_array = np.array(y_target).flatten()
                preds_array = np.array(preds)
                
                # Log prediction plot
                self._log_predictions(y_target_array, preds_array, step=0, tag="rolling_predictions")
                
                # Log prediction metrics
                if len(y_target_array) == len(preds_array):
                    mae = np.mean(np.abs(y_target_array - preds_array))
                    rmse = np.sqrt(np.mean((y_target_array - preds_array) ** 2))
                    self.logger.log_metrics({
                        'predictions/mae': mae,
                        'predictions/rmse': rmse,
                        'predictions/num_predictions': len(preds_array)
                    }, step=0, model_name='svr')
                    
            except Exception as e:
                print(f"[WARNING] Failed to log predictions: {e}")
        
        return np.array(preds)

    def predict(self, y_context=None, y_target=None, **kwargs) -> np.ndarray:
        """
        Make direct multi-output predictions using the trained SVR model.
        If y_target is provided, use rolling_predict to predict the entire length autoregressively.
        Returns a vector of length forecast_horizon or full test length.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None:
            raise ValueError("y_context must be provided to determine prediction context.")
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")
        
        # Use rolling prediction to cover the full test set
        preds = self.rolling_predict(y_context, y_target)
        print(f"[DEBUG][SVR] Final rolling preds shape: {preds.shape}")
        return preds

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying scikit-learn model.
        """
        if self.model:
            return self.model.get_params()
        # Return config params if model is not yet instantiated
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'SvrModel':
        """
        Set model parameters. This will rebuild the model instance with the new parameters.
        Handles MultiOutputRegressor by prefixing SVR params with 'estimator__'.
        Model-level params (lookback_window, forecast_horizon) are set as attributes.
        """
        # Handle model-level params
        model_level_keys = ['lookback_window', 'forecast_horizon']
        for k in model_level_keys:
            if k in params:
                setattr(self, k, params[k])
        
        # Separate SVR parameters from model-level parameters
        svr_params = {}
        for k, v in params.items():
            if k in ['kernel', 'C', 'epsilon', 'gamma']:
                svr_params[k] = v
        
        # Convert parameter types for scikit-learn compatibility
        converted_params = {}
        for k, v in svr_params.items():
            if k == 'gamma':
                # Handle gamma parameter: convert string numbers to float
                if isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                    converted_params[k] = float(v)
                else:
                    converted_params[k] = v
            elif k == 'C' and isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                converted_params[k] = float(v)
            elif k == 'epsilon' and isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                converted_params[k] = float(v)
            else:
                converted_params[k] = v
        
        # Update stored SVR params
        self.svr_params.update(converted_params)
        
        # Prepare params for underlying SVR
        svr_param_names = SVR().get_params().keys()
        mo_params = {}
        for k, v in converted_params.items():
            if k in svr_param_names:
                mo_params[f'estimator__{k}'] = v
        
        if hasattr(self, 'model') and isinstance(self.model, MultiOutputRegressor):
            self.model.set_params(**mo_params)
        elif hasattr(self, 'model') and self.model is not None:
            # fallback for non-multioutput
            self.model.set_params(**{k: v for k, v in converted_params.items() if k in svr_param_names})
        
        # Update config as well
        for k in model_level_keys:
            if k in params:
                self.config[k] = params[k]
        for k, v in converted_params.items():
            self.config[k] = v
            
        return self
        
    def save(self, path: str) -> None:
        """
        Save the trained SVR model and its scaler to disk using pickle.
        
        Args:
            path: Path to save the model objects.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # We need to save both the model and the scaler for consistent predictions
        model_and_scaler = {'model': self.model, 'scaler': self.scaler}
        
        with open(path, 'wb') as f:
            pickle.dump(model_and_scaler, f)
            
    def load(self, path: str) -> 'SvrModel':
        """
        Load a trained SVR model and its scaler from disk.
        
        Args:
            path: Path to load the model objects from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_and_scaler = pickle.load(f)
            self.model = model_and_scaler['model']
            self.scaler = model_and_scaler['scaler']
            
        self.is_fitted = True
        return self

    def _log_training_progress(self, X_train, y_train, X_val=None, y_val=None):
        """Log training progress to TensorBoard if logger is available."""
        if self.logger is None:
            return
            
        try:
            # Log training data shapes
            self.logger.log_metrics({
                'training/X_shape': X_train.shape[0],
                'training/y_shape': y_train.shape[0],
                'training/lookback_window': self.lookback_window,
                'training/forecast_horizon': self.forecast_horizon
            }, step=0, model_name='svr')
            
            # Log SVR parameters as metrics (for numeric values) or use log_info for text
            for param_name, param_value in self.svr_params.items():
                if isinstance(param_value, (int, float)):
                    self.logger.log_metrics({f'hyperparameters/{param_name}': param_value}, step=0, model_name='svr')
                else:
                    self.logger.log_info(f'SVR hyperparameter {param_name}: {param_value}')
                    
            print(f"[INFO] Logged training progress to TensorBoard")
        except Exception as e:
            print(f"[WARNING] Failed to log training progress: {e}")
    
    def _log_predictions(self, y_true, y_pred, step=0, tag="predictions"):
        """Log prediction plots to TensorBoard if logger is available."""
        if self.logger is None:
            return
            
        try:
            # Create prediction plot
            fig = plt.figure(figsize=(12, 6))
            
            # Plot actual vs predicted
            plt.plot(y_true, label='Actual', alpha=0.7, linewidth=2)
            plt.plot(y_pred, label='Predicted', alpha=0.7, linewidth=2)
            
            plt.title(f"SVR {tag.title()}")
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Log the figure to TensorBoard
            self.logger.log_figure(fig, tag=f"svr/{tag}", step=step)
            plt.close(fig)  # Close the figure to free memory
            
            print(f"[INFO] Logged {tag} plot to TensorBoard")
        except Exception as e:
            print(f"[WARNING] Failed to log {tag} plot: {e}")