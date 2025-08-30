"""
Multivariate SVR model implementation.

This model extends the univariate SVR to handle multiple target variables simultaneously.
Uses sklearn's MultiOutputRegressor to handle multiple targets with a single SVR base estimator.
"""

import os
import pickle
from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.multioutput import MultiOutputRegressor


class MultivariateSVRModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Multivariate Support Vector Regression (SVR) model with a given configuration.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor for multiple target variables.
        
        Args:
            config: Configuration dictionary containing model parameters
                - lookback_window: int, number of past timesteps to use as features
                - forecast_horizon: int, number of future timesteps to predict
                - model_params: dict, parameters for the underlying SVR model
                - training_loss: str, primary loss function for training
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        self.scaler = StandardScaler()  # SVR is sensitive to feature scaling
        
        # Extract model parameters from config
        if 'lookback_window' not in self.config:
            raise ValueError("lookback_window must be specified in config")
        if 'forecast_horizon' not in self.config:
            raise ValueError("forecast_horizon must be specified in config")
        self.lookback_window = self.config['lookback_window']
        self.forecast_horizon = self.config['forecast_horizon']
        # num_targets will be calculated from data during training
        
        self._build_model()

    def _build_model(self):
        """
        Build the Multivariate SVR model instance from the configuration using MultiOutputRegressor 
        for direct multi-output forecasting across multiple target variables.
        """
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        model_params = self.config['model_params']
        base_svr = SVR(**model_params)
        self.model = MultiOutputRegressor(base_svr)
        self.is_fitted = False

    def _create_features_targets(self, y_series: np.ndarray, lookback_window: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features and multi-step targets for direct multi-output forecasting across multiple variables.
        Each sample uses the previous lookback_window values for all targets as features and 
        the next forecast_horizon values for all targets as targets.
        
        Args:
            y_series: Input time series data with shape (timesteps, num_targets)
            lookback_window: Number of past timesteps to use as features
            forecast_horizon: Number of future timesteps to predict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and targets (y) arrays
        """
        if y_series.ndim == 1:
            y_series = y_series.reshape(-1, 1)
        
        X, y = [], []
        for i in range(len(y_series) - lookback_window - forecast_horizon + 1):
            # Features: flatten the lookback window across all targets
            # Shape: (lookback_window * num_targets,)
            X.append(y_series[i:i+lookback_window].flatten())
            
            # Targets: flatten the forecast horizon across all targets
            # Shape: (forecast_horizon * num_targets,)
            y.append(y_series[i+lookback_window:i+lookback_window+forecast_horizon].flatten())
            
        return np.array(X), np.array(y)

    def train(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame], y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_start_date: pd.Timestamp = None, **kwargs) -> 'MultivariateSVRModel':
        """
        Train the Multivariate SVR model for direct multi-output forecasting using MultiOutputRegressor.
        
        Args:
            y_context: Past target values (time series) - used for training (can be DataFrame for multivariate)
            y_target: Future target values (optional, for extended training)
            y_start_date: The start date timestamp for y_context and y_target
            **kwargs: Additional keyword arguments
            
        Returns:
            self: The fitted model instance
        """
        if self.model is None:
            self._build_model()
            
        if y_context is None:
            raise ValueError("y_context (target series) must be provided for training.")

        # Convert input to numpy array and handle multivariate data
        if isinstance(y_context, pd.DataFrame):
            # Multivariate case - use all columns
            y_series = y_context.values
        elif isinstance(y_context, pd.Series):
            # Univariate case - reshape to 2D
            y_series = y_context.values.reshape(-1, 1)
        else:
            # Numpy array case
            y_series = y_context
            if y_series.ndim == 1:
                y_series = y_series.reshape(-1, 1)

        # Calculate num_targets from data
        self.num_targets = y_series.shape[1]
        
        # Combine context and target for full training series if y_target is provided
        if y_target is not None:
            if isinstance(y_target, pd.DataFrame):
                y_target_vals = y_target.values
            elif isinstance(y_target, pd.Series):
                y_target_vals = y_target.values.reshape(-1, 1)
            else:
                y_target_vals = y_target
                if y_target_vals.ndim == 1:
                    y_target_vals = y_target_vals.reshape(-1, 1)
            
            y_series = np.concatenate([y_series, y_target_vals], axis=0)

        X, y = self._create_features_targets(y_series, self.lookback_window, self.forecast_horizon)
        print(f"[DEBUG][MultivariateSVR] Training X shape: {X.shape}, y shape: {y.shape}")
        print(f"[DEBUG][MultivariateSVR] num_targets: {self.num_targets}")
        
        # Fail fast on NaN data - don't silently replace
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values. Please clean your data before training.")
        
        # Scale features (flattened lookback window values)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        print("[INFO][MultivariateSVR] Training completed successfully")
        
        return self

    def rolling_predict(self, y_context: np.ndarray, y_target: Union[pd.Series, np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Autoregressive rolling prediction for Multivariate SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        
        Args:
            y_context: Historical context data with shape (timesteps, num_targets)
            y_target: Target data to determine prediction length
            
        Returns:
            np.ndarray: Predictions with shape (forecast_steps, num_targets)
        """
        # Convert y_context to proper shape
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        print(f"[DEBUG][MultivariateSVR] Rolling predict - y_context shape: {y_context.shape}")
        print(f"[DEBUG][MultivariateSVR] Rolling predict - lookback_window: {self.lookback_window}")
        
        if len(y_context) < self.lookback_window:
            raise ValueError(f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}")
        if np.isnan(y_context).any():
            raise ValueError("NaNs in initial y_context!")
        
        preds = []
        # Use the entire context, maintaining the multivariate structure
        context = y_context.copy()
        total_steps = len(y_target)
        steps_done = 0
        
        while steps_done < total_steps:
            steps = min(self.forecast_horizon, total_steps - steps_done)
            
            # Take the last lookback_window timesteps and flatten for SVR input
            X_pred = context[-self.lookback_window:].flatten().reshape(1, -1)
            
            # Fail fast on NaN input - don't silently replace
            if np.isnan(X_pred).any():
                raise ValueError("Prediction input contains NaN values. This indicates data corruption.")
            
            X_pred_scaled = self.scaler.transform(X_pred)
            pred_flat = self.model.predict(X_pred_scaled).flatten()
            
            # Fail fast on NaN predictions - don't silently replace
            if np.isnan(pred_flat).any():
                raise ValueError("Model produced NaN predictions. This indicates a training or data issue.")
            
            # Reshape predictions back to (forecast_horizon, num_targets)
            pred_reshaped = pred_flat.reshape(self.forecast_horizon, self.num_targets)
            
            print(f"[DEBUG][MultivariateSVR] Rolling predict X_pred shape: {X_pred.shape}, pred_reshaped shape: {pred_reshaped.shape}")
            
            # Only take as many steps as needed
            pred_steps = pred_reshaped[:steps]
            preds.append(pred_steps)
            
            # Update context with new predictions
            context = np.concatenate([context, pred_steps], axis=0)
            steps_done += steps
        
        # Concatenate all predictions
        result = np.concatenate(preds, axis=0)
        return result

    def predict(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """
        Make direct multi-output predictions using the trained Multivariate SVR model.
        If y_target is provided, use rolling_predict to predict the entire length autoregressively.
        
        Args:
            y_context: Historical context data
            y_target: Target data (used to determine prediction length)
            x_context: Exogenous context data (optional, ignored for now)
            x_target: Exogenous target data (optional, ignored for now)
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None:
            raise ValueError("y_context must be provided to determine prediction context.")
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")

        # Convert y_context to proper format
        if isinstance(y_context, pd.DataFrame):
            y_context_vals = y_context.values
        elif isinstance(y_context, pd.Series):
            y_context_vals = y_context.values.reshape(-1, 1)
        else:
            y_context_vals = y_context
            if y_context_vals.ndim == 1:
                y_context_vals = y_context_vals.reshape(-1, 1)

        # Use rolling prediction to cover the full test set
        preds = self.rolling_predict(y_context_vals, y_target)
        print(f"[DEBUG][MultivariateSVR] Final rolling preds shape: {preds.shape}")
        return preds

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying scikit-learn model.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        params = {
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'num_targets': self.num_targets,
            'training_loss': self.training_loss
        }
        
        if self.model:
            params.update(self.model.get_params())
            
        return params

    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateSVRModel':
        """
        Set model parameters. This will rebuild the model instance with the new parameters.
        Handles MultiOutputRegressor by prefixing SVR params with 'estimator__'.
        Model-level params (lookback_window, forecast_horizon) are set as attributes.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        # Handle model-level params
        model_level_keys = ['lookback_window', 'forecast_horizon']
        for k in model_level_keys:
            if k in params:
                setattr(self, k, params[k])
        
        # Convert parameter types for scikit-learn compatibility
        converted_params = {}
        for k, v in params.items():
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
        
        # Prepare params for underlying SVR
        svr_param_names = SVR().get_params().keys()
        mo_params = {}
        for k, v in converted_params.items():
            if k in svr_param_names:
                mo_params[f'estimator__{k}'] = v
            elif k == 'n_jobs':
                mo_params[k] = v
                
        if hasattr(self, 'model') and isinstance(self.model, MultiOutputRegressor):
            self.model.set_params(**mo_params)
        elif hasattr(self, 'model') and self.model is not None:
            # fallback for non-multioutput
            self.model.set_params(**{k: v for k, v in converted_params.items() if k in svr_param_names or k == 'n_jobs'})
            
        # Update config as well
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update({k: v for k, v in converted_params.items() if k in svr_param_names or k == 'n_jobs'})
        
        for k in model_level_keys:
            if k in params:
                self.config[k] = params[k]
                
        return self
        
    def save(self, path: str) -> None:
        """
        Save the trained Multivariate SVR model and its scaler to disk using pickle.
        
        Args:
            path: Path to save the model objects.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Save model state
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params(),
            'model': self.model,
            'scaler': self.scaler
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'MultivariateSVRModel':
        """
        Load a trained Multivariate SVR model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            self: The loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.model = model_state['model']
        self.scaler = model_state['scaler']
        
        # Restore attributes
        saved_params = model_state['params']
        for key in ['lookback_window', 'forecast_horizon', 'num_targets']:
            if key in saved_params:
                setattr(self, key, saved_params[key])
                
        return self

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values with shape (timesteps, num_targets)
            y_pred: Predicted values with shape (timesteps, num_targets)
            loss_function: Name of the loss function to use (defaults to training_loss)
            
        Returns:
            Dict[str, float]: Dictionary of computed loss metrics
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        elif isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Store for potential logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # For multivariate data, compute loss for each target and average
        if y_true.ndim == 2 and y_pred.ndim == 2:
            # Multivariate case - compute loss for each target and average
            all_metrics = {}
            for i in range(y_true.shape[1]):
                target_true = y_true[:, i]
                target_pred = y_pred[:, i]
                
                # Ensure both arrays have the same length
                min_length = min(len(target_true), len(target_pred))
                target_true = target_true[:min_length]
                target_pred = target_pred[:min_length]
                
                # Compute metrics for this target
                target_metrics = self.evaluator.evaluate(target_pred, target_true)
                
                # Store metrics for this target
                for metric_name, metric_value in target_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Average metrics across all targets
            averaged_metrics = {}
            for metric_name, metric_values in all_metrics.items():
                averaged_metrics[metric_name] = np.mean(metric_values)
            
            return averaged_metrics
        else:
            # Univariate case - use original logic
            # Handle shape mismatches
            if y_pred.ndim == 2 and y_true.ndim == 1:
                # If predictions are 2D and true values are 1D, flatten predictions
                if y_pred.shape[0] == 1:
                    # Single prediction row, flatten it
                    y_pred = y_pred.flatten()
                elif y_pred.shape[1] == 1:
                    # Single prediction column, flatten it
                    y_pred = y_pred.flatten()
                else:
                    # Multiple predictions, take the first row
                    y_pred = y_pred[0]
            
            # Ensure both arrays have the same length
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            
            # Use evaluator to compute all metrics
            return self.evaluator.evaluate(y_pred, y_true)