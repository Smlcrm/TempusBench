"""
Multivariate XGBoost model implementation for time series forecasting.

This model extends the univariate XGBoost to handle multiple target variables simultaneously.
Uses sklearn's MultiOutputRegressor to handle multiple targets with advanced feature engineering.
"""

import os
import pickle
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class MultivariateXGBoostModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Multivariate XGBoost model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - lookback_window: int, number of past timesteps to use as features
                - forecast_horizon: int, number of steps to forecast ahead
                - model_params: dict, parameters for the underlying XGBRegressor model
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        if 'lookback_window' not in self.config:
            raise ValueError("lookback_window must be specified in config")
        if 'forecast_horizon' not in self.config:
            raise ValueError("forecast_horizon must be specified in config")
        
        self.lookback_window = self.config['lookback_window']
        self.forecast_horizon = self.config['forecast_horizon']
        
        self._build_model()
        
    def _build_model(self):
        """
        Build the Multivariate XGBoost model instance using MultiOutputRegressor 
        for direct multi-output forecasting across multiple target variables.
        """
        # Get hyperparameters from config
        if 'model_params' not in self.config:
            raise ValueError("model_params must be specified in config")
        model_params = self.config['model_params']
        
        # Ensure random_state for reproducibility if not provided
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
            
        base_xgb = XGBRegressor(**model_params)
        self.model = MultiOutputRegressor(base_xgb)
        self.is_fitted = False

    def _create_multivariate_features(self, y_series: np.ndarray, x_series: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create advanced multivariate time series features for XGBoost.
        
        Args:
            y_series: Target time series with shape (timesteps, num_targets)
            x_series: Exogenous variables (optional)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets arrays
        """
        if y_series.ndim == 1:
            y_series = y_series.reshape(-1, 1)
            
        n_samples = len(y_series) - self.lookback_window - self.forecast_horizon + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.lookback_window + self.forecast_horizon} samples.")
        
        features = []
        targets = []
        
        for i in range(n_samples):
            sample_features = []
            
            # Get lookback window for all targets
            lookback_data = y_series[i:i + self.lookback_window]  # Shape: (lookback_window, num_targets)
            
            # 1. Lag features for each target (flattened)
            lag_features = lookback_data.flatten()  # Shape: (lookback_window * num_targets,)
            sample_features.extend(lag_features)
            
            # 2. Rolling statistics for each target individually
            for target_idx in range(self.num_targets):
                target_data = lookback_data[:, target_idx]
                
                # Basic statistics
                rolling_mean = np.mean(target_data)
                rolling_std = np.std(target_data)
                rolling_min = np.min(target_data)
                rolling_max = np.max(target_data)
                rolling_median = np.median(target_data)
                
                # Trend features
                trend = np.polyfit(range(self.lookback_window), target_data, 1)[0]
                
                # Volatility features
                rolling_range = rolling_max - rolling_min
                rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)
                
                sample_features.extend([
                    rolling_mean, rolling_std, rolling_min, rolling_max, rolling_median,
                    trend, rolling_range, rolling_iqr
                ])
            
            # 3. Cross-correlation features between targets (if multivariate)
            if self.num_targets > 1:
                for i_target in range(self.num_targets):
                    for j_target in range(i_target + 1, self.num_targets):
                        # Correlation between target pairs
                        corr = np.corrcoef(lookback_data[:, i_target], lookback_data[:, j_target])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0  # Handle constant series
                        sample_features.append(corr)
                        
                        # Ratio features
                        mean_i = np.mean(lookback_data[:, i_target])
                        mean_j = np.mean(lookback_data[:, j_target])
                        if mean_j != 0:
                            ratio = mean_i / mean_j
                        else:
                            ratio = 0.0
                        sample_features.append(ratio)
            
            # 4. Temporal features (if window is large enough)
            if self.lookback_window >= 7:
                # Weekly patterns (last 7 values for each target)
                recent_data = lookback_data[-7:]
                for target_idx in range(self.num_targets):
                    recent_mean = np.mean(recent_data[:, target_idx])
                    sample_features.append(recent_mean)
            
            # 5. Add exogenous features if available
            if x_series is not None and len(x_series) > i + self.lookback_window:
                current_x = x_series[i + self.lookback_window]
                sample_features.extend(current_x.flatten())
            
            features.append(sample_features)
            
            # Create target: next forecast_horizon steps for all targets, flattened
            future_values = y_series[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon]
            targets.append(future_values.flatten())  # Shape: (forecast_horizon * num_targets,)
        
        return np.array(features), np.array(targets)

    def train(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame], y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_start_date: pd.Timestamp = None, **kwargs) -> 'MultivariateXGBoostModel':
        """
        Train the Multivariate XGBoost model for direct multi-output forecasting.
        
        TECHNIQUE: Advanced Multivariate Feature Engineering with Gradient Boosting
        - Creates lag features from all target variables
        - Adds rolling statistics per target (mean, std, min, max, median, trend, range, IQR)
        - Includes cross-correlation and ratio features between target pairs
        - Incorporates temporal patterns (weekly means if window ≥ 7)
        - Uses XGBoost's gradient boosting with MultiOutputRegressor for non-linear pattern learning
        
        Args:
            y_context: Past target values (DataFrame for multivariate)
            y_target: Future target values (optional, for validation)
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
            y_data = y_context.values
            # Calculate num_targets from data shape
            self.num_targets = y_data.shape[1]
        elif isinstance(y_context, pd.Series):
            # Univariate case - reshape to 2D
            y_data = y_context.values.reshape(-1, 1)
            self.num_targets = 1
        else:
            # Numpy array case
            y_data = y_context
            if y_data.ndim == 1:
                y_data = y_data.reshape(-1, 1)
            self.num_targets = y_data.shape[1]

        # Store the raw target data for later use
        self.y_context = y_data
        
        # Handle exogenous variables
        if x_context is not None:
            if isinstance(x_context, pd.DataFrame):
                x_data = x_context.values
            elif isinstance(x_context, pd.Series):
                x_data = x_context.values.reshape(-1, 1)
            else:
                x_data = x_context
                if x_data.ndim == 1:
                    x_data = x_data.reshape(-1, 1)
            self.x_context = x_data
        else:
            self.x_context = None
        
        # Store target data for later use
        if y_target is not None:
            if isinstance(y_target, pd.DataFrame):
                self.y_target = y_target.values
            elif isinstance(y_target, pd.Series):
                self.y_target = y_target.values.reshape(-1, 1)
            else:
                self.y_target = y_target
                if self.y_target.ndim == 1:
                    self.y_target = self.y_target.reshape(-1, 1)
        else:
            self.y_target = None
        
        # Store timestamps
        self.y_start_date = y_start_date
        self.x_start_date = x_start_date
        
        # Prepare features for training
        features = self._create_multivariate_features(y_data, self.x_context)
        
        # Prepare targets for training
        if self.y_target is not None:
            targets = self.y_target
        else:
            # If no target provided, use the last forecast_horizon steps as target
            targets = y_data[-self.forecast_horizon:]
            features = features[:-self.forecast_horizon]
        
        # Train the model
        self.model.fit(features, targets)
        self.is_fitted = True
        
        print(f"[DEBUG][MultivariateXGBoost] num_targets: {self.num_targets}")
        
        return self

    def rolling_predict(self, y_context: np.ndarray, y_target: Union[pd.Series, np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Autoregressive rolling prediction for Multivariate XGBoost.
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

        print(f"[DEBUG][MultivariateXGBoost] Rolling predict - y_context shape: {y_context.shape}")
        print(f"[DEBUG][MultivariateXGBoost] Rolling predict - lookback_window: {self.lookback_window}")
        
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
            
            # Take the last lookback_window timesteps for feature creation
            current_window = context[-self.lookback_window:]
            
            try:
                # Create features using the current window
                sample_features = []
                
                # 1. Lag features (flattened)
                lag_features = current_window.flatten()
                sample_features.extend(lag_features)
                
                # 2. Rolling statistics for each target
                for target_idx in range(self.num_targets):
                    target_data = current_window[:, target_idx]
                    
                    rolling_mean = np.mean(target_data)
                    rolling_std = np.std(target_data)
                    rolling_min = np.min(target_data)
                    rolling_max = np.max(target_data)
                    rolling_median = np.median(target_data)
                    trend = np.polyfit(range(self.lookback_window), target_data, 1)[0]
                    rolling_range = rolling_max - rolling_min
                    rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)
                    
                    sample_features.extend([
                        rolling_mean, rolling_std, rolling_min, rolling_max, rolling_median,
                        trend, rolling_range, rolling_iqr
                    ])
                
                # 3. Cross-correlation features (if multivariate)
                if self.num_targets > 1:
                    for i_target in range(self.num_targets):
                        for j_target in range(i_target + 1, self.num_targets):
                            corr = np.corrcoef(current_window[:, i_target], current_window[:, j_target])[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            sample_features.append(corr)
                            
                            mean_i = np.mean(current_window[:, i_target])
                            mean_j = np.mean(current_window[:, j_target])
                            ratio = mean_i / mean_j if mean_j != 0 else 0.0
                            sample_features.append(ratio)
                
                # 4. Temporal features (if applicable)
                if self.lookback_window >= 7:
                    recent_data = current_window[-7:]
                    for target_idx in range(self.num_targets):
                        recent_mean = np.mean(recent_data[:, target_idx])
                        sample_features.append(recent_mean)
                
                # 5. No exogenous features (removed as per cleanup)
                
                # Convert to numpy array and predict
                X_pred = np.array(sample_features).reshape(1, -1)
                
                # Fail fast on NaN input - don't silently replace
                if np.isnan(X_pred).any():
                    raise ValueError("Prediction features contain NaN values. This indicates data corruption.")
                
                pred_flat = self.model.predict(X_pred).flatten()
                
                # Fail fast on NaN predictions - don't silently replace
                if np.isnan(pred_flat).any():
                    raise ValueError("Model produced NaN predictions. This indicates a training or data issue.")
                
                # Reshape predictions back to (forecast_horizon, num_targets)
                pred_reshaped = pred_flat.reshape(self.forecast_horizon, self.num_targets)
                
                print(f"[DEBUG][MultivariateXGBoost] Rolling predict X_pred shape: {X_pred.shape}, pred_reshaped shape: {pred_reshaped.shape}")
                
                # Only take as many steps as needed
                pred_steps = pred_reshaped[:steps]
                preds.append(pred_steps)
                
                # Update context with new predictions
                context = np.concatenate([context, pred_steps], axis=0)
                steps_done += steps
                
            except Exception as e:
                raise RuntimeError(f"Error during prediction step: {str(e)}")
        
        # Concatenate all predictions
        result = np.concatenate(preds, axis=0)
        return result

    def predict(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained Multivariate XGBoost model.
        
        TECHNIQUE: Autoregressive Rolling Window Multi-step Forecasting with Advanced Features
        - Uses last lookback_window values to create comprehensive multivariate features
        - Predicts forecast_horizon steps ahead using trained MultiOutputRegressor
        - Uses its own predictions to update the window and predict further steps
        - Repeats until forecast_steps are reached
        
        Args:
            y_context: Historical context data
            y_target: Target data (used to determine prediction length)
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")

        # Convert y_context to proper format
        if isinstance(y_context, pd.DataFrame):
            y_context_vals = y_context.values
        elif isinstance(y_context, pd.Series):
            y_context_vals = y_context.values.reshape(-1, 1)
        else:
            y_context_vals = y_context
            if y_context_vals.ndim == 1:
                y_context_vals = y_context_vals.reshape(-1, 1)

        # Use rolling prediction to cover the full test set (no exogenous variables)
        preds = self.rolling_predict(y_context_vals, y_target, None)
        print(f"[DEBUG][MultivariateXGBoost] Final rolling preds shape: {preds.shape}")
        return preds

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'lookback_window': self.lookback_window,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'num_targets': self.num_targets,
            'forecast_horizon': self.forecast_horizon
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateXGBoostModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if hasattr(self, key):
                # Ensure proper type conversion for numeric parameters
                if key in ['n_estimators', 'max_depth', 'forecast_horizon']:
                    value = int(value)
                elif key in ['learning_rate', 'subsample', 'colsample_bytree']:
                    value = float(value)
                elif key in ['random_state']:
                    value = int(value)
                elif key in ['n_jobs']:
                    value = int(value)
                setattr(self, key, value)
        return self

    def save(self, path: str) -> None:
        """
        Save the trained Multivariate XGBoost model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params(),
            'fitted_model': self.model,
            'y_context': self.y_context,
            'x_context': self.x_context,
            'y_target': self.y_target
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'MultivariateXGBoostModel':
        """
        Load a trained Multivariate XGBoost model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        # Load model state from file
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.model = model_state['fitted_model']
        self.y_context = model_state['y_context']
        self.x_context = model_state['x_context']
        self.y_target = model_state['y_target']
        self.set_params(**model_state['params'])
        
        return self

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values with shape (timesteps, num_targets)
            y_pred: Predicted values with shape (timesteps, num_targets)
            loss_function: Name of the loss function to use (defaults to None)
            
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