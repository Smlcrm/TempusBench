"""
Multivariate Theta model implementation.

This model extends the univariate Theta to handle multiple target variables simultaneously.

Since Theta is mainly a univariate model, the structure had to be changed for multivariate forecasting:
- Uses a parameter matrix Θ instead of a single θ parameter.
- Each target's θ-line depends on all targets via cross-dependencies.
- Estimates parameters via multivariate least squares or reduced rank regression.
- Predicts all targets simultaneously using sktime's multivariate capabilities.

Source of the method: 
https://onlinelibrary.wiley.com/doi/full/10.1002/for.2334 ( Forecasting Multivariate Time Series with the Theta Method )
"""

import os
import pickle
from typing import Dict, Any, Union, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sktime.forecasting.theta import ThetaForecaster
from benchmarking_pipeline.models.base_model import BaseModel


class MultivariateTheta(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the Multivariate Theta model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    Example Format:
                    {
                        'sp': 12,  # seasonal period
                        'use_reduced_rank': False,  # whether to use cointegration/reduced rank
                        'theta_method': 'least_squares'  # 'least_squares' or 'correlation_optimal'
                    }
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        if 'sp' not in self.config:
            raise ValueError("sp must be specified in config")
        if 'use_reduced_rank' not in self.config:
            raise ValueError("use_reduced_rank must be specified in config")
        if 'theta_method' not in self.config:
            raise ValueError("theta_method must be specified in config")
        
        self.sp = self.config['sp']  # Seasonality period
        self.use_reduced_rank = self.config['use_reduced_rank']
        self.theta_method = self.config['theta_method']
        self.univariate_models = {}
        self.theta_matrix = None
        self.drift_vector = None
        self.is_fitted = False
        
    def _estimate_drift(self, y_context: np.ndarray) -> np.ndarray:
        """
        Estimate drift vector μ from first differences.
        
        Args:
            y_context: Historical target data (T, num_targets)
            
        Returns:
            np.ndarray: Drift vector of shape (num_targets,)
        """
        # Calculate first differences
        diff_data = np.diff(y_context, axis=0)
        # Drift is mean of first differences
        return np.mean(diff_data, axis=0)
    
    def _detrend_data(self, y_context: np.ndarray, drift_vector: np.ndarray) -> np.ndarray:
        """
        Remove linear trend from data.
        
        Args:
            y_context: Historical target data (T, num_targets)
            drift_vector: Drift parameters (num_targets,)
            
        Returns:
            np.ndarray: Detrended data
        """
        T = y_context.shape[0]
        time_index = np.arange(1, T + 1).reshape(-1, 1)
        linear_trend = time_index @ drift_vector.reshape(1, -1)
        return y_context - linear_trend
    
    def _estimate_theta_matrix_least_squares(self, detrended_data: np.ndarray) -> np.ndarray:
        """
        Estimate Θ matrix using multivariate least squares.
        
        Args:
            detrended_data: Detrended target data (T, num_targets)
            
        Returns:
            np.ndarray: Theta matrix (num_targets, num_targets)
        """
        # For each target, regress its differences on all targets' levels
        theta_matrix = np.zeros((self.num_targets, self.num_targets))
        
        # Calculate first differences of detrended data
        diff_data = np.diff(detrended_data, axis=0)  # (T-1, num_targets)
        lagged_data = detrended_data[:-1, :]  # (T-1, num_targets)
        
        for i in range(self.num_targets):
            # Regress diff_i on all lagged levels
            y = diff_data[:, i]  # dependent variable
            X = lagged_data  # independent variables (all targets)
            
            # Fit regression
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y)
            theta_matrix[i, :] = reg.coef_
            
        return theta_matrix
    
    def _estimate_theta_matrix_correlation_optimal(self, detrended_data: np.ndarray) -> np.ndarray:
        """
        Estimate Θ matrix using correlation-optimal method from the paper.
        
        Args:
            detrended_data: Detrended target data (T, num_targets)
            
        Returns:
            np.ndarray: Theta matrix (num_targets, num_targets)
        """
        # Calculate first differences
        diff_data = np.diff(detrended_data, axis=0)
        
        # Estimate correlation matrices
        cov_matrix = np.cov(diff_data.T)
        
        # For simplicity, use the correlation structure to set off-diagonals
        # In practice, this would involve more complex optimization as in the paper
        theta_matrix = np.eye(self.num_targets)  # Start with identity
        
        # Set off-diagonal elements based on correlations
        corr_matrix = np.corrcoef(diff_data.T)
        for i in range(self.num_targets):
            for j in range(self.num_targets):
                if i != j:
                    # Use correlation as proxy for optimal theta (simplified)
                    theta_matrix[i, j] = 0.5 * corr_matrix[i, j]
                    
        return theta_matrix
    
    def _create_theta_lines(self, detrended_data: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
        """
        Create multivariate θ-lines using the parameter matrix.
        
        Args:
            detrended_data: Detrended target data (T, num_targets)
            theta_matrix: Theta parameter matrix (num_targets, num_targets)
            
        Returns:
            np.ndarray: Theta-lines (T, num_targets)
        """
        # θ-line = Θ @ detrended_data.T
        # Each row of theta_matrix multiplied by each column of detrended_data.T
        theta_lines = detrended_data @ theta_matrix.T  # (T, num_targets)
        return theta_lines
    
    def train(self, 
              y_context: Union[pd.Series, np.ndarray, pd.DataFrame], 
              y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, 
              y_start_date: Optional[str] = None,
              **kwargs
    ) -> 'MultivariateTheta':
        """
        Train the Multivariate Theta model on given data.
        
        TECHNIQUE: Multivariate Theta with Parameter Matrix
        - Estimates drift vector μ from first differences of all targets
        - Detrends all target series by removing linear trends
        - Estimates parameter matrix Θ via least squares or correlation-optimal method
        - Creates θ-lines for each target that depend on all targets
        - Fits exponential smoothing to each θ-line for forecasting
        
        Args:
            y_context: Historical target values (DataFrame for multivariate, Series for univariate)
            y_target: Target values for validation (ignored during training)
            y_start_date: Start date for y_context (optional)
        
        Returns:
            self: The fitted model instance
        """
        # Convert input to numpy array
        if isinstance(y_context, pd.DataFrame):
            # Multivariate case - use all columns
            target_data = y_context.values  # (T, num_targets)
        elif isinstance(y_context, pd.Series):
            # Univariate case - reshape to 2D
            target_data = y_context.values.reshape(-1, 1)
        else:
            # Numpy array case
            target_data = y_context
            if target_data.ndim == 1:
                target_data = target_data.reshape(-1, 1)
        
        # Calculate num_targets from data
        self.num_targets = target_data.shape[1]
        
        print(f"Training Multivariate Theta with {self.num_targets} targets...")
        
        # Step 1: Estimate drift vector μ
        self.drift_vector = self._estimate_drift(target_data)
        print(f"Estimated drift vector: {self.drift_vector}")
        
        # Step 2: Detrend the data
        detrended_data = self._detrend_data(target_data, self.drift_vector)
        
        # Step 3: Estimate Θ parameter matrix
        if self.theta_method == 'correlation_optimal':
            self.theta_matrix = self._estimate_theta_matrix_correlation_optimal(detrended_data)
        else:
            self.theta_matrix = self._estimate_theta_matrix_least_squares(detrended_data)
        
        print(f"Estimated Θ matrix:\n{self.theta_matrix}")
        
        # Step 4: Create θ-lines
        theta_lines = self._create_theta_lines(detrended_data, self.theta_matrix)
        
        # Step 5: Fit individual Theta models to each θ-line
        for i in range(self.num_targets):
            # Convert to pandas Series for sktime
            theta_line_series = pd.Series(theta_lines[:, i])
            
            # Create and fit univariate Theta model
            theta_model = ThetaForecaster(sp=self.sp)
            theta_model.fit(y=theta_line_series)
            
            self.univariate_models[i] = theta_model
        
        self.is_fitted = True
        print("Multivariate Theta training complete.")
        return self
        
    def predict(self, 
                y_context: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None,
                forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using the trained Multivariate Theta model.
        
        TECHNIQUE: Multivariate Theta Forecasting
        - Uses fitted individual Theta models to forecast each θ-line
        - Combines forecasts and adds back linear trends
        - Returns forecasts for all targets simultaneously
        
        Args:
            y_context: Recent/past target values (ignored - uses training data)
            x_context: Recent/past exogenous variables (ignored)
            x_target: Future exogenous variables (ignored, used only for forecast length)
            forecast_horizon: Number of steps to forecast
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Determine forecast horizon
        if forecast_horizon is not None:
            fh = np.arange(1, forecast_horizon + 1)
        elif x_target is not None:
            fh = np.arange(1, len(x_target) + 1)
        else:
            fh = np.arange(1, self.forecast_horizon + 1)
        
        forecast_steps = len(fh)
        
        # Get predictions from each individual Theta model
        all_predictions = np.zeros((forecast_steps, self.num_targets))
        
        for i in range(self.num_targets):
            # Get forecast from individual model
            theta_forecast = self.univariate_models[i].predict(fh=fh)
            
            # Add back the linear trend
            future_times = np.arange(1, forecast_steps + 1)
            linear_trend = future_times * self.drift_vector[i]
            
            all_predictions[:, i] = theta_forecast.values + linear_trend
        
        return all_predictions

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        params = {
            'sp': self.sp,
            'num_targets': self.num_targets,
            'use_reduced_rank': self.use_reduced_rank,
            'theta_method': self.theta_method,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon
        }
        
        # Add learned parameters if fitted
        if self.is_fitted:
            params.update({
                'theta_matrix': self.theta_matrix.tolist() if self.theta_matrix is not None else None,
                'drift_vector': self.drift_vector.tolist() if self.drift_vector is not None else None
            })
            
        return params

    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateTheta':
        """
        Set model parameters. This will reset the fitted model.
        """
        model_params_changed = False
        for key, value in params.items():
            if hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['sp', 'use_reduced_rank', 'theta_method'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed and self.is_fitted:
            self.univariate_models = {}
            self.theta_matrix = None
            self.drift_vector = None
            self.is_fitted = False
            
        return self

    def save(self, path: str) -> None:
        """
        Save the trained Multivariate Theta model to disk using pickle.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Package all model components
        model_data = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'theta_matrix': self.theta_matrix,
            'drift_vector': self.drift_vector,
            'univariate_models': self.univariate_models,
            'num_targets': self.num_targets,
            'sp': self.sp,
            'use_reduced_rank': self.use_reduced_rank,
            'theta_method': self.theta_method
        }
            
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load(self, path: str) -> 'MultivariateTheta':
        """
        Load a trained Multivariate Theta model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore all model components
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.theta_matrix = model_data['theta_matrix']
        self.drift_vector = model_data['drift_vector']
        self.univariate_models = model_data['univariate_models']
        self.num_targets = model_data['num_targets']
        self.sp = model_data['sp']
        self.use_reduced_rank = model_data['use_reduced_rank']
        self.theta_method = model_data['theta_method']
        
        return self
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values.
        For multivariate case, computes loss for each target and averages.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
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
            
        # Store for logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # For multivariate data, compute loss for each target and average
        if y_true.ndim == 2 and y_pred.ndim == 2 and self.num_targets > 1:
            # Multivariate case - compute loss for each target and average
            all_metrics = {}
            for i in range(min(y_true.shape[1], y_pred.shape[1], self.num_targets)):
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
            if y_pred.ndim == 2 and y_true.ndim == 1:
                if y_pred.shape[0] == 1:
                    y_pred = y_pred.flatten()
                elif y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    y_pred = y_pred[0]
            
            # Ensure both arrays have the same length
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            
            return self.evaluator.evaluate(y_pred, y_true)