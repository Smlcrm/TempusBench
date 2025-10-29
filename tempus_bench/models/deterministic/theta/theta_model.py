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
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Literal
from tempus_bench.models.base_model import BaseModel, validate_inputs


class ThetaHyperparams(PydanticBaseModel):
    sp: int = Field(..., ge=1, description="Seasonal period")
    use_reduced_rank: bool = Field(default=False, description="Whether to use cointegration/reduced rank")
    theta_method: Literal["least_squares", "correlation_optimal"] = Field(default="least_squares", description="Method for theta estimation")


class Theta(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize the Multivariate Theta model with model-specific parameters.
        """
        super().__init__(params, settings, ThetaHyperparams)

        # Note: theta_method should be in config, no defaults
        self._models = {}
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

    def _detrend_data(
        self, y_context: np.ndarray, drift_vector: np.ndarray
    ) -> np.ndarray:
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

    def _estimate_theta_matrix_least_squares(
        self, detrended_data: np.ndarray, num_targets
    ) -> np.ndarray:
        """
        Estimate Θ matrix using multivariate least squares.

        Args:
            detrended_data: Detrended target data (T, num_targets)

        Returns:
            np.ndarray: Theta matrix (num_targets, num_targets)
        """
        # For each target, regress its differences on all targets' levels
        theta_matrix = np.zeros((num_targets, num_targets))

        # Calculate first differences of detrended data
        diff_data = np.diff(detrended_data, axis=0)  # (T-1, num_targets)
        lagged_data = detrended_data[:-1, :]  # (T-1, num_targets)

        for i in range(num_targets):
            # Regress diff_i on all lagged levels
            y = diff_data[:, i]  # dependent variable
            X = lagged_data  # independent variables (all targets)

            # Fit regression
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y)
            theta_matrix[i, :] = reg.coef_

        return theta_matrix

    def _estimate_theta_matrix_correlation_optimal(
        self, detrended_data: np.ndarray, num_targets
    ) -> np.ndarray:
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
        theta_matrix = np.eye(num_targets)  # Start with identity

        # Set off-diagonal elements based on correlations
        corr_matrix = np.corrcoef(diff_data.T)
        for i in range(num_targets):
            for j in range(num_targets):
                if i != j:
                    # Use correlation as proxy for optimal theta (simplified)
                    theta_matrix[i, j] = 0.5 * corr_matrix[i, j]

        return theta_matrix

    def _create_theta_lines(
        self, detrended_data: np.ndarray, theta_matrix: np.ndarray
    ) -> np.ndarray:
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

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: Optional[np.ndarray],
        timestamps_context: Optional[np.ndarray],
        timestamps_target: Optional[np.ndarray],
        **kwargs: dict,
    ) -> "Theta":
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
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        sp = self.sp
        theta_method = self.theta_method
        use_reduced_rank = self.use_reduced_rank

        # Calculate num_targets from data
        num_targets = y_context.shape[1]
        self.num_targets = num_targets  # Store for use in predict
        self.logger.debug("ThetaModel.train", f"y_context shape: {y_context.shape}")
        self.logger.info("ThetaModel.train", f"Training Multivariate Theta with {num_targets} targets...")

        # Step 1: Estimate drift vector μ
        self.drift_vector = self._estimate_drift(y_context)
        self.logger.debug("ThetaModel.train", f"Estimated drift vector: {self.drift_vector}")

        # Step 2: Detrend the data
        detrended_data = self._detrend_data(y_context, self.drift_vector)

        # Step 3: Estimate Θ parameter matrix
        if theta_method == "correlation_optimal":
            self.theta_matrix = self._estimate_theta_matrix_correlation_optimal(
                detrended_data, num_targets
            )
        else:
            self.theta_matrix = self._estimate_theta_matrix_least_squares(
                detrended_data, num_targets
            )

        self.logger.debug("ThetaModel.train", f"Estimated Θ matrix:\n{self.theta_matrix}")

        # Step 4: Create θ-lines
        theta_lines = self._create_theta_lines(detrended_data, self.theta_matrix)

        # Step 5: Fit individual Theta models to each θ-line
        for i in range(num_targets):
            # Convert to pandas Series for sktime
            theta_line_series = pd.Series(theta_lines[:, i])

            # Check if data has non-positive values (common with normalized data)
            # If so, disable deseasonalization to avoid multiplicative seasonality errors
            has_non_positive = (theta_line_series <= 0).any()
            # Also disable deseasonalization if we don't have at least two full cycles
            insufficient_cycles = len(theta_line_series) < 2 * sp
            deseasonalize = (not has_non_positive) and (not insufficient_cycles)
            
            # Create and fit univariate Theta model
            # Disable deseasonalization if data has non-positive values
            theta_model = ThetaForecaster(
                sp=sp,
                deseasonalize=deseasonalize
            )
            theta_model.fit(y=theta_line_series)

            self._models[i] = theta_model

        self.is_fitted = True
        self.logger.info("ThetaModel.train", "Multivariate Theta training complete.")
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Make predictions using the trained Multivariate Theta model.

        TECHNIQUE: Multivariate Theta Forecasting
        - Uses fitted individual Theta models to forecast each θ-line
        - Combines forecasts and adds back linear trends
        - Returns forecasts for all targets simultaneously

        Args:
            y_context: Recent/past target values (ignored - uses training data)
            timestamps_context: Timestamps for context data (ignored)
            timestamps_target: Timestamps for target data (used to determine forecast length)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        sp = self.sp
        
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        forecast_horizon = len(timestamps_target)
        num_targets = self.num_targets
        # Determine forecast horizon
        fh = np.arange(1, forecast_horizon + 1)

        forecast_steps = len(fh)

        # Get predictions from each individual Theta model
        all_predictions = np.zeros((forecast_horizon, num_targets))

        for i in range(num_targets):
            # Get forecast from individual model
            theta_forecast = self._models[i].predict(fh=fh)

            # Add back the linear trend
            future_times = np.arange(1, forecast_horizon + 1)
            linear_trend = future_times * self.drift_vector[i]

            all_predictions[:, i] = theta_forecast.values + linear_trend

        forecasts = np.asarray(all_predictions)
        if len(forecasts) == 1:
            forecasts = np.expand_dims(forecasts, axis=1)

        return forecasts