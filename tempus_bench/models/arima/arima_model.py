"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides an ARIMA model implementation for univariate time series forecasting.
ARIMA models combine autoregression, differencing, and moving average components to
capture temporal dependencies in time series data.

The model supports both seasonal and non-seasonal ARIMA variants and can handle
exogenous variables for enhanced forecasting performance.
"""

from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field
from statsmodels.tsa.arima.model import ARIMA

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ArimaHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    p: int = Field(..., ge=0, description="AR order (autoregressive)")
    d: int = Field(..., ge=0, description="Differencing order (integration)")
    q: int = Field(..., ge=0, description="MA order (moving average)")
    s: int = Field(..., ge=1, description="Seasonality period")


class ArimaModel(BaseModel):
    """
    ARIMA model for univariate time series forecasting.

    Supports both standard ARIMA(p, d, q) and seasonal ARIMA by specifying a seasonality period s.
    Implements traditional statsmodels ARIMA fitting, with future expandability to exogenous variables and
    rolling window forecasts.

    Attributes:
        p: int - Autoregressive order
        d: int - Differencing order
        q: int - Moving average order
        s: int - Seasonality period (s=1 disables seasonal ARIMA)

    The trained model is stored as a statsmodels ARIMA result object in self._model.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ArimaHyperparams)
        self._models = []

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "ArimaModel":
        """
        Fit the ARIMA model on the provided context data.

        Args:
            y_context (np.ndarray): Historical target values for model fitting.
            y_target (np.ndarray): Future target values (unused; included for compatibility).
            timestamps_context (np.ndarray): Timestamps corresponding to y_context (unused).
            timestamps_target (np.ndarray): Timestamps corresponding to y_target (unused).
            freq (str): Frequency string (required for interface, unused by ARIMA).

        Returns:
            ArimaModel: The fitted model instance.

        Notes:
            Only y_context is used for fitting the ARIMA model. All other arguments are ignored to avoid data leakage.
        """

        freq = kwargs["freq"]

        exog = None

        if x_context is not None:
            exog = x_context.squeeze()
        endog = y_context.squeeze()

        if self.s > 1:
            model = ARIMA(
                endog=endog,
                order=(self.p, self.d, self.q),
                seasonal_order=(0, 0, 0, self.s),
                exog=exog,
            )
        else:
            model = ARIMA(
                endog=endog,
                order=(self.p, self.d, self.q),
                exog=exog,
            )

        fitted_model = model.fit()

        return fitted_model

    @validate_inputs
    def _predict(
        self,
        arima_model,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Predicts future values using the trained ARIMA model.

        Args:
            y_context: Past target values (unused, present for interface consistency)
            timestamps_context: Context timestamps (unused)
            timestamps_target: Timestamps for the forecast horizon
            x_target: Exogenous variables for the forecast horizon (if model was trained with exog)
            freq: Frequency string (must be provided in kwargs)

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If the model is not fitted, freq is missing/empty, or forecast horizon is invalid.
        """

        if not self.is_fitted:
            raise ValueError("ArimaModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for ARIMA prediction."
            )

        forecast_steps = len(timestamps_target)

        exog = None
        if x_target is not None:
            exog = x_target.squeeze()

        forecast = arima_model.forecast(steps=forecast_steps, exog=exog)
        y_pred = np.asarray(forecast).reshape(-1, 1)

        return y_pred

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "ArimaModel":
        """
        Trains a separate ARIMA model for each variate in a (potentially multivariate) time series.

        Expects y_context and y_target as 2D arrays: (num_steps, num_targets).
        """
        num_targets = y_context.shape[1]

        for k in range(num_targets):
            fitted_model = self._train(
                y_context=y_context[:, k : k + 1],
                y_target=y_target[:, k : k + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                x_context=x_context,
                x_target=x_target,
                **kwargs,
            )
            self._models.append(fitted_model)

        self.is_fitted = True

        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Predicts future values for each variate and concatenates the results.

        Args:
            y_context (np.ndarray): Context values, shape (num_steps, num_variates).
            timestamps_context (np.ndarray): Timestamps for context data.
            timestamps_target (np.ndarray): Timestamps for target/future data.
            x_context (Optional[np.ndarray]): Optional covariate data for context.
            x_target (Optional[np.ndarray]): Optional covariate data for prediction horizon.
            **kwargs: Should include 'freq' key.

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, num_variates).

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("ArimaModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                arima_model=fitted_model,
                y_context=y_context[:, idx : idx + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                x_context=x_context,
                x_target=x_target,
                **kwargs,
            )
            preds.append(prediction)

        result = np.concatenate(preds, axis=-1)
        return result
