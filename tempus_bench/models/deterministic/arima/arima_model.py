"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides an ARIMA model implementation for univariate time series forecasting.
ARIMA models combine autoregression, differencing, and moving average components to
capture temporal dependencies in time series data.

The model supports both seasonal and non-seasonal ARIMA variants and can handle
exogenous variables for enhanced forecasting performance.
"""

from ast import Dict
import numpy as np
from typing import Any

from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel as PydanticBaseModel, Field

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

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
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

        self.logger.debug(
            "ArimaModel._train",
            f"Start training: y_context={y_context.shape}, y_target={getattr(y_target, 'shape', None)}"
        )

        if len(timestamps_context) > 0:
            self.logger.debug(
                "ArimaModel._train",
                f"Timestamps context: {timestamps_context[0]} to {timestamps_context[-1]}"
            )
        if len(timestamps_target) > 0:
            self.logger.debug(
                "ArimaModel._train",
                f"Timestamps target: {timestamps_target[0]} to {timestamps_target[-1]}"
            )
        self.logger.debug("ArimaModel._train", f"Frequency: {freq}")

        endog = y_context.squeeze()
        self.logger.debug(
            "ArimaModel._train",
            f"Endogenous shape: {endog.shape}, sample: {endog[:5] if endog.size >= 5 else endog}"
        )

        if self.s > 1:
            self.logger.debug("ArimaModel._train", f"Seasonal ARIMA (s={self.s})")
            model = ARIMA(
                endog=endog,
                order=(self.p, self.d, self.q),
                seasonal_order=(0, 0, 0, self.s),
                exog=None,
            )
        else:
            self.logger.debug("ArimaModel._train", "Non-seasonal ARIMA")
            model = ARIMA(
                endog=endog,
                order=(self.p, self.d, self.q),
                exog=None,
            )

        self.logger.debug("ArimaModel._train", f"ARIMA params: {self.params}")
        self.logger.debug("ArimaModel._train", "Fitting model...")

        self._model = model.fit()

        self.logger.debug("ArimaModel._train", "Model fitted")
        self.logger.debug("ArimaModel._train", f"AIC: {self._model.aic}, BIC: {self._model.bic}")
        self.logger.debug("ArimaModel._train", f"Coefficients: {self._model.params}")
        self.logger.info("ArimaModel._train", "Training complete")
        self.is_fitted = True
        return self

    @validate_inputs
    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Predicts future values using the trained ARIMA model.

        Args:
            y_context: Past target values (unused, present for interface consistency)
            timestamps_context: Context timestamps (unused)
            timestamps_target: Timestamps for the forecast horizon
            freq: Frequency string (must be provided in kwargs)

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If the model is not fitted, freq is missing/empty, or forecast horizon is invalid.
        """
        self.logger.debug(
            "ArimaModel._predict",
            f"Context shape: {y_context.shape}, "
            f"Timestamps context: {timestamps_context[0] if len(timestamps_context) > 0 else 'empty'} to "
            f"{timestamps_context[-1] if len(timestamps_context) > 0 else 'empty'}, "
            f"Timestamps target: {timestamps_target[0] if len(timestamps_target) > 0 else 'empty'} to "
            f"{timestamps_target[-1] if len(timestamps_target) > 0 else 'empty'}"
        )

        if not self.is_fitted:
            raise ValueError("ArimaModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError("timestamps_target must be provided and non-empty for ARIMA prediction.")

        forecast_steps = len(timestamps_target)
        forecast = self._model.forecast(steps=forecast_steps)
        y_pred = np.asarray(forecast).reshape(-1, 1)

        self.logger.debug("ArimaModel._predict", f"Prediction shape: {y_pred.shape}")
        self.logger.info("ArimaModel._predict", "Prediction completed")

        return y_pred

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "ArimaModel":
        """
        Trains a separate ARIMA model for each variate in a (potentially multivariate) time series.

        Expects y_context and y_target as 2D arrays: (num_steps, num_targets).
        """
        num_targets = y_context.shape[1]
        self.logger.debug("ArimaModel.train", f"Number of features/variates detected: {num_targets}")

        self._models = []
        for k in range(num_targets):
            self.logger.debug("ArimaModel.train", f"Training variate k={k}")
            model = ArimaModel(params=self.model_dump())
            model._train(
                y_context=y_context[:, k],
                y_target=y_target[:, k],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs
            )
            self._models.append(model)
        self.is_fitted = True

        self.logger.info("ArimaModel.train", f"Training completed for {num_targets} variates")
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Predicts future values for each variate and concatenates the results.

        Args:
            y_context (np.ndarray): Context values, shape (num_steps, num_variates).
            timestamps_context (np.ndarray): Timestamps for context data.
            timestamps_target (np.ndarray): Timestamps for target/future data.
            **kwargs: Should include 'freq' key.

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, num_variates).

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("ArimaModel not fitted. Call train() first.")

        self.logger.debug(
            "ArimaModel.predict",
            f"Multivariate prediction for {len(self._models)} variates"
        )

        preds = []
        for idx, model in enumerate(self._models):
            self.logger.debug("ArimaModel.predict", f"Predicting for variate {idx}")
            prediction = model._predict(
                y_context=y_context[:, idx],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs
            )
            preds.append(prediction)
        result = np.hstack(preds)

        self.logger.debug("ArimaModel.predict", f"Prediction shape: {result.shape}")
        self.logger.info("ArimaModel.predict", "Prediction completed successfully")
        return result
