"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides an ARIMA model implementation for univariate time series forecasting.
ARIMA models combine autoregression, differencing, and moving average components to
capture temporal dependencies in time series data.

The model supports both seasonal and non-seasonal ARIMA variants and can handle
exogenous variables for enhanced forecasting performance.
"""

import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel

class ArimaParams(PydanticBaseModel):
    p: int = Field(..., ge=0, description="AR order (autoregressive)")
    d: int = Field(..., ge=0, description="Differencing order (integration)")
    q: int = Field(..., ge=0, description="MA order (moving average)")
    s: int = Field(..., ge=1, description="Seasonality period")

class ArimaModel(BaseModel):
    """
    ARIMA model for univariate time series forecasting.

    This class implements the ARIMA model with support for:
    - Non-seasonal ARIMA(p,d,q) models
    - Seasonal ARIMA(p,d,q)(P,D,Q,s) models
    - Exogenous variable support
    - Rolling window predictions
    - Model persistence and loading

    Attributes:
        p: AR order (autoregressive)
        d: Differencing order (integration)
        q: MA order (moving average)
        s: Seasonality period
        model_: Fitted ARIMA model instance
        loss_function: Loss function for training
    """

    def __init__(self, config: JobConfig, logs_path: str):
        """
        Initialize ARIMA model with given configuration.

        Args:
            config: JobConfig instance containing model and task configuration
            logs_path: Directory for storing log files (required)
        """
        super().__init__(config, logs_path, ArimaParams)
        self.model_ = None

    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ArimaModel":
        """
        Train the ARIMA model on given data.

        Args:
            y_context: Past target values - training data (required)
            y_target: Future target values (not used in training, for compatibility)
            timestamps_context: Timestamps for y_context (not used in ARIMA)
            timestamps_target: Timestamps for y_target (not used in ARIMA)
            freq: Frequency string (required by interface, not used in ARIMA)

        Returns:
            self: The fitted model instance

        Note:
            ARIMA models only use y_context for training.
            y_target, timestamps_context, timestamps_target, and freq are ignored to prevent data leakage.
        """

        self.logger.debug("ArimaModel._train", f"Starting training with input shapes: y_context={y_context.shape}, y_target={y_target.shape if y_target is not None else None}")
        self.logger.debug("ArimaModel._train", f"Timestamps context range: {timestamps_context[0] if len(timestamps_context) > 0 else 'empty'} to {timestamps_context[-1] if len(timestamps_context) > 0 else 'empty'}")
        self.logger.debug("ArimaModel._train", f"Timestamps target range: {timestamps_target[0] if len(timestamps_target) > 0 else 'empty'} to {timestamps_target[-1] if len(timestamps_target) > 0 else 'empty'}")
        self.logger.debug("ArimaModel._train", f"Frequency parameter: {freq}")

        # Ensure endogenous series is 1D for statsmodels
        endog = y_context.squeeze()

        self.logger.debug("ArimaModel._train", f"Endogenous series shape after squeeze: {endog.shape}")
        self.logger.debug("ArimaModel._train", f"Endogenous series sample values: {endog[:5] if len(endog) >= 5 else endog}")

        # Use seasonal_order only if seasonal period is greater than 1
        if self.model_config["s"] > 1:

            self.logger.debug("ArimaModel._train", f"Using seasonal ARIMA with seasonal period s={self.model_config['s']}")
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config["p"],
                    self.model_config["d"],
                    self.model_config["q"],
                ),
                seasonal_order=(0, 0, 0, self.model_config["s"]),
                exog=None,
            )
        else:
            # Non-seasonal ARIMA
            self.logger.debug("ArimaModel._train", f"Using non-seasonal ARIMA")
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config["p"],
                    self.model_config["d"],
                    self.model_config["q"],
                ),
                exog=None,
            )

        self.logger.debug("ArimaModel._train", f"Model order parameters: {self.model_config}")
        self.logger.debug("ArimaModel._train", f"Starting model fitting...")

        self.model_ = model.fit()

        self.logger.debug("ArimaModel._train", f"Model fitted successfully")
        self.logger.debug("ArimaModel._train", f"Model summary: {self.model_.summary()}")
        self.logger.debug("ArimaModel._train", f"Model AIC: {self.model_.aic}, BIC: {self.model_.bic}")
        self.logger.debug("ArimaModel._train", f"Model coefficients: {self.model_.params}")
        self.logger.info("ArimaModel._train", "Training completed successfully")

        self.is_fitted = True
        return self

    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Make predictions using the trained ARIMA model, rolling forward using the fitted model.

        Args:
            y_context: Recent/past target values (not used for ARIMA prediction)
            timestamps_context: Timestamps for y_context (not used for ARIMA prediction)
            timestamps_target: Timestamps for the prediction horizon (used to determine forecast length)
            freq: Frequency string (must be provided from CSV data, required)

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If model is not fitted, freq is not provided, or forecast length cannot be determined
        """

        self.logger.debug("ArimaModel._predict", f"Starting prediction with context shape: {y_context.shape}")
        self.logger.debug("ArimaModel._predict", f"Timestamps context: {timestamps_context[0] if len(timestamps_context) > 0 else 'empty'} to {timestamps_context[-1] if len(timestamps_context) > 0 else 'empty'}")
        self.logger.debug("ArimaModel._predict", f"Timestamps target: {timestamps_target[0] if len(timestamps_target) > 0 else 'empty'} to {timestamps_target[-1] if len(timestamps_target) > 0 else 'empty'}")

        if not self.is_fitted:
            raise ValueError("ArimaModel not fitted. Call train() first.")

        if freq is None or freq == "":
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )

        if timestamps_target is None:
            raise ValueError(
                "timestamps_target must be provided to determine forecast horizon for ARIMA."
            )
        forecast_steps = len(timestamps_target)
        if forecast_steps <= 0:
            raise ValueError(
                "Forecast horizon must be positive (timestamps_target must be non-empty)."
            )

        self.logger.debug("ArimaModel._predict", f"Frequency validation passed: {freq}")
        self.logger.debug("ArimaModel._predict", f"Forecast steps calculated from timestamps_target: {forecast_steps}")
        self.logger.debug("ArimaModel._predict", f"Generating forecast...")

        forecast = self.model_.forecast(steps=forecast_steps, exog=None)
        forecast_array = np.asarray(forecast)

        self.logger.debug("ArimaModel._predict", f"Forecast values generated: {forecast_array}")
        self.logger.debug("ArimaModel._predict", f"Forecast array shape: {forecast_array.shape}")

        y_pred = forecast_array.reshape(-1, 1)

        self.logger.debug("ArimaModel._predict", f"Final prediction shape: {y_pred.shape}")
        self.logger.info("ArimaModel._predict", "Prediction completed successfully")

        return y_pred

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ArimaModel":
        """
        Anyvariate wrapper: trains a separate ARIMA per variate if multivariate,
        or a single ARIMA in the univariate case.

        Assumes y_context and y_target are 2D ndarrays: (num_steps, num_targets), even for univariate.
        """
        num_targets = y_context.shape[1]


        self.logger.debug("ArimaModel.train", f"Number of features/variates detected: {num_targets}")

        # Multivariate: more than one feature (column)
        if num_targets > 1:

            self.logger.debug("ArimaModel.train", "Taking multivariate path - training separate ARIMA per variate")
            self.models = []
            for k in range(num_targets):

                self.logger.debug("ArimaModel.train", f"Training variate k={k}")
                yc = y_context[:, k]    # Already 1D
                yt = y_target[:, k] if y_target is not None else None  # Already 1D
                # No need to reshape to 2D column; _train can handle 1D array for this variate
                m = ArimaModel(self.config, logs_path=self.logs_path)
                m._train(
                    y_context=yc,
                    y_target=yt,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                )
                self.models.append(m)
            # For compatibility, mirror first model state to top-level attributes
            self.model_ = self.models[0].model_
            self.is_fitted = True

            self.logger.info("ArimaModel.train", f"Multivariate training completed for {num_targets} variates")
            return self
        else:
            # Univariate: input is always (num_steps, 1)
            self.logger.debug("ArimaModel.train", "Taking univariate path")
            return self._train(
                y_context=y_context,
                y_target=y_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
            )

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Anyvariate wrapper: predicts per variate and stacks columns.
        """
        if hasattr(self, "models") and self.models:

            self.logger.debug("ArimaModel.predict", f"Multivariate prediction for {len(self.models)} variates")
            preds = []
            num_variates = len(self.models)
            for k, m in enumerate(self.models):

                self.logger.debug("ArimaModel.predict", f"Predicting for variate k={k}")
                yc = y_context[:, k] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(
                    y_context=yc,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                )
                preds.append(pk.reshape(-1, 1))
            result = np.concatenate(preds, axis=1)

            self.logger.debug("ArimaModel.predict", f"Final concatenated prediction shape: {result.shape}")
            self.logger.info("ArimaModel.predict", "Multivariate prediction completed successfully")
            return result
        # Univariate

        self.logger.debug("ArimaModel.predict", "Univariate prediction")
        return self._predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            freq=freq,
        )
