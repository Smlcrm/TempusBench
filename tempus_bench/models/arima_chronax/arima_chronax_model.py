"""
Chronax ARIMA (JAX-based) model implementation.

This module provides a TempusBench wrapper around the JAX-accelerated
``chronax.models.ARIMA`` forecaster. The wrapper mirrors the conventions of
``tempus_bench/models/arima/arima_model.py`` so that Chronax models behave like
other TempusBench models:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax ARIMA is fit for each variate.
    * Optional past exogenous regressors (``x_context``) passed to
      ``ARIMA.fit(y, X=...)``.
    * Optional future exogenous regressors (``x_target``) passed to
      ``ARIMA.predict(h, X=...)``.

Chronax's ``ARIMA`` declares ``uses_exog = True`` and accepts ``X`` in both
``fit`` and ``predict``, so exogenous covariates are forwarded directly without
any shape or dtype guards beyond JAX float64 conversion.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, field_validator

from chronax.models import ARIMA

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxArimaHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.ARIMA`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    order: tuple[int, int, int] = Field(..., description="Non-seasonal ARIMA order (p, d, q)")
    seasonal_order: tuple[int, int, int] = Field(..., description="Seasonal ARIMA order (P, D, Q)")
    period: int = Field(..., ge=1, description="Seasonal period length (period=1 disables seasonal ARIMA)")
    include_mean: bool = Field(..., description="Whether to include intercept/drift term")
    method: Literal["CSS", "ML", "CSS-ML"] = Field(..., description="Optimization method")
    standardize: bool = Field(..., description="Whether to standardize data before fitting")
    alias: str = Field(default="ARIMA", description="Display name for the estimator (not tuned)")

    @field_validator("order", "seasonal_order", mode="before")
    @classmethod
    def _coerce_orders_tuple(cls, v: Any) -> tuple[int, int, int]:
        if isinstance(v, (list, tuple)):
            t = tuple(int(x) for x in v)
            if len(t) != 3:
                raise ValueError("order and seasonal_order must have exactly three integers (p,d,q) / (P,D,Q).")
            return t
        raise TypeError("order and seasonal_order must be a sequence of three integers.")


ChronaxArimaHyperparams.model_rebuild()


def _to_jax_array(x: Optional[np.ndarray]) -> Optional[jnp.ndarray]:
    """Convert an optional NumPy array to a float64 JAX array, preserving ``None``."""
    if x is None:
        return None
    return jnp.asarray(x, dtype=jnp.float64)


class ChronaxArimaModel(BaseModel):
    """
    Chronax ARIMA model for univariate and multivariate time series forecasting.

    Supports both standard ARIMA(p, d, q) and seasonal ARIMA by specifying a
    non-trivial ``seasonal_order`` and ``period``. Exogenous regressors for
    both the context (past) and target (future) horizons are forwarded to
    Chronax when provided, mirroring the behaviour of the statsmodels-based
    ``ArimaModel`` in this project.

    Attributes:
        order: tuple[int, int, int] - Non-seasonal ARIMA order (p, d, q)
        seasonal_order: tuple[int, int, int] - Seasonal ARIMA order (P, D, Q)
        period: int - Seasonal period length (period=1 disables seasonal ARIMA)
        include_mean: bool - Whether to include intercept/drift term
        method: str - Optimization method ("CSS", "ML", or "CSS-ML")
        alias: str - Display name for the estimator
        standardize: bool - Whether to standardize data before fitting

    The fitted per-target Chronax ARIMA instances are stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxArimaHyperparams)
        self._models: list[ARIMA] = []

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
    ) -> ARIMA:
        """
        Fit a single Chronax ARIMA model on one variate of the context data.

        Args:
            y_context (np.ndarray): Historical target values for a single variate,
                shape (num_steps_context, 1).
            y_target (np.ndarray): Future target values (unused; included for
                interface compatibility).
            timestamps_context (np.ndarray): Timestamps aligned with y_context (unused).
            timestamps_target (np.ndarray): Timestamps aligned with y_target (unused).
            x_context (Optional[np.ndarray]): Past exogenous regressors aligned
                with y_context.
            x_target (Optional[np.ndarray]): Future exogenous regressors (unused
                at fit time; included for interface consistency).

        Returns:
            ARIMA: The fitted chronax.models.ARIMA estimator.
        """
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        exog = _to_jax_array(x_context)

        model = ARIMA(
            order=tuple(self.order),
            seasonal_order=tuple(self.seasonal_order),
            period=int(self.period),
            include_mean=bool(self.include_mean),
            method=str(self.method),
            alias=str(self.alias),
            standardize=bool(self.standardize),
        )
        model.fit(endog, X=exog)
        return model

    @validate_inputs
    def _predict(
        self,
        arima_model: ARIMA,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Predict future values for a single variate using a fitted Chronax ARIMA.

        Args:
            arima_model (ARIMA): A fitted chronax.models.ARIMA estimator.
            y_context (np.ndarray): Past target values for a single variate
                (unused; present for interface consistency).
            timestamps_context (np.ndarray): Context timestamps (unused).
            timestamps_target (np.ndarray): Timestamps for the forecast horizon.
            x_context (Optional[np.ndarray]): Past exogenous regressors (unused).
            x_target (Optional[np.ndarray]): Future exogenous regressors for the
                forecast horizon (forwarded to Chronax when provided).

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, 1).

        Raises:
            ValueError: If the model is not fitted or the forecast horizon is empty.
        """
        if not self.is_fitted:
            raise ValueError("ChronaxArimaModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax ARIMA prediction."
            )

        forecast_steps = int(len(timestamps_target))
        use_covariates = bool(kwargs.pop("use_covariates", True))
        exog_future = _to_jax_array(x_target) if use_covariates else None

        output = arima_model.predict(h=forecast_steps, X=exog_future)
        y_pred = np.asarray(output["mean"], dtype=np.float64).reshape(-1, 1)
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
    ) -> "ChronaxArimaModel":
        """
        Train a separate Chronax ARIMA model for each variate in a (potentially
        multivariate) time series.

        Expects y_context and y_target as 2D arrays: (num_steps, num_targets).
        """
        num_targets = y_context.shape[1]
        self._models = []

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
        Predict future values for each variate and concatenate the results.

        Args:
            y_context (np.ndarray): Context values, shape (num_steps, num_variates).
            timestamps_context (np.ndarray): Timestamps for context data.
            timestamps_target (np.ndarray): Timestamps for target/future data.
            x_context (Optional[np.ndarray]): Optional covariate data for context.
            x_target (Optional[np.ndarray]): Optional covariate data for the
                prediction horizon.
            **kwargs: Additional keyword arguments (unused by Chronax ARIMA).

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, num_variates).

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("ChronaxArimaModel not fitted. Call train() first.")

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


class ArimaChronaxModel(ChronaxArimaModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
