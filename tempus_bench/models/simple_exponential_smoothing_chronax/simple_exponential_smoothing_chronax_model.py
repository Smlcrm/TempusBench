"""
Chronax SimpleExponentialSmoothing model implementation.

This module provides a TempusBench wrapper around
``chronax.models.SimpleExponentialSmoothing``. Mirrors the conventions of
``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax SimpleExponentialSmoothing is fit for each variate.
    * Chronax SimpleExponentialSmoothing declares ``uses_exog = False``;
      exogenous covariates are accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from chronax.models import SimpleExponentialSmoothing

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxSimpleExponentialSmoothingHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.SimpleExponentialSmoothing`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    alpha: float = Field(..., ge=0, le=1, description="Smoothing parameter")


ChronaxSimpleExponentialSmoothingHyperparams.model_rebuild()


class ChronaxSimpleExponentialSmoothingModel(BaseModel):
    """
    Chronax SimpleExponentialSmoothing model for univariate/multivariate forecasting.

    For multivariate series (num_targets > 1), a separate Chronax
    SimpleExponentialSmoothing is fit for each target. The fitted instances are
    stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxSimpleExponentialSmoothingHyperparams)
        self._models: list[SimpleExponentialSmoothing] = []

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
    ) -> SimpleExponentialSmoothing:
        """Fit a single Chronax SimpleExponentialSmoothing on one variate."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        model = SimpleExponentialSmoothing(alpha=float(self.alpha))
        model.fit(endog)
        return model

    @validate_inputs
    def _predict(
        self,
        simple_exponential_smoothing_model: SimpleExponentialSmoothing,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax SimpleExponentialSmoothing."""
        if not self.is_fitted:
            raise ValueError(
                "ChronaxSimpleExponentialSmoothingModel not fitted. Call train() first."
            )

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax SimpleExponentialSmoothing prediction."
            )

        forecast_steps = int(len(timestamps_target))
        output = simple_exponential_smoothing_model.predict(h=forecast_steps)
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
    ) -> "ChronaxSimpleExponentialSmoothingModel":
        """Train a separate Chronax SimpleExponentialSmoothing for each variate."""
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
        """Predict future values for each variate and concatenate the results."""
        if not self.is_fitted:
            raise ValueError(
                "ChronaxSimpleExponentialSmoothingModel not fitted. Call train() first."
            )

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                simple_exponential_smoothing_model=fitted_model,
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


class SimpleExponentialSmoothingChronaxModel(ChronaxSimpleExponentialSmoothingModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
