"""
Seasonal Naive model implementation.
"""

import numpy as np
import pandas as pd

from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field
from sktime.forecasting.naive import NaiveForecaster

from tempus_bench.models.base_model import BaseModel, validate_inputs


class SeasonalNaiveHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    sp: int = Field(..., ge=1, description="Seasonal period")


class SeasonalNaiveModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Seasonal Naive model with model-specific parameters.
        """
        super().__init__(params, settings, SeasonalNaiveHyperparams)

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
    ) -> "SeasonalNaiveModel":
        """
        Train the Seasonal Naive model on given data. For this model, "training"
        simply means storing the historical data for future lookups.

        Args:
            y_context: Past target values (pd.Series or np.ndarray).
            y_target: Future target values (not used by this model, but included for compatibility).
            timestamps_context: Timestamps for y_context (not used).
            timestamps_target: Timestamps for y_target (not used).
            **kwargs: Additional keyword arguments.

        Returns:
            self: The fitted model instance.
        """

        sp = self.sp

        if not self.is_fitted:
            num_targets = y_context.shape[1]
            self._models = []
            for i in range(num_targets):
                model = NaiveForecaster(strategy="last", sp=sp)
                series_data = pd.Series(y_context[:, i])
                model.fit(y=series_data, X=None)
                self._models.append(model)

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
    ):
        """
        Make predictions using the trained Seasonal Naive model.

        Args:
            y_context: Context time series values (pd.Series or np.ndarray).
            timestamps_context: Timestamps for context data.
            timestamps_target: Timestamps for target data.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Model predictions with shape (num_series, forecast_horizon).
        """

        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        # extract freq (validated by base pattern elsewhere if needed)
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        fh = np.arange(1, forecast_horizon + 1)

        predictions = np.zeros((forecast_horizon, num_targets))

        for i, model in enumerate(self._models):
            pred = model.predict(fh=fh)
            if len(pred.shape) == 1:
                pred = np.asarray(pred)
            predictions[:, i] = pred
        return predictions
