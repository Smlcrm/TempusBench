"""
Seasonal Naive model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster
from pydantic import BaseModel as PydanticBaseModel, Field
from tempus_bench.models.base_model import BaseModel


class SeasonalNaiveParams(PydanticBaseModel):
    sp: int = Field(..., ge=1, description="Seasonal period")


class SeasonalNaiveModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Seasonal Naive model with model-specific parameters.
        """
        super().__init__(params, settings, SeasonalNaiveParams)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
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
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        sp = self.params.sp

        if not self.is_fitted:
            # Handle multivariate data by creating separate models for each feature
            if isinstance(y_context, np.ndarray) and y_context.ndim == 2:
                num_targets = y_context.shape[1]
                self._models = []
                for i in range(num_targets):
                    model = NaiveForecaster(strategy="last", sp=sp)
                    series_data = pd.Series(y_context[:, i])
                    model.fit(y=series_data, X=None)
                    self._models.append(model)
            else:
                # Handle univariate data
                if not isinstance(y_context, pd.Series):
                    y_context = pd.Series(y_context)
                self._model = NaiveForecaster(strategy="last", sp=sp)
                self._model.fit(y=y_context, X=None)

        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
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
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        sp = self.params.sp
        
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        # extract freq (validated by base pattern elsewhere if needed)
        forecast_horizon = len(timestamps_target)
        fh = np.arange(1, forecast_horizon + 1)
        
        # Handle multivariate data
        if hasattr(self, 'models') and self._models:
            # Multivariate case: predict for each feature
            num_targets = len(self._models)
            predictions = np.zeros((forecast_horizon, num_targets))
            
            for i, model in enumerate(self._models):
                pred = model.predict(fh=fh)
                if len(pred.shape) == 1:
                    pred = np.asarray(pred)
                predictions[:, i] = pred
        else:
            # Univariate case
            predictions = self._model.predict(fh=fh)
            if len(predictions.shape) == 1:
                predictions = np.asarray(predictions)
                predictions = np.expand_dims(predictions, axis=0)  # Make it (1, forecast_horizon)

        return predictions
