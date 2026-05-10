"""
XGBoost model implementation for time series forecasting using skforecast.

Univariate  (1 target) : ForecasterRecursive
Multivariate (N targets): ForecasterDirectMultiVariate, one instance per target column
Covariates             : exog parameter passed to fit() and predict()
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor

from tempus_bench.models.base_model import BaseModel, validate_inputs


class XgboostHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    n_estimators: int = Field(..., ge=1, description="Number of boosting rounds")
    max_depth: int = Field(..., ge=1, description="Maximum tree depth")
    learning_rate: float = Field(..., gt=0, description="Learning rate")
    # Fixed Hyperparameters - Optional for User to override
    reg_alpha: float = Field(
        default=0.0, ge=0, description="L1 regularization strength (optional)"
    )
    reg_lambda: float = Field(
        default=1.0, ge=0, description="L2 regularization strength (optional)"
    )
    subsample: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Subsample ratio of the training instances (optional)",
    )
    colsample_bytree: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Subsample ratio of columns for each tree (optional)",
    )
    min_child_weight: int = Field(
        default=1,
        ge=1,
        description="Minimum sum of instance weight needed in a child (optional)",
    )
    gamma: float = Field(
        default=0.0,
        ge=0,
        description="Minimum loss reduction required to make a further partition (optional)",
    )


class XgboostModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, XgboostHyperparams)
        self._n_targets: int = 0
        self._n_covariates: int = 0
        self._lags: int = 0
        self._forecast_horizon: int = 0
        self._col_names: List[str] = []
        self._cov_names: List[str] = []
        # Univariate forecaster (used when n_targets == 1)
        self._forecaster: Optional[ForecasterRecursive] = None
        # Multivariate forecasters (used when n_targets > 1), one per target column
        self._forecasters: List[ForecasterDirectMultiVariate] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_regressor(self) -> XGBRegressor:
        """Build a fresh XGBRegressor from current hyperparams."""
        valid_keys = set(XGBRegressor().get_params().keys())
        all_params = {
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "reg_alpha": float(self.reg_alpha),
            "reg_lambda": float(self.reg_lambda),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "min_child_weight": int(self.min_child_weight),
            "gamma": float(self.gamma),
        }
        return XGBRegressor(**{k: v for k, v in all_params.items() if k in valid_keys})

    @staticmethod
    def _build_index(timestamps: np.ndarray, freq: Optional[str] = None) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(timestamps.flatten())
        # Always infer freq from the actual timestamps — the kwarg freq may use a
        # different alias (e.g. 'ME' vs 'MS') that conflicts with the real data.
        if idx.freq is None:
            inferred = pd.infer_freq(idx)
            if inferred is not None:
                try:
                    idx.freq = pd.tseries.frequencies.to_offset(inferred)
                except Exception:
                    pass
        return idx

    @staticmethod
    def _to_series(arr: np.ndarray, index: pd.DatetimeIndex, name: str) -> pd.Series:
        return pd.Series(arr.flatten(), index=index, name=name)

    @staticmethod
    def _to_df(
        arr: np.ndarray, index: pd.DatetimeIndex, columns: List[str]
    ) -> pd.DataFrame:
        return pd.DataFrame(arr, index=index, columns=columns)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "XgboostModel":
        """
        Train the XGBoost model using skforecast.

        Univariate  (1 target) : ForecasterRecursive with XGBRegressor.
        Multivariate (N targets): One ForecasterDirectMultiVariate per target, each
                                  receiving all N series so cross-target patterns are
                                  captured as features.
        Covariates             : Concatenated context+target covariates passed as exog.
        """
        n_targets = int(y_context.shape[1])
        forecast_horizon = int(y_target.shape[0])
        # Cap lags to available context length so at least one window is usable
        lags = max(1, min(int(self.lookback_window), int(y_context.shape[0])))

        # Full training series: context + target
        freq = kwargs.get("freq", None)
        y_full = np.concatenate([y_context, y_target], axis=0)
        ts_full = np.concatenate([timestamps_context, timestamps_target], axis=0)
        idx_full = self._build_index(ts_full, freq=freq)

        # Exogenous variables (covariates): past + future combined for training
        exog_full: Optional[pd.DataFrame] = None
        cov_names: List[str] = []
        if x_context is not None and x_target is not None:
            x_full = np.concatenate([x_context, x_target], axis=0)
            cov_names = [f"cov_{i}" for i in range(x_full.shape[1])]
            exog_full = self._to_df(x_full, idx_full, cov_names)

        self._n_targets = n_targets
        self._n_covariates = len(cov_names)
        self._lags = lags
        self._forecast_horizon = forecast_horizon
        self._cov_names = cov_names

        if n_targets == 1:
            # ---- Univariate: ForecasterRecursive ----
            self._col_names = ["target_0"]
            y_series = self._to_series(y_full[:, 0], idx_full, "target_0")
            self._forecaster = ForecasterRecursive(
                estimator=self._make_regressor(),
                lags=lags,
            )
            self._forecaster.fit(y=y_series, exog=exog_full)
        else:
            # ---- Multivariate: ForecasterDirectMultiVariate per target ----
            col_names = [f"target_{i}" for i in range(n_targets)]
            self._col_names = col_names
            series_df = self._to_df(y_full, idx_full, col_names)
            self._forecasters = []
            for col in col_names:
                f = ForecasterDirectMultiVariate(
                    estimator=self._make_regressor(),
                    level=col,
                    lags=lags,
                    steps=forecast_horizon,
                )
                f.fit(series=series_df, exog=exog_full)
                self._forecasters.append(f)

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
        **kwargs,
    ) -> np.ndarray:
        """
        Predict future values using the fitted skforecast forecaster.

        Args:
            y_context: Observed history, shape (n_steps, n_targets).
            timestamps_context: Timestamps aligned with y_context.
            timestamps_target: Timestamps for the forecast horizon.
            x_context: Past covariates aligned with y_context (unused at predict time,
                       only x_target is needed by skforecast).
            x_target: Future covariates aligned with timestamps_target.

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, n_targets).
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        forecast_horizon = int(timestamps_target.shape[0])
        freq = kwargs.get("freq", None)
        idx_ctx = self._build_index(timestamps_context, freq=freq)
        idx_tgt = self._build_index(timestamps_target, freq=freq)

        # Future exog covers only the forecast horizon (skforecast requirement)
        exog_future: Optional[pd.DataFrame] = None
        if self._n_covariates > 0:
            if x_target is None:
                raise ValueError(
                    "Model was trained with covariates; x_target is required for predict()."
                )
            exog_future = self._to_df(x_target, idx_tgt, self._cov_names)

        if self._n_targets == 1:
            # ---- Univariate ----
            last_window = self._to_series(y_context[:, 0], idx_ctx, "target_0")
            preds = self._forecaster.predict(
                steps=forecast_horizon,
                last_window=last_window,
                exog=exog_future,
            )
            return preds.values.reshape(forecast_horizon, 1)

        # ---- Multivariate ----
        last_window_df = self._to_df(y_context, idx_ctx, self._col_names)
        results = []
        for f in self._forecasters:
            preds = f.predict(
                steps=forecast_horizon,
                last_window=last_window_df,
                exog=exog_future,
            )
            # predict() returns a DataFrame with columns ['level', 'pred'] in skforecast 0.22+
            results.append(preds["pred"].to_numpy(dtype=float))

        # Each entry is (forecast_horizon,); stack to (forecast_horizon, n_targets)
        return np.stack(results, axis=1)
