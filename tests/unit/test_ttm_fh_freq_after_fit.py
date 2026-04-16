"""Regression: TTM R1/R2 predict() must re-inject fh.freq after fit() to avoid
TypeError ('int' * NoneType) in sktime _predict → to_absolute → _to_offset.

The monkeypatch for sktime's _predict uses ``freq=self.fh.freq``. If sktime
normalises the stored FH and loses ``.freq`` during fit(), predict() must
restore it before calling ``self._model.predict()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import MonthEnd


@pytest.fixture()
def _minimal_ttm_model():
    """Build a TinyTimeMixerR1Model with patched dependencies to avoid loading real weights."""
    with (
        patch(
            "tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model._get_ttm_context_length",
            return_value=512,
        ),
        patch(
            "tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model._patch_transformers_tiny_time_mixer_tied_weights",
        ),
        patch(
            "tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model._patch_sktime_ttm_predict_forecasting_horizon_freq",
        ),
        patch(
            "tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model.TinyTimeMixerForecaster",
        ) as MockForecaster,
    ):
        mock_instance = MagicMock()
        mock_fh = MagicMock()
        mock_fh.freq = None
        mock_instance.fh = mock_fh
        mock_instance.predict.return_value = pd.DataFrame(
            np.zeros((12, 2)), columns=[0, 1]
        )
        MockForecaster.return_value = mock_instance

        from tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model import (
            TinyTimeMixerR1Model,
        )

        settings = {
            "hf_model_name": "ibm-granite/granite-timeseries-ttm-r2",
            "revision": "main",
        }
        model = TinyTimeMixerR1Model(params={}, settings=settings)
        model.is_fitted = True
        yield model, mock_instance, mock_fh


def test_predict_reinjects_fh_freq_when_none(_minimal_ttm_model) -> None:
    model, mock_forecaster, mock_fh = _minimal_ttm_model
    n_context = 100
    n_targets = 2
    forecast_horizon = 12
    y = np.random.randn(n_context, n_targets)
    base = pd.Timestamp("2020-01-01")
    ts_ctx = np.array([base + pd.DateOffset(months=i) for i in range(n_context)])
    ts_tgt = np.array(
        [base + pd.DateOffset(months=n_context + i) for i in range(forecast_horizon)]
    )
    model.predict(y, ts_ctx, ts_tgt)

    assert mock_fh.freq is not None, (
        "predict() must re-inject fh.freq when sktime normalises it to None after fit()"
    )
