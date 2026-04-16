"""TTM / sktime: ForecastingHorizon needs a non-null ``freq`` for irregular covariate indices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas.tseries.frequencies import to_offset

from tempus_bench.utils.sktime_datetime_freq import (
    infer_pandas_freq_offset_for_datetime_index,
)


def test_infer_uses_index_freq_when_set() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    off = infer_pandas_freq_offset_for_datetime_index(idx)
    assert off is not None
    assert off == to_offset("h")


def test_infer_uses_pd_infer_freq_when_regular_daily() -> None:
    idx = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
        freq=None,
    )
    off = infer_pandas_freq_offset_for_datetime_index(idx)
    assert off is not None


def test_infer_median_step_when_irregular_like_nifty_minutes() -> None:
    """Gaps break ``pd.infer_freq``; median delta still yields a usable offset."""
    idx = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:40:00",
        ]
    )
    assert pd.infer_freq(idx) is None
    off = infer_pandas_freq_offset_for_datetime_index(idx)
    assert off is not None


def test_infer_raises_on_empty_index() -> None:
    idx = pd.DatetimeIndex([])
    with pytest.raises(ValueError, match="empty"):
        infer_pandas_freq_offset_for_datetime_index(idx)


def test_infer_raises_on_single_timestamp() -> None:
    idx = pd.to_datetime(["2024-01-01"])
    with pytest.raises(ValueError, match="at least 2"):
        infer_pandas_freq_offset_for_datetime_index(idx)


def test_forecasting_horizon_has_freq_with_inferred_offset() -> None:
    sktime = pytest.importorskip("sktime")
    ForecastingHorizon = sktime.forecasting.base.ForecastingHorizon

    idx = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:35:00",
        ]
    )
    freq_offset = infer_pandas_freq_offset_for_datetime_index(
        pd.DatetimeIndex(idx)
    )
    fh = ForecastingHorizon(
        np.arange(1, 4, dtype=np.int64),
        is_relative=True,
        freq=freq_offset,
    )
    assert fh.freq is not None
