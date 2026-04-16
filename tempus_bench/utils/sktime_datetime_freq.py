"""Helpers for sktime ``ForecastingHorizon`` when series indices have no pandas ``freq``."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset


def infer_pandas_freq_offset_for_datetime_index(index: pd.DatetimeIndex):
    """Return a pandas offset so sktime ``ForecastingHorizon`` can call ``to_absolute``.

    NIFTY / stocks-style covariate series often use a ``DatetimeIndex`` with no
    ``.freq`` and no ``pd.infer_freq`` result (irregular gaps). sktime then leaves
    ``fh.freq`` as None and ``predict`` raises ``int * NoneType`` in ``_to_offset``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            "infer_pandas_freq_offset_for_datetime_index requires pd.DatetimeIndex, "
            f"got {type(index).__name__}"
        )
    if len(index) == 0:
        raise ValueError("Cannot infer frequency: DatetimeIndex is empty")
    if index.freq is not None:
        return to_offset(index.freq)
    if len(index) < 2:
        raise ValueError(
            "Cannot infer frequency: need at least 2 timestamps when index has no freq"
        )
    inferred = pd.infer_freq(index)
    if inferred is not None:
        return to_offset(inferred)
    median_delta = index.to_series().diff().median()
    if pd.isna(median_delta):
        raise ValueError("Cannot infer frequency: median timestamp delta is NaT")
    return to_offset(median_delta)
