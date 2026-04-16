"""Covariate CSV timestamps must be ISO 8601 strings; loader uses pandas only."""

import pandas as pd

from tempus_bench.pipeline.data_loader import _infer_freq_from_calendar_index


def test_iso_yearly_strings_infer_annual():
    ts = pd.DatetimeIndex(
        pd.to_datetime([f"{y}-01-01T00:00:00Z" for y in (1980, 1981, 1982)], utc=True)
    )
    assert ts[0].year == 1980
    inf = _infer_freq_from_calendar_index(ts)
    assert inf is not None
    assert "Y" in inf or inf in ("YE", "YS", "YS-JAN")


def test_iso_weekly_strings_infer_weekly():
    ts = pd.DatetimeIndex(
        pd.to_datetime(
            ["2010-02-05T00:00:00Z", "2010-02-12T00:00:00Z", "2010-02-19T00:00:00Z"],
            utc=True,
        )
    )
    inf = _infer_freq_from_calendar_index(ts)
    assert inf is not None and inf.startswith("W")
