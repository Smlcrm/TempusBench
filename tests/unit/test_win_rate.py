"""Tests for win-rate aggregators."""

import numpy as np
import pandas as pd
import pytest

from tempus_bench.aggregators.win_rate import WinRate, average_win_rate_across_metrics


def test_average_win_rate_across_metrics_balances_metrics() -> None:
    """Two metrics with opposite winners → each model averages 0.5."""
    mae = pd.DataFrame({"t1": [1.0, 5.0]}, index=["a", "b"])
    rmse = pd.DataFrame({"t1": [10.0, 1.0]}, index=["a", "b"])
    combined = average_win_rate_across_metrics({"mae": mae, "rmse": rmse})
    assert combined["a"] == pytest.approx(0.5)
    assert combined["b"] == pytest.approx(0.5)


def test_average_win_rate_across_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        average_win_rate_across_metrics({})


def test_win_rate_single_metric_matches_manual() -> None:
    pivot = pd.DataFrame({"t1": [1.0, 3.0], "t2": [2.0, 2.0]}, index=["a", "b"])
    s = WinRate(pivot)()
    # a wins t1 vs b, tie t2 → (1 + 0.5) / 2 comparisons on t1 + (0.5) on t2 for vs b only
    # Actually per task: for a, task t1: vs b one comparison, 1 win. task t2: vs b one comparison, tie 0.5
    # Total score 1.5, comparisons 2 → 0.75
    assert s["a"] == pytest.approx(0.75)
    assert s["b"] == pytest.approx(0.25)


def test_average_win_rate_skips_nan_metric_for_model() -> None:
    mae = pd.DataFrame({"t1": [1.0, 2.0]}, index=["a", "b"])
    rmse = pd.DataFrame({"t1": [np.nan, 1.0]}, index=["a", "b"])
    combined = average_win_rate_across_metrics({"mae": mae, "rmse": rmse})
    # a: mae win rate defined; rmse for a is all NaN comparisons → WinRate gives nan for a on rmse
    wr_mae = WinRate(mae)()
    assert not pd.isna(wr_mae["a"])
    wr_rmse = WinRate(rmse)()
    assert pd.isna(wr_rmse["a"])
    # average uses only mae for a
    assert combined["a"] == pytest.approx(float(wr_mae["a"]))
