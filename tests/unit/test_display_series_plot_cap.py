"""Plot payload cap for display_series export (long tasks e.g. NIFTY)."""

from __future__ import annotations

from tempus_bench.utils.display_series import _plot_display_prefix_len


def test_plot_prefix_len_minimum_covers_one_benchmark_span() -> None:
    """At least context + 2 * forecast_horizon (context + train + validate at window 0)."""
    cw = 20
    fh = 8
    assert _plot_display_prefix_len(10_000, cw, fh, max_window_idx=0) == cw + 2 * fh


def test_plot_prefix_len_caps_at_series_length() -> None:
    assert _plot_display_prefix_len(50, 20, 8, max_window_idx=1023) == 50


def test_plot_prefix_len_nifty_style_order_of_magnitude() -> None:
    cw = 2048
    fh = 64
    cap = _plot_display_prefix_len(400_275, cw, fh, max_window_idx=1023)
    assert cap < 200_000
    assert cap == cw + fh * (2 + 1023)
