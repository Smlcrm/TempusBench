"""Regression: TabPFNTSPipeline constructor kwargs must match the pinned tabpfn-time-series API."""

from __future__ import annotations

import inspect

import pytest


def test_tabpfn_ts_pipeline_init_accepts_tabpfn_mode() -> None:
    pytest.importorskip("tabpfn_time_series")
    from tabpfn_time_series import TabPFNTSPipeline

    params = inspect.signature(TabPFNTSPipeline.__init__).parameters
    assert "tabpfn_mode" in params


def test_tabpfn_ts_pipeline_instantiation_kwargs_match_model() -> None:
    """Mirrors filtering in tabpfn_ts_model.predict — no TypeError from extra kwargs."""
    pytest.importorskip("tabpfn_time_series")
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

    pipeline_kw: dict = {"tabpfn_mode": TabPFNMode.LOCAL}
    if "ignore_pretraining_limits" in inspect.signature(
        TabPFNTSPipeline.__init__
    ).parameters:
        pipeline_kw["ignore_pretraining_limits"] = True
    TabPFNTSPipeline(**pipeline_kw)
