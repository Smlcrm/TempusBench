"""Regression: sktime TTM ``_predict`` else-branch must pass ``freq`` to ``ForecastingHorizon``."""

from __future__ import annotations

from tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model import (
    _fix_sktime_ttm_predict_source_for_tests,
)


def test_fix_inserts_freq_on_else_branch_index() -> None:
    sample = """
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1))
                .to_absolute(self._cutoff)
                ._values
            )
"""
    out, n = _fix_sktime_ttm_predict_source_for_tests(sample)
    assert n == 1
    assert "ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)" in out
    assert (
        out.count("ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)")
        == 1
    )


def test_fix_idempotent_on_already_patched_snippet() -> None:
    sample = """
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values
            )
"""
    _out, n = _fix_sktime_ttm_predict_source_for_tests(sample)
    assert n == 0
