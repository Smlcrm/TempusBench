"""Tests for Granite TTM max-context alignment (history tail truncation)."""

from __future__ import annotations

import numpy as np
import pytest

from tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model import (
    TTM_MAX_CONTEXT_LENGTH,
    _truncate_ttm_aligned_history,
)


def test_no_op_when_short() -> None:
    y = np.zeros((100, 2), dtype=np.float64)
    ts = np.arange(100, dtype=np.int64)
    y2, ts2, x2 = _truncate_ttm_aligned_history(y, ts, None, max_len=1536)
    assert y2.shape == (100, 2)
    assert ts2.shape == (100,)
    assert x2 is None


def test_truncates_tail_preserves_width() -> None:
    n = TTM_MAX_CONTEXT_LENGTH + 500
    y = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    ts = np.arange(n, dtype=np.int64)
    cov = np.ones((n, 5), dtype=np.float64)
    y2, ts2, x2 = _truncate_ttm_aligned_history(
        y, ts, cov, max_len=TTM_MAX_CONTEXT_LENGTH
    )
    assert y2.shape == (TTM_MAX_CONTEXT_LENGTH, 3)
    assert ts2.shape == (TTM_MAX_CONTEXT_LENGTH,)
    assert x2 is not None
    assert x2.shape == (TTM_MAX_CONTEXT_LENGTH, 5)
    np.testing.assert_array_equal(y2, y[-TTM_MAX_CONTEXT_LENGTH:])
    np.testing.assert_array_equal(ts2, ts[-TTM_MAX_CONTEXT_LENGTH:])


def test_mismatched_x_length_raises() -> None:
    y = np.zeros((10, 1))
    ts = np.arange(10)
    x = np.zeros((5, 1))
    with pytest.raises(ValueError):
        _truncate_ttm_aligned_history(y, ts, x, max_len=8)
