"""Tests for Granite TTM max-context alignment (history tail truncation)."""

from __future__ import annotations

import numpy as np
import pytest

import tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model as ttm_model
from tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model import (
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
    max_len = 1536
    n = max_len + 500
    y = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    ts = np.arange(n, dtype=np.int64)
    cov = np.ones((n, 5), dtype=np.float64)
    y2, ts2, x2 = _truncate_ttm_aligned_history(
        y, ts, cov, max_len=max_len
    )
    assert y2.shape == (max_len, 3)
    assert ts2.shape == (max_len,)
    assert x2 is not None
    assert x2.shape == (max_len, 5)
    np.testing.assert_array_equal(y2, y[-max_len:])
    np.testing.assert_array_equal(ts2, ts[-max_len:])


def test_mismatched_x_length_raises() -> None:
    y = np.zeros((10, 1))
    ts = np.arange(10)
    x = np.zeros((5, 1))
    with pytest.raises(ValueError):
        _truncate_ttm_aligned_history(y, ts, x, max_len=8)


def test_get_ttm_context_length_uses_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    ttm_model._get_ttm_context_length.cache_clear()
    try:

        def fake_loader(model_path: str, revision: str) -> int:
            assert model_path == "dummy/repo"
            assert revision == "main"
            return 1024

        monkeypatch.setattr(
            ttm_model,
            "_load_ttm_context_length_from_pretrained",
            fake_loader,
        )
        assert ttm_model._get_ttm_context_length("dummy/repo", "main") == 1024
    finally:
        ttm_model._get_ttm_context_length.cache_clear()


def test_get_ttm_context_length_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ttm_model._get_ttm_context_length.cache_clear()
    try:
        monkeypatch.setattr(
            ttm_model,
            "_load_ttm_context_length_from_pretrained",
            lambda _p, _r: 0,
        )
        with pytest.raises(ValueError, match="invalid context_length"):
            ttm_model._get_ttm_context_length("dummy/repo", "main")
    finally:
        ttm_model._get_ttm_context_length.cache_clear()
