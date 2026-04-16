"""Tests for Lag-Llama multivariate + covariate column layout (no torch imports)."""

from __future__ import annotations

import numpy as np
import pytest

from tempus_bench.utils.lagllama_channel_fanout import variates_and_num_targets_for_predict


def test_variates_multivariate_only() -> None:
    y = np.zeros((10, 3), dtype=np.float64)
    v, m = variates_and_num_targets_for_predict(y, None)
    assert m == 3
    assert v.shape == (10, 3)
    assert v is y


def test_variates_targets_plus_covariates() -> None:
    y = np.ones((8, 2), dtype=np.float64)
    x = np.full((8, 4), 2.0, dtype=np.float64)
    v, m = variates_and_num_targets_for_predict(y, x)
    assert m == 2
    assert v.shape == (8, 6)
    assert np.array_equal(v[:, :2], y)
    assert np.array_equal(v[:, 2:], x)


def test_variates_rejects_steps_mismatch() -> None:
    y = np.zeros((5, 1), dtype=np.float64)
    x = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="same number of time steps"):
        variates_and_num_targets_for_predict(y, x)


def test_variates_rejects_non_2d_y() -> None:
    y = np.zeros((5,), dtype=np.float64)
    with pytest.raises(ValueError, match="y_context must be 2D"):
        variates_and_num_targets_for_predict(y, None)


def test_variates_rejects_non_2d_x() -> None:
    y = np.zeros((5, 1), dtype=np.float64)
    x = np.zeros(5, dtype=np.float64)
    with pytest.raises(ValueError, match="x_context must be 2D"):
        variates_and_num_targets_for_predict(y, x)
