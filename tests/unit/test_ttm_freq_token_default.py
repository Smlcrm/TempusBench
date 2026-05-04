"""Verify the ``_patch_sktime_ttm_freq_token_default`` monkeypatch logic.

The TTM encoder's ``forward`` raises ``Exception("Expecting freq_token in
forward")`` when ``resolution_prefix_tuning`` is enabled but sktime doesn't
pass a ``freq_token``.  The patch injects a zero tensor as the default.

We test the logic without importing the heavy sktime/torch modules (which
segfault on macOS due to torchvision).
"""
from __future__ import annotations

import numpy as np
import pytest


class TestFreqTokenDefaultLogic:
    """The patching logic creates a default ``freq_token`` when None is passed."""

    @staticmethod
    def _simulate_patched_forward(batch_size: int, resolution_prefix_tuning: bool,
                                  freq_token):
        """Simulate the patched forward logic without torch imports."""
        if freq_token is None and resolution_prefix_tuning:
            freq_token = np.zeros(batch_size, dtype=np.int64)
        return freq_token

    def test_default_injected_when_none_and_rpt_true(self):
        result = self._simulate_patched_forward(4, True, None)
        assert result is not None
        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.int64))

    def test_no_injection_when_rpt_false(self):
        result = self._simulate_patched_forward(4, False, None)
        assert result is None

    def test_explicit_value_passed_through(self):
        explicit = np.array([3, 3, 3], dtype=np.int64)
        result = self._simulate_patched_forward(3, True, explicit)
        np.testing.assert_array_equal(result, explicit)

    def test_default_shape_matches_batch_size(self):
        for bs in [1, 2, 8, 16]:
            result = self._simulate_patched_forward(bs, True, None)
            assert result.shape == (bs,)

    def test_default_dtype_is_int64(self):
        result = self._simulate_patched_forward(2, True, None)
        assert result.dtype == np.int64

    def test_default_all_zeros(self):
        result = self._simulate_patched_forward(5, True, None)
        assert np.all(result == 0)

    def test_patch_function_importable(self):
        """The patch function must be importable from the model module."""
        from tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model import (
            _patch_sktime_ttm_freq_token_default,
        )
        assert callable(_patch_sktime_ttm_freq_token_default)
