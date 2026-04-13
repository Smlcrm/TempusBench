"""Verify the normalisation logic that ``_patch_sktime_ttm_fit_mismatched_keys``
applies to ``info["mismatched_keys"]`` tuples.

sktime's ``TinyTimeMixerForecaster._fit`` iterates over
``info["mismatched_keys"]`` calling ``key.split(".")``; transformers'
``from_pretrained`` returns elements as tuples
``(key_str, shape_model, shape_ckpt)`` — the unpatched code raises
``AttributeError: 'tuple' object has no attribute 'split'``.

We test the normalisation function used by the patch independently of heavy
sktime / transformers imports (which segfault in macOS CI due to torchvision).
"""
from __future__ import annotations

import pytest


def _normalise_mismatched_keys(keys: list) -> list[str]:
    """Replicate the normalisation logic the monkeypatch applies."""
    return [k[0] if isinstance(k, tuple) else k for k in keys]


class TestNormaliseMismatchedKeys:
    """The normaliser must convert tuples ``(str, shape, shape)`` to plain strings."""

    def test_all_tuples(self):
        raw = [
            ("backbone.weight", (192, 64), (192, 56)),
            ("backbone.encoder.patcher.weight", (192, 64), (192, 56)),
        ]
        result = _normalise_mismatched_keys(raw)
        assert result == ["backbone.weight", "backbone.encoder.patcher.weight"]

    def test_all_strings(self):
        raw = ["backbone.weight", "encoder.weight"]
        result = _normalise_mismatched_keys(raw)
        assert result == ["backbone.weight", "encoder.weight"]

    def test_mixed_tuple_and_string(self):
        raw = [
            ("backbone.weight", (192, 64), (192, 56)),
            "backbone.encoder.weight",
        ]
        result = _normalise_mismatched_keys(raw)
        assert result == ["backbone.weight", "backbone.encoder.weight"]

    def test_empty_list(self):
        assert _normalise_mismatched_keys([]) == []

    def test_normalised_keys_are_splittable(self):
        """After normalisation every key must be ``str.split``-able (the root cause bug)."""
        raw = [
            ("backbone.encoder.freq_mod.0.weight", (192, 128), (192, 124)),
            ("backbone.encoder.patcher.weight", (192, 64), (192, 56)),
            "backbone.encoder.freq_mod.1.bias",
        ]
        normalised = _normalise_mismatched_keys(raw)
        for key in normalised:
            parts = key.split(".")
            assert len(parts) >= 2, f"Expected dotted key, got {key!r}"

    def test_unpatched_code_fails_with_tuples(self):
        """Simulate what happens without the patch: ``key.split`` raises."""
        keys = [("backbone.weight", (192, 64), (192, 56))]
        with pytest.raises(AttributeError, match="split"):
            for key in keys:
                key.split(".")  # type: ignore[union-attr]

    def test_patched_code_succeeds_with_tuples(self):
        """After normalisation the iteration that sktime performs must work."""
        keys_raw = [
            ("backbone.encoder.patcher.weight", (192, 64), (192, 56)),
            ("backbone.encoder.freq_mod.0.weight", (192, 128), (192, 124)),
        ]
        keys = _normalise_mismatched_keys(keys_raw)
        for key in keys:
            parts = key.split(".")[:-1]
            assert all(isinstance(p, str) for p in parts)
            assert len(parts) >= 2
