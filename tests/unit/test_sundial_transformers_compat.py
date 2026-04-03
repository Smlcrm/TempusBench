"""Unit tests for Sundial / transformers version gating (no weights required)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("version", "expect_mask"),
    [
        ("4.47.0", False),
        ("4.48.0", True),
        ("4.48.3", True),
        ("4.49.0", True),
        ("5.0.0", False),
        ("5.4.0", False),
    ],
)
def test_sundial_patch_attention_mask_gating(version: str, expect_mask: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempus_bench.models.sundial.sundial_model as sm

    monkeypatch.setattr(sm.transformers, "__version__", version)
    assert sm._sundial_use_patch_attention_mask() is expect_mask
