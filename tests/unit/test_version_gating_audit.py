"""TT03: version-gating helpers stay importable (extend with more models when needed)."""

from __future__ import annotations

import tempus_bench.models.sundial.sundial_model as sm


def test_sundial_attention_mask_gating_exists() -> None:
    assert callable(sm._sundial_use_patch_attention_mask)
