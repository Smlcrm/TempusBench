"""Regression: TTM R2.1 settings.yaml must use a synthetic FUSE path for local weights.

The r2.1 variant lives on a non-default branch of ibm-granite/granite-timeseries-ttm-r2.
A synthetic ``hf_model_name`` (e.g. *-r2-r2.1) gives it its own GCS directory so
``resolve_weights_path`` does not load the main-branch (r2) files.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _settings_path(model_name: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tempus_bench"
        / "models"
        / model_name
        / "settings.yaml"
    )


def test_ttm_r2_1_has_unique_hf_model_name() -> None:
    r2 = _settings_path("tiny_time_mixer_r2")
    r21 = _settings_path("tiny_time_mixer_r2_1")
    with open(r2) as f:
        d_r2 = yaml.safe_load(f)
    with open(r21) as f:
        d_r21 = yaml.safe_load(f)
    assert d_r21["hf_model_name"] != d_r2["hf_model_name"], (
        "TTM R2.1 must have a distinct hf_model_name from R2 so resolve_weights_path "
        "picks up the correct branch files from GCS FUSE."
    )


def test_ttm_r2_1_synthetic_name_contains_r2_1() -> None:
    p = _settings_path("tiny_time_mixer_r2_1")
    with open(p) as f:
        d = yaml.safe_load(f)
    assert "r2.1" in d["hf_model_name"] or "r2-r2.1" in d["hf_model_name"], (
        f"Synthetic name {d['hf_model_name']!r} should be identifiable as an r2.1 variant"
    )


def test_ttm_r2_main_branch_has_no_revision() -> None:
    p = _settings_path("tiny_time_mixer_r2")
    with open(p) as f:
        d = yaml.safe_load(f)
    assert d["hf_model_name"] == "ibm-granite/granite-timeseries-ttm-r2"
    assert d.get("revision") is None, "TTM R2 should use the default (main) branch"
