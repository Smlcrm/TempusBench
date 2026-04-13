"""Tests for ModelCapabilities.single_target_covariate field.

Regression test for the crash that occurred when varmax/settings.yaml
defined ``single_target_covariate: true`` inside the ``capabilities``
block but ModelCapabilities (extra="forbid") did not declare the field.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tempus_bench.utils.model_settings import (
    ModelCapabilities,
    clear_model_settings_cache,
    get_covariate_support,
    get_past_only_covariate_models,
    load_capabilities_for_model,
    parse_capabilities_from_settings,
)


class TestSingleTargetCovariateField:
    """ModelCapabilities must accept the optional ``single_target_covariate`` boolean."""

    def test_accepts_single_target_covariate_true(self) -> None:
        raw = {
            "capabilities": {
                "covariates": "past_only",
                "univariate": False,
                "multivariate": True,
                "single_target_covariate": True,
            },
        }
        cap = parse_capabilities_from_settings(raw, model_name="varmax")
        assert cap.single_target_covariate is True

    def test_accepts_single_target_covariate_false(self) -> None:
        raw = {
            "capabilities": {
                "covariates": "none",
                "univariate": True,
                "multivariate": False,
                "single_target_covariate": False,
            },
        }
        cap = parse_capabilities_from_settings(raw, model_name="test_explicit_false")
        assert cap.single_target_covariate is False

    def test_defaults_to_false_when_omitted(self) -> None:
        raw = {
            "capabilities": {
                "covariates": "past_future",
                "univariate": True,
                "multivariate": True,
            },
        }
        cap = parse_capabilities_from_settings(raw, model_name="test_default")
        assert cap.single_target_covariate is False

    def test_rejects_unknown_extra_field(self) -> None:
        raw = {
            "capabilities": {
                "covariates": "none",
                "univariate": True,
                "multivariate": False,
                "bogus_field": 42,
            },
        }
        with pytest.raises(ValueError, match="bogus_field"):
            parse_capabilities_from_settings(raw, model_name="test_bogus")

    def test_rejects_non_bool_single_target_covariate(self) -> None:
        raw = {
            "capabilities": {
                "covariates": "past_only",
                "univariate": False,
                "multivariate": True,
                "single_target_covariate": [1, 2],
            },
        }
        with pytest.raises(ValueError, match="single_target_covariate"):
            parse_capabilities_from_settings(raw, model_name="test_non_bool")


class TestVarmaxCapabilitiesIntegration:
    """The real varmax settings.yaml must load without error."""

    def setup_method(self) -> None:
        clear_model_settings_cache()

    def test_varmax_capabilities_load(self) -> None:
        cap = load_capabilities_for_model("varmax")
        assert cap.covariates == "past_only"
        assert cap.multivariate is True
        assert cap.univariate is False
        assert cap.single_target_covariate is True

    def test_varmax_appears_in_past_only_set(self) -> None:
        past_only = get_past_only_covariate_models()
        assert "varmax" in past_only

    def test_get_covariate_support_varmax(self) -> None:
        assert get_covariate_support("varmax") == "past_only"


class TestAllModelsCapabilitiesScan:
    """The full model scan that runs at import time must not crash."""

    def setup_method(self) -> None:
        clear_model_settings_cache()

    def test_get_past_only_covariate_models_succeeds(self) -> None:
        result = get_past_only_covariate_models()
        assert isinstance(result, frozenset)
        assert len(result) > 0
