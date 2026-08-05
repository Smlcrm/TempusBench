"""Unit tests for model capability gating with multi-target covariates."""

import pytest

from tempus_bench.utils.model_settings import (
    ModelCapabilities,
    assert_model_supports_task_family,
    task_path_to_family,
)


def test_task_path_to_family_requires_mode():
    with pytest.raises(ValueError, match="without task_mode"):
        task_path_to_family("commerce_and_trade/Some Task")


def test_task_path_to_family_uses_mode():
    assert task_path_to_family("anything", task_mode="covariate") == "covariate"


def test_multi_target_covariate_requires_multivariate():
    cap = ModelCapabilities(
        covariates="past_future",
        univariate=True,
        multivariate=False,
    )
    with pytest.raises(ValueError, match="multi-target covariate"):
        assert_model_supports_task_family(
            cap, model_name="demo", family="covariate", num_targets=2
        )


def test_single_target_covariate_ok_without_multivariate():
    cap = ModelCapabilities(
        covariates="past_future",
        univariate=True,
        multivariate=False,
    )
    assert_model_supports_task_family(
        cap, model_name="demo", family="covariate", num_targets=1
    )


def test_multi_target_covariate_ok_with_multivariate():
    cap = ModelCapabilities(
        covariates="past_future",
        univariate=True,
        multivariate=True,
    )
    assert_model_supports_task_family(
        cap, model_name="demo", family="covariate", num_targets=3
    )
