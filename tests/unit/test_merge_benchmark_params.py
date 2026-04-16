"""Tests for merging benchmark model params with default_hyperparameter_grid."""

from unittest.mock import patch

import pytest

from tempus_bench.utils.model_settings import merge_benchmark_params_with_default_grid


def test_foundation_model_no_merge():
    settings = {
        "foundation": True,
        "default_hyperparameter_grid": {"a": [1, 2, 3]},
        "capabilities": {
            "covariates": "none",
            "univariate": True,
            "multivariate": True,
        },
    }
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        out = merge_benchmark_params_with_default_grid("fake_foundation", {})
    assert out == {}


def test_foundation_model_preserves_explicit_params():
    settings = {"foundation": True, "default_hyperparameter_grid": {}}
    bench = {"num_samples": [50]}
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        out = merge_benchmark_params_with_default_grid("fake_foundation", bench)
    assert out == {"num_samples": [50]}
    assert out is not bench


def test_non_foundation_fills_missing_with_full_grid_list():
    settings = {
        "foundation": False,
        "default_hyperparameter_grid": {
            "input_size": [256, 512],
            "max_steps": [50, 100],
        },
    }
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        out = merge_benchmark_params_with_default_grid(
            "fake_nf",
            {"max_steps": [100]},
        )
    assert out == {"max_steps": [100], "input_size": [256, 512]}


def test_non_foundation_empty_benchmark_uses_all_grid_defaults():
    settings = {
        "foundation": False,
        "default_hyperparameter_grid": {"x": [10, 20], "y": [0.1]},
    }
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        out = merge_benchmark_params_with_default_grid("fake_nf", {})
    assert out == {"x": [10, 20], "y": [0.1]}


def test_non_foundation_scalar_grid_value_wrapped():
    settings = {
        "foundation": False,
        "default_hyperparameter_grid": {"stride": 8},
    }
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        out = merge_benchmark_params_with_default_grid("fake_nf", {})
    assert out == {"stride": [8]}


def test_missing_foundation_raises():
    settings = {"default_hyperparameter_grid": {}}
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        with pytest.raises(ValueError, match="foundation"):
            merge_benchmark_params_with_default_grid("m", {})


def test_non_foundation_invalid_grid_type_raises():
    settings = {"foundation": False, "default_hyperparameter_grid": [1, 2]}
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        with pytest.raises(ValueError, match="default_hyperparameter_grid"):
            merge_benchmark_params_with_default_grid("m", {})


def test_non_foundation_empty_list_in_grid_raises():
    settings = {"foundation": False, "default_hyperparameter_grid": {"a": []}}
    with patch(
        "tempus_bench.utils.model_settings.load_model_settings_yaml",
        return_value=settings,
    ):
        with pytest.raises(ValueError, match="empty list"):
            merge_benchmark_params_with_default_grid("m", {})


def test_real_nbeats_partial_benchmark_merges_from_packaged_settings():
    """Integration: packaged ``nbeats/settings.yaml`` grid fills omitted keys."""
    from tempus_bench.utils.model_settings import clear_model_settings_cache

    clear_model_settings_cache()
    out = merge_benchmark_params_with_default_grid(
        "nbeats",
        {"max_steps": [100]},
    )
    assert out["max_steps"] == [100]
    assert out["input_size"] == [128, 256]


def test_real_chronos_tiny_foundation_empty_stays_empty():
    from tempus_bench.utils.model_settings import clear_model_settings_cache

    clear_model_settings_cache()
    out = merge_benchmark_params_with_default_grid("chronos_tiny", {})
    assert out == {}


def test_real_patchtst_fm_empty_benchmark_gets_full_stochastic_samples_grid():
    from tempus_bench.utils.model_settings import clear_model_settings_cache

    clear_model_settings_cache()
    out = merge_benchmark_params_with_default_grid("patchtst_fm", {})
    assert out == {"stochastic_samples": [50, 100, 200]}
