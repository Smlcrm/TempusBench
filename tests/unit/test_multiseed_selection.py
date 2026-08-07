"""Seed-averaged hyperparameter selection."""

import numpy as np
import pytest

from tempus_bench.pipeline.hyperparameter_tuner import HyperparameterTuner

A = (("alpha", 0.1),)
B = (("alpha", 0.5),)


def test_selects_the_config_with_the_lowest_mean_across_seeds():
    # A wins on seed 0 but loses badly on seed 1; B is steadier and wins on average.
    losses = {A: {0: 1.0, 1: 9.0}, B: {0: 4.0, 1: 4.0}}
    assert HyperparameterTuner._select_best_params(losses) == B


def test_a_single_seed_reduces_to_that_seeds_loss():
    losses = {A: {0: 1.0}, B: {0: 2.5}}
    assert HyperparameterTuner._select_best_params(losses) == A


def test_configs_missing_a_seed_are_not_selectable():
    # A has the lower loss where it ran, but it failed on seed 1 entirely.
    losses = {A: {0: 0.1}, B: {0: 5.0, 1: 5.0}}
    assert HyperparameterTuner._select_best_params(losses, required_seeds={0, 1}) == B


def test_raises_when_no_config_covers_every_seed():
    with pytest.raises(ValueError, match="no hyperparameter configuration"):
        HyperparameterTuner._select_best_params({A: {0: 1.0}}, required_seeds={0, 1})


def test_none_losses_are_ignored():
    losses = {A: {0: None, 1: None}, B: {0: 3.0, 1: 3.0}}
    assert HyperparameterTuner._select_best_params(losses) == B


def test_mean_metrics_averages_scalars_and_drops_artifacts():
    windows = [
        {"mae": 1.0, "rmse": 2.0, "y_true": [1, 2, 3], "y_pred": [1, 2, 3]},
        {"mae": 3.0, "rmse": 4.0, "y_true": [4, 5, 6], "y_pred": [4, 5, 6]},
    ]
    means = HyperparameterTuner._mean_metrics_over_windows(windows)
    assert means == {"mae": 2.0, "rmse": 3.0}


def test_mean_metrics_handles_numpy_scalars():
    windows = [{"mae": np.float64(1.0)}, {"mae": np.float64(3.0)}]
    assert HyperparameterTuner._mean_metrics_over_windows(windows) == {"mae": 2.0}


def test_mean_metrics_drops_non_finite_values():
    windows = [{"mae": 1.0}, {"mae": float("nan")}]
    assert HyperparameterTuner._mean_metrics_over_windows(windows) == {"mae": 1.0}
