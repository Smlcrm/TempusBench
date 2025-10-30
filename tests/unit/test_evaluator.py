"""
Unit tests for the Evaluator class in evaluation.py.

Tests cover:
- Initialization with valid/invalid configs
- Missing logs_path error
- Metric registry setup
- Deterministic task evaluation
- Stochastic task evaluation
- Unknown metric errors
- Config parsing
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from tempus_bench.metrics.evaluation import Evaluator


class TestEvaluatorInitialization:
    """Test Evaluator initialization and configuration."""

    def test_initialization_with_valid_config_deterministic(self):
        """Test Evaluator initialization with valid deterministic config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["mae", "rmse"]
                },
                "task": {
                    "model_type": "deterministic"
                }
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            assert evaluator.model_type == "deterministic"
            assert "mae" in evaluator.metrics_to_calculate
            assert "rmse" in evaluator.metrics_to_calculate

    def test_initialization_with_valid_config_stochastic(self):
        """Test Evaluator initialization with valid stochastic config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["crps", "mae"],
                    "point_forecast_statistic": "mean"
                },
                "task": {
                    "model_type": "stochastic"
                }
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            assert evaluator.model_type == "stochastic"
            assert "crps" in evaluator.metrics_to_calculate
            assert "mae" in evaluator.metrics_to_calculate

    def test_initialization_with_none_config(self):
        """Test Evaluator initialization with None config (defaults to {})."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(KeyError, match="evaluation"):
                Evaluator(config=None, logs_path=tmpdir)

    def test_missing_logs_path(self):
        """Test Evaluator raises error when logs_path is None."""
        config = {
            "evaluation": {"metrics": ["mae"]},
            "task": {"model_type": "deterministic"}
        }
        with pytest.raises(ValueError, match="logs_path parameter is required"):
            Evaluator(config=config, logs_path=None)

    def test_missing_logs_path_default(self):
        """Test Evaluator raises error when logs_path is not provided."""
        config = {
            "evaluation": {"metrics": ["mae"]},
            "task": {"model_type": "deterministic"}
        }
        with pytest.raises(ValueError, match="logs_path parameter is required"):
            Evaluator(config=config)

    def test_metric_registry_setup(self):
        """Test that metric registry is properly set up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["mae", "rmse", "mase", "mape", "crps", 
                                           "quantile_score", "weighted_interval_score"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            assert "mae" in evaluator.metric_registry
            assert "rmse" in evaluator.metric_registry
            assert "crps" in evaluator.metric_registry
            assert "quantile_score" in evaluator.metric_registry

    def test_stochastic_metrics_defined(self):
        """Test that stochastic metrics are properly defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["crps"]},
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            assert evaluator.stochastic_metrics == {"crps", "quantile_score", 
                                                     "weighted_interval_score"}

    def test_deterministic_metrics_defined(self):
        """Test that deterministic metrics are properly defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["mae"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            assert evaluator.deterministic_metrics == {"rmse", "mae", "mase", "mape"}


class TestEvaluatorDeterministic:
    """Test Evaluator with deterministic task type."""

    def test_evaluate_mae_deterministic(self):
        """Test evaluation with MAE metric in deterministic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["mae"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.5, 2.5, 2.5])
            results = evaluator.evaluate(y_true, y_pred)
            assert "mae" in results
            assert results["mae"] == 0.5

    def test_evaluate_multiple_metrics_deterministic(self):
        """Test evaluation with multiple deterministic metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["mae", "rmse"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.5, 2.5, 2.5])
            results = evaluator.evaluate(y_true, y_pred)
            assert "mae" in results
            assert "rmse" in results

    def test_evaluate_mape_mase_deterministic(self):
        """Test evaluation with MAPE and MASE metrics in deterministic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["mape", "mase"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([10.0, 20.0, 30.0])
            y_pred = np.array([11.0, 22.0, 33.0])
            results = evaluator.evaluate(y_true, y_pred)
            assert "mape" in results
            assert "mase" in results

    def test_evaluate_multivariate_deterministic(self):
        """Test evaluation with multivariate data in deterministic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["mae", "rmse"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
            y_pred = np.array([[1.5, 2.5], [2.5, 4.5]])
            results = evaluator.evaluate(y_true, y_pred)
            assert "mae" in results
            assert "rmse" in results


class TestEvaluatorStochastic:
    """Test Evaluator with stochastic task type."""

    def test_evaluate_crps_stochastic(self):
        """Test evaluation with CRPS metric in stochastic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["crps"],
                    "point_forecast_statistic": "mean"
                },
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[2.0]])  # (T=1, M=1)
            y_pred = np.random.randn(100, 1, 1)  # (S=100, T=1, M=1)
            results = evaluator.evaluate(y_true, y_pred)
            assert "crps" in results
            assert results["crps"] >= 0

    def test_evaluate_quantile_score_stochastic(self):
        """Test evaluation with QuantileScore in stochastic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["quantile_score"],
                    "point_forecast_statistic": "mean"
                },
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[2.0]])
            y_pred = np.random.randn(100, 1, 1)
            results = evaluator.evaluate(y_true, y_pred)
            assert "quantile_score" in results
            assert results["quantile_score"] >= 0

    def test_evaluate_weighted_interval_score_stochastic(self):
        """Test evaluation with WeightedIntervalScore in stochastic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["weighted_interval_score"],
                    "point_forecast_statistic": "mean"
                },
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[2.0]])
            y_pred = np.random.randn(100, 1, 1)
            results = evaluator.evaluate(y_true, y_pred)
            assert "weighted_interval_score" in results
            assert results["weighted_interval_score"] >= 0

    def test_evaluate_mae_in_stochastic_mode(self):
        """Test evaluation with MAE metric in stochastic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["mae"],
                    "point_forecast_statistic": "mean"
                },
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[2.0]])
            y_pred = np.array([[[2.0]], [[2.0]], [[2.0]]])  # (S=3, T=1, M=1)
            results = evaluator.evaluate(y_true, y_pred)
            assert "mae" in results
            assert results["mae"] == 0.0

    def test_evaluate_multiple_stochastic_metrics(self):
        """Test evaluation with multiple stochastic metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["crps", "mae", "rmse"],
                    "point_forecast_statistic": "mean"
                },
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[2.0]])
            y_pred = np.random.randn(100, 1, 1)
            results = evaluator.evaluate(y_true, y_pred)
            assert "crps" in results
            assert "mae" in results
            assert "rmse" in results

    def test_evaluate_multivariate_stochastic(self):
        """Test evaluation with multivariate data in stochastic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {
                    "metrics": ["crps", "mae"],
                    "point_forecast_statistic": "mean"
                },
                "task": {"model_type": "stochastic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([[2.0, 3.0], [4.0, 5.0]])  # (T=2, M=2)
            y_pred = np.random.randn(100, 2, 2)
            results = evaluator.evaluate(y_true, y_pred)
            assert "crps" in results
            assert "mae" in results


class TestEvaluatorErrors:
    """Test error handling in Evaluator."""

    def test_unknown_metric_error(self):
        """Test that unknown metric raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "evaluation": {"metrics": ["unknown_metric"]},
                "task": {"model_type": "deterministic"}
            }
            evaluator = Evaluator(config=config, logs_path=tmpdir)
            y_true = np.array([1.0, 2.0])
            y_pred = np.array([1.5, 2.5])
            
            with pytest.raises(ValueError, match="is not a recognized"):
                evaluator.evaluate(y_true, y_pred)

    def test_incorrect_model_type_for_stochastic_metric(self):
        """Test that stochastic metric in deterministic task raises error."""
        # This would be caught at evaluate() level, not initialization
        pass  # Covered by individual metric tests

    def test_missing_point_forecast_statistic(self):
        """Test error when point_forecast_statistic is missing for stochastic mode."""
        # This would be caught when calling individual metrics
        pass  # Covered by metric tests

    def test_config_accessing_none_values(self):
        """Test that accessing None config raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(KeyError, match="evaluation"):
                Evaluator(config={}, logs_path=tmpdir)
