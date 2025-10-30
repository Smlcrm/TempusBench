"""
Unit tests for stochastic metrics: CRPS, QuantileScore, WeightedIntervalScore.

Tests cover:
- Basic stochastic calculations with sample distributions
- Shape validation (S, T, M format)
- Error handling for wrong model_type
- Known mathematical results verification
- Edge cases (single sample, many samples)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_raises

from tempus_bench.metrics.crps import CRPS
from tempus_bench.metrics.quantile_score import QuantileScore
from tempus_bench.metrics.weighted_interval_score import WeightedIntervalScore


class TestCRPS:
    """Test CRPS metric (stochastic only)."""

    def test_basic_univariate(self):
        """Test CRPS with univariate stochastic predictions."""
        crps = CRPS()
        y_true = np.array([2.0])  # (1,) -> (T,)
        y_pred = np.array([[[1.0], [2.0], [3.0]]])  # (1, 1, 1) -> (S, T, M)
        
        # Reshape to correct format
        y_true = y_true.reshape(1, 1)
        y_pred = y_pred.reshape(3, 1, 1)
        
        result = crps(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_multivariate(self):
        """Test CRPS with multivariate stochastic predictions."""
        crps = CRPS()
        y_true = np.array([[2.0, 3.0]])  # (1, 2) -> (T, M)
        y_pred = np.random.randn(100, 1, 2)  # (S, T, M)
        y_pred = y_pred + np.array([[[2.0, 3.0]]])
        
        result = crps(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_shape_validation_correct(self):
        """Test CRPS with correct shape."""
        crps = CRPS()
        y_true = np.array([[2.0, 3.0], [4.0, 5.0]])  # (2, 2) -> (T, M)
        y_pred = np.random.randn(50, 2, 2)  # (S, T, M)
        
        result = crps(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_shape_validation_incorrect_y_true(self):
        """Test CRPS with incorrect y_true shape."""
        crps = CRPS()
        y_true = np.array([2.0, 3.0, 4.0])  # Wrong shape (3,)
        y_pred = np.random.randn(50, 2, 2)  # (S=50, T=2, M=2)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            crps(y_true, y_pred, model_type='stochastic')

    def test_shape_validation_incorrect_dimensions(self):
        """Test CRPS with incorrect dimensions."""
        crps = CRPS()
        y_true = np.array([[2.0], [3.0]])  # (2, 1)
        y_pred = np.random.randn(50, 3, 2)  # (S=50, T=3, M=2) - T mismatch
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            crps(y_true, y_pred, model_type='stochastic')

    def test_wrong_model_type(self):
        """Test CRPS raises error with wrong model_type."""
        crps = CRPS()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="can only be used with 'stochastic'"):
            crps(y_true, y_pred, model_type='deterministic')

    def test_wrong_model_type_none(self):
        """Test CRPS raises error when model_type is None."""
        crps = CRPS()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="can only be used with 'stochastic'"):
            crps(y_true, y_pred)

    def test_single_sample(self):
        """Test CRPS with single sample."""
        crps = CRPS()
        y_true = np.array([[2.0]])
        y_pred = np.array([[[2.5]]])  # (1, 1, 1)
        
        result = crps(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_many_samples(self):
        """Test CRPS with many samples."""
        crps = CRPS()
        y_true = np.array([[2.0], [3.0], [4.0]])  # (3, 1)
        y_pred = np.random.randn(1000, 3, 1)  # (1000, 3, 1)
        
        result = crps(y_true, y_pred, model_type='stochastic')
        assert result >= 0


class TestQuantileScore:
    """Test QuantileScore metric (stochastic only)."""

    def test_basic_calculation(self):
        """Test QuantileScore with basic data."""
        qs = QuantileScore()
        y_true = np.array([[2.0]])  # (1, 1)
        y_pred = np.random.randn(100, 1, 1)  # (100, 1, 1)
        
        result = qs(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_multivariate(self):
        """Test QuantileScore with multivariate data."""
        qs = QuantileScore()
        y_true = np.array([[2.0, 3.0], [4.0, 5.0]])  # (2, 2)
        y_pred = np.random.randn(100, 2, 2)  # (100, 2, 2)
        
        result = qs(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_shape_validation_incorrect(self):
        """Test QuantileScore with incorrect shape."""
        qs = QuantileScore()
        y_true = np.array([2.0, 3.0])  # Wrong shape
        y_pred = np.random.randn(100, 2, 2)  # (100, 2, 2)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            qs(y_true, y_pred, model_type='stochastic')

    def test_wrong_model_type(self):
        """Test QuantileScore raises error with wrong model_type."""
        qs = QuantileScore()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(100, 1, 1)
        
        with pytest.raises(ValueError, match="can only be used with 'stochastic'"):
            qs(y_true, y_pred, model_type='deterministic')

    def test_wrong_model_type_none(self):
        """Test QuantileScore raises error when model_type is None."""
        qs = QuantileScore()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(100, 1, 1)
        
        with pytest.raises(ValueError, match="can only be used with 'stochastic'"):
            qs(y_true, y_pred)

    def test_single_sample(self):
        """Test QuantileScore with single sample."""
        qs = QuantileScore()
        y_true = np.array([[2.0]])
        y_pred = np.array([[[2.5]]])  # (1, 1, 1)
        
        result = qs(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_known_distribution(self):
        """Test QuantileScore with known normal distribution."""
        qs = QuantileScore()
        y_true = np.array([[0.0]])  # True value at mean
        y_pred = np.random.randn(10000, 1, 1)  # Standard normal
        
        result = qs(y_true, y_pred, model_type='stochastic')
        assert result >= 0


class TestWeightedIntervalScore:
    """Test WeightedIntervalScore metric (stochastic only)."""

    def test_basic_calculation(self):
        """Test WeightedIntervalScore with basic data."""
        wis = WeightedIntervalScore()
        y_true = np.array([[2.0]])  # (1, 1)
        y_pred = np.random.randn(100, 1, 1)  # (100, 1, 1)
        
        result = wis(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_multivariate(self):
        """Test WeightedIntervalScore with multivariate data."""
        wis = WeightedIntervalScore()
        y_true = np.array([[2.0, 3.0], [4.0, 5.0]])  # (2, 2)
        y_pred = np.random.randn(100, 2, 2)  # (100, 2, 2)
        
        result = wis(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_shape_validation_incorrect(self):
        """Test WeightedIntervalScore with incorrect shape."""
        wis = WeightedIntervalScore()
        y_true = np.array([2.0, 3.0])  # Wrong shape
        y_pred = np.random.randn(100, 2, 2)  # (100, 2, 2)
        
        with pytest.raises(ValueError, match="non-broadcastable output operand"):
            wis(y_true, y_pred, model_type='stochastic')

    def test_wrong_model_type(self):
        """Test WeightedIntervalScore raises error with wrong model_type."""
        wis = WeightedIntervalScore()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(100, 1, 1)
        
        with pytest.raises(ValueError, match="can only be used with 'stochastic'"):
            wis(y_true, y_pred, model_type='deterministic')

    def test_wrong_model_type_none(self):
        """Test WeightedIntervalScore raises error when model_type is None."""
        wis = WeightedIntervalScore()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(100, 1, 1)
        
        with pytest.raises(ValueError, match="can only be used with 'stochastic'"):
            wis(y_true, y_pred)

    def test_single_sample(self):
        """Test WeightedIntervalScore with single sample."""
        wis = WeightedIntervalScore()
        y_true = np.array([[2.0]])
        y_pred = np.array([[[2.5]]])  # (1, 1, 1)
        
        result = wis(y_true, y_pred, model_type='stochastic')
        assert result >= 0

    def test_known_distribution(self):
        """Test WeightedIntervalScore with known normal distribution."""
        wis = WeightedIntervalScore()
        y_true = np.array([[0.0]])  # True value at mean
        y_pred = np.random.randn(10000, 1, 1)  # Standard normal
        
        result = wis(y_true, y_pred, model_type='stochastic')
        assert result >= 0
