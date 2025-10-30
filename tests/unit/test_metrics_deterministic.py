"""
Unit tests for deterministic metrics: MAE, MAPE, MASE, RMSE.

Tests cover:
- Basic calculations (univariate & multivariate)
- Perfect predictions (zero error)
- Edge cases (zeros, negatives, small values)
- Shape validation
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_raises

from tempus_bench.metrics.mae import MAE
from tempus_bench.metrics.mape import MAPE
from tempus_bench.metrics.mase import MASE
from tempus_bench.metrics.rmse import RMSE


class TestMAEDeterministic:
    """Test MAE metric in deterministic mode."""

    def test_basic_univariate(self):
        """Test MAE with simple univariate data."""
        mae = MAE()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5, 5.5])
        # Errors: [0.5, 0.5, 0.5, 0.5, 0.5] -> mean = 0.5
        result = mae(y_true, y_pred, model_type='deterministic')
        assert result == 0.5

    def test_multivariate(self):
        """Test MAE with multivariate data."""
        mae = MAE()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.5, 2.5], [2.5, 4.5]])
        # Errors: [0.5, 0.5, 0.5, 0.5] -> mean = 0.5
        result = mae(y_true, y_pred, model_type='deterministic')
        assert result == 0.5

    def test_perfect_predictions(self):
        """Test MAE with perfect predictions (zero error)."""
        mae = MAE()
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = mae(y_true, y_pred, model_type='deterministic')
        assert result == 0.0

    def test_negative_values(self):
        """Test MAE with negative values."""
        mae = MAE()
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.5, -2.5, -2.5])
        result = mae(y_true, y_pred, model_type='deterministic')
        assert result == 0.5

    def test_zero_values(self):
        """Test MAE with zero values."""
        mae = MAE()
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mae(y_true, y_pred, model_type='deterministic')
        assert result == 2.0

    def test_default_model_type(self):
        """Test MAE with default model_type (deterministic)."""
        mae = MAE()
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])
        result = mae(y_true, y_pred)
        assert result == 0.5

    def test_single_value(self):
        """Test MAE with single value."""
        mae = MAE()
        y_true = np.array([10.0])
        y_pred = np.array([11.0])
        result = mae(y_true, y_pred, model_type='deterministic')
        assert result == 1.0


class TestMAPEDeterministic:
    """Test MAPE metric in deterministic mode."""

    def test_basic_calculation(self):
        """Test MAPE with simple data."""
        mape = MAPE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 3.3])
        # Errors: 10% each -> mean = 10%
        result = mape(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 10.0)

    def test_perfect_predictions(self):
        """Test MAPE with perfect predictions."""
        mape = MAPE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mape(y_true, y_pred, model_type='deterministic')
        assert result == 0.0

    def test_zero_true_values(self):
        """Test MAPE with zero true values (epsilon protection)."""
        mape = MAPE()
        y_true = np.array([0.0, 0.0, 5.0])
        y_pred = np.array([1.0, 2.0, 5.5])
        # Should not crash due to epsilon
        result = mape(y_true, y_pred, model_type='deterministic')
        assert result >= 0

    def test_small_values(self):
        """Test MAPE with very small values."""
        mape = MAPE()
        y_true = np.array([0.001, 0.002, 0.003])
        y_pred = np.array([0.0011, 0.0022, 0.0033])
        result = mape(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 10.0, atol=0.1)

    def test_negative_values(self):
        """Test MAPE with negative values."""
        mape = MAPE()
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.1, -2.2, -3.3])
        result = mape(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 10.0)


class TestMASEDeterministic:
    """Test MASE metric in deterministic mode."""

    def test_basic_calculation(self):
        """Test MASE with simple data."""
        mase = MASE()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.0, 3.5, 4.0, 5.5])
        # MAE = mean(|[0.5, 0.0, 0.5, 0.0, 0.5]|) = 0.3
        # Denominator = mean(|y_true[i+1] - y_true[i]|) = mean([1, 1, 1, 1]) = 1
        # MASE = 0.3 / 1 = 0.3
        result = mase(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 0.3)

    def test_perfect_predictions(self):
        """Test MASE with perfect predictions."""
        mase = MASE()
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = mase(y_true, y_pred, model_type='deterministic')
        assert result == 0.0

    def test_constant_series(self):
        """Test MASE with constant time series."""
        mase = MASE()
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([6.0, 6.0, 6.0, 6.0])
        # Denominator = 0, but code uses np.max(1e-10, ...) to avoid division by zero
        # MAE = 1.0
        # MASE = 1.0 / 1e-10
        result = mase(y_true, y_pred, model_type='deterministic')
        assert result > 0

    def test_single_step(self):
        """Test MASE with minimal data."""
        mase = MASE()
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])
        result = mase(y_true, y_pred, model_type='deterministic')
        # MAE = 0.5, denominator = abs(2-1) = 1
        assert_allclose(result, 0.5)


class TestRMSEDeterministic:
    """Test RMSE metric in deterministic mode."""

    def test_basic_univariate(self):
        """Test RMSE with simple univariate data."""
        rmse = RMSE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 3.5])
        # Squared errors: [0.25, 0.0, 0.25] -> mean = 0.1667 -> sqrt ≈ 0.408
        result = rmse(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 0.4082483, atol=1e-5)

    def test_multivariate(self):
        """Test RMSE with multivariate data."""
        rmse = RMSE()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.5, 2.0], [2.5, 4.5]])
        # Squared errors: [0.25, 0.0, 0.25, 0.25] -> mean = 0.1875 -> sqrt ≈ 0.433
        result = rmse(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 0.4330127, atol=1e-5)

    def test_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        rmse = RMSE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = rmse(y_true, y_pred, model_type='deterministic')
        assert result == 0.0

    def test_negative_values(self):
        """Test RMSE with negative values."""
        rmse = RMSE()
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.5, -2.0, -3.5])
        # Squared errors: [0.25, 0.0, 0.25]
        result = rmse(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 0.4082483, atol=1e-5)

    def test_zero_values(self):
        """Test RMSE with zero values."""
        rmse = RMSE()
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # Squared errors: [1.0, 4.0, 9.0] -> mean = 4.667 -> sqrt ≈ 2.16
        result = rmse(y_true, y_pred, model_type='deterministic')
        assert_allclose(result, 2.1602469, atol=1e-5)

    def test_single_value(self):
        """Test RMSE with single value."""
        rmse = RMSE()
        y_true = np.array([10.0])
        y_pred = np.array([11.0])
        result = rmse(y_true, y_pred, model_type='deterministic')
        assert result == 1.0
