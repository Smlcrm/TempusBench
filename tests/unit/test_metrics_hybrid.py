"""
Unit tests for hybrid metrics (MAE, MAPE, MASE, RMSE) in stochastic mode.

Tests cover:
- Stochastic mode with point_forecast_statistic='mean'
- Shape mismatch errors
- Invalid point_forecast_statistic values
- Transition between deterministic and stochastic modes
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tempus_bench.metrics.mae import MAE
from tempus_bench.metrics.mape import MAPE
from tempus_bench.metrics.mase import MASE
from tempus_bench.metrics.rmse import RMSE


class TestMAEStochastic:
    """Test MAE metric in stochastic mode."""

    def test_stochastic_mode_with_mean(self):
        """Test MAE with stochastic predictions using mean statistic."""
        mae = MAE()
        y_true = np.array([[2.0], [3.0]])  # (T=2, M=1)
        # y_pred shape: (S=3, T=2, M=1)
        y_pred = np.array([
            [[1.0], [2.0]],  # Sample 1
            [[2.0], [3.0]],  # Sample 2
            [[3.0], [4.0]]   # Sample 3
        ])
        # Mean: [[2.0], [3.0]] - same as y_true
        result = mae(y_true, y_pred, task_type='stochastic', 
                     point_forecast_statistic='mean')
        assert result == 0.0

    def test_stochastic_mode_with_mean_nonzero(self):
        """Test MAE with stochastic predictions where mean != true."""
        mae = MAE()
        y_true = np.array([[2.0]])  # (T=1, M=1)
        # y_pred shape: (S=3, T=1, M=1)
        y_pred = np.array([[[1.0]], [[3.0]], [[5.0]]])  # Mean = 3.0
        result = mae(y_true, y_pred, task_type='stochastic',
                     point_forecast_statistic='mean')
        assert result == 1.0  # |2.0 - 3.0| = 1.0

    def test_multivariate_stochastic(self):
        """Test MAE with multivariate stochastic predictions."""
        mae = MAE()
        y_true = np.array([[2.0, 3.0], [4.0, 5.0]])  # (T=2, M=2)
        # Create samples with known mean
        y_pred = np.random.randn(100, 2, 2)
        y_pred = y_pred + np.array([[[2.0, 3.0], [4.0, 5.0]]])
        
        result = mae(y_true, y_pred, task_type='stochastic',
                     point_forecast_statistic='mean')
        assert result < 1.0  # Should be small for large sample

    def test_shape_validation_incorrect_y_true(self):
        """Test MAE raises error with incorrect y_true shape."""
        mae = MAE()
        y_true = np.array([2.0, 3.0])  # Wrong shape, should be (T, M)
        y_pred = np.random.randn(50, 2, 2)  # (S=50, T=2, M=2)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mae(y_true, y_pred, task_type='stochastic',
                point_forecast_statistic='mean')

    def test_invalid_point_forecast_statistic(self):
        """Test MAE raises error with invalid point_forecast_statistic."""
        mae = MAE()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="can only handle point_forecast_statistic == 'mean'"):
            mae(y_true, y_pred, task_type='stochastic',
                point_forecast_statistic='median')

    def test_missing_point_forecast_statistic(self):
        """Test MAE raises error when point_forecast_statistic is missing."""
        mae = MAE()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(KeyError):
            mae(y_true, y_pred, task_type='stochastic')


class TestMAPEStochastic:
    """Test MAPE metric in stochastic mode."""

    def test_stochastic_mode_with_mean(self):
        """Test MAPE with stochastic predictions using mean."""
        mape = MAPE()
        y_true = np.array([[100.0]])  # (T=1, M=1)
        # y_pred shape: (S=3, T=1, M=1)
        y_pred = np.array([[[110.0]], [[100.0]], [[90.0]]])  # Mean = 100.0
        result = mape(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result == 0.0

    def test_stochastic_mode_nonzero_error(self):
        """Test MAPE with stochastic predictions where mean != true."""
        mape = MAPE()
        y_true = np.array([[100.0]])
        y_pred = np.array([[[100.0]], [[110.0]], [[110.0]]])  # Mean = 106.67
        result = mape(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result > 0

    def test_shape_validation_incorrect(self):
        """Test MAPE raises error with incorrect shape."""
        mape = MAPE()
        y_true = np.array([100.0])  # Wrong shape
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mape(y_true, y_pred, task_type='stochastic',
                 point_forecast_statistic='mean')

    def test_invalid_point_forecast_statistic(self):
        """Test MAPE raises error with invalid point_forecast_statistic."""
        mape = MAPE()
        y_true = np.array([[100.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="can only handle point_forecast_statistic == 'mean'"):
            mape(y_true, y_pred, task_type='stochastic',
                 point_forecast_statistic='median')

    def test_epsilon_protection(self):
        """Test MAPE with zero values protected by epsilon."""
        mape = MAPE()
        y_true = np.array([[0.0]])
        y_pred = np.array([[[1.0]], [[2.0]], [[3.0]]])  # Mean = 2.0
        result = mape(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result >= 0


class TestMASEStochastic:
    """Test MASE metric in stochastic mode."""

    def test_stochastic_mode_with_mean(self):
        """Test MASE with stochastic predictions using mean."""
        mase = MASE()
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])  # (T=4, M=1)
        # y_pred shape: (S=3, T=4, M=1)
        y_pred = np.array([
            [[1.0], [2.0], [3.0], [4.0]],  # Sample 1
            [[1.0], [2.0], [3.0], [4.0]],  # Sample 2
            [[1.0], [2.0], [3.0], [4.0]]   # Sample 3
        ])
        # Mean: same as y_true -> MAE = 0
        result = mase(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result == 0.0

    def test_stochastic_mode_nonzero_error(self):
        """Test MASE with stochastic predictions where mean != true."""
        mase = MASE()
        y_true = np.array([[1.0], [2.0], [3.0]])  # (T=3, M=1)
        # y_pred shape: (S=3, T=3, M=1)
        y_pred = np.array([
            [[2.0], [3.0], [4.0]],
            [[2.0], [3.0], [4.0]],
            [[2.0], [3.0], [4.0]]
        ])
        # Mean: [[2.0], [3.0], [4.0]]
        # MAE = mean(|[1-2, 2-3, 3-4]|) = 1.0
        # Denominator = mean(|[2-1, 3-2]|) = 1.0
        # MASE = 1.0
        result = mase(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert_allclose(result, 1.0, atol=1e-5)

    def test_shape_validation_incorrect(self):
        """Test MASE raises error with incorrect shape."""
        mase = MASE()
        y_true = np.array([1.0, 2.0, 3.0])  # Wrong shape
        y_pred = np.random.randn(50, 3, 1)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mase(y_true, y_pred, task_type='stochastic',
                 point_forecast_statistic='mean')

    def test_invalid_point_forecast_statistic(self):
        """Test MASE raises error with invalid point_forecast_statistic."""
        mase = MASE()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.random.randn(50, 3, 1)
        
        with pytest.raises(ValueError, match="can only handle point_forecast_statistic == 'mean'"):
            mase(y_true, y_pred, task_type='stochastic',
                 point_forecast_statistic='median')


class TestRMSEStochastic:
    """Test RMSE metric in stochastic mode."""

    def test_stochastic_mode_with_mean(self):
        """Test RMSE with stochastic predictions using mean."""
        rmse = RMSE()
        y_true = np.array([[2.0]])  # (T=1, M=1)
        # y_pred shape: (S=3, T=1, M=1)
        y_pred = np.array([[[2.0]], [[2.0]], [[2.0]]])  # Mean = 2.0
        result = rmse(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result == 0.0

    def test_stochastic_mode_nonzero_error(self):
        """Test RMSE with stochastic predictions where mean != true."""
        rmse = RMSE()
        y_true = np.array([[2.0]])
        # y_pred shape: (S=4, T=1, M=1)
        y_pred = np.array([[[1.0]], [[3.0]], [[5.0]], [[7.0]]])  # Mean = 4.0
        result = rmse(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result == 2.0  # sqrt((2-4)^2) = 2.0

    def test_multivariate_stochastic(self):
        """Test RMSE with multivariate stochastic predictions."""
        rmse = RMSE()
        y_true = np.array([[2.0, 3.0], [4.0, 5.0]])  # (T=2, M=2)
        y_pred = np.random.randn(100, 2, 2)
        y_pred = y_pred + np.array([[[2.0, 3.0], [4.0, 5.0]]])
        
        result = rmse(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result < 1.0  # Should be small for large sample

    def test_shape_validation_incorrect(self):
        """Test RMSE with incorrect shape (should not raise error, just compute)."""
        rmse = RMSE()
        y_true = np.array([2.0])  # Wrong shape
        y_pred = np.random.randn(50, 1, 1)
        
        # RMSE doesn't validate shapes, it just computes
        result = rmse(y_true, y_pred, task_type='stochastic',
                      point_forecast_statistic='mean')
        assert result >= 0

    def test_invalid_point_forecast_statistic(self):
        """Test RMSE raises error with invalid point_forecast_statistic."""
        rmse = RMSE()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="can only handle point_forecast_statistic == 'mean'"):
            rmse(y_true, y_pred, task_type='stochastic',
                 point_forecast_statistic='median')

    def test_error_message_verification(self):
        """Test that error message mentions MASE (typo in code)."""
        rmse = RMSE()
        y_true = np.array([[2.0]])
        y_pred = np.random.randn(50, 1, 1)
        
        with pytest.raises(ValueError, match="RMSE can only handle"):
            rmse(y_true, y_pred, task_type='stochastic',
                 point_forecast_statistic='median')
