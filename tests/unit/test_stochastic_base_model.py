"""
Unit tests for BaseModel class.

This test suite ensures that BaseModel correctly inherits from BaseModel,
handles stochastic-specific functionality, and integrates with the evaluation system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tempus_bench.models.base_model import BaseModel
from tempus_bench.pipeline.metric_registry import MetricRegistry


class DummyStochasticModel(BaseModel):
    """Minimal concrete implementation of BaseModel for testing."""
    
    def train(self, y_context=None, y_target=None, timestamps_context=None, 
              timestamps_target=None, freq=None):
        self.is_fitted = True
        return self
    
    def predict(self, y_context=None, timestamps_context=None, 
                timestamps_target=None, freq=None):
        # Call parent's freq validation
        super().predict(y_context, timestamps_context, timestamps_target, freq)
        # Return samples with shape (num_samples, forecast_horizon, num_targets)
        return np.random.randn(5, 3, 2)  # 5 samples, 3 horizon, 2 targets


class TestBaseModel:
    """Test suite for BaseModel class."""
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_init_with_valid_config(self, mock_evaluator, mock_load_config):
        """Test initialization with valid configuration."""
        config_dict = {
            'model': {'dummy': {'param1': 10, 'param2': 'value'}},
            'evaluation': {
                'tuning_loss': 'mae',
                'metrics': ['mae', 'rmse'],
                'num_samples': 5,
                'point_forecast_statistic': 'mean'
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        hyperparameters = {'param1': 10, 'param2': 'value'}
        model = DummyStochasticModel("dummy_config.yaml", "logs_path", hyperparameters)
        
        assert model.model_config == {'param1': 10, 'param2': 'value'}
        assert model.training_loss == 'mae'
        assert model.num_samples == 5
        assert model.point_forecast_statistic == 'mean'
        assert model.is_fitted == False
        assert model.scaler is None
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_init_invalid_point_forecast_statistic(self, mock_evaluator, mock_load_config):
        """Test initialization raises error with invalid point forecast statistic."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'median',  # Only 'mean' is supported
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        with pytest.raises(ValueError, match="Only 'mean' point forecast statistic is supported"):
            DummyStochasticModel("config.yaml", "logs_path", {})
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_compute_point_forecast_mean(self, mock_evaluator, mock_load_config):
        """Test compute_point_forecast with mean statistic."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'mean',
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyStochasticModel("config.yaml", "logs_path", {})
        
        # Create samples with shape (num_samples, forecast_horizon, num_targets)
        samples = np.array([
            [[1.0, 2.0], [3.0, 4.0]],  # Sample 1
            [[2.0, 3.0], [4.0, 5.0]],  # Sample 2
            [[0.0, 1.0], [2.0, 3.0]]   # Sample 3
        ])  # Shape: (3, 2, 2)
        
        point_forecast = model.compute_point_forecast(samples)
        
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])  # Mean across samples
        np.testing.assert_array_almost_equal(point_forecast, expected)
        assert point_forecast.shape == (2, 2)  # (forecast_horizon, num_targets)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_compute_point_forecast_invalid_statistic(self, mock_evaluator, mock_load_config):
        """Test compute_point_forecast raises error with invalid statistic."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'mean',
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyStochasticModel("config.yaml", "logs_path", {})
        
        # Manually change the statistic to test error handling
        model.point_forecast_statistic = 'median'
        
        samples = np.random.randn(3, 2, 2)
        
        with pytest.raises(ValueError, match="Point forecast statistic 'median' not supported"):
            model.compute_point_forecast(samples)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_compute_metrics_with_samples(self, mock_evaluator_class, mock_load_config):
        """Test compute_metrics method with samples input."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'mean',
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        mock_evaluator = Mock()
        mock_evaluator.compute_metrics.return_value = {'mae': 0.5, 'rmse': 0.7}
        mock_evaluator_class.return_value = mock_evaluator
        
        model = DummyStochasticModel("config.yaml", "logs_path", {})
        
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])  # Shape: (forecast_horizon, num_targets)
        y_pred = np.random.randn(5, 2, 2)  # Shape: (num_samples, forecast_horizon, num_targets)
        
        result = model.compute_metrics(y_true, y_pred)
        
        # The result should be computed by the real evaluator
        # We just check that the method runs without error and stores the values
        assert isinstance(result, dict)
        
        # Check that last eval values are stored
        assert model._last_y_true is y_true
        assert model._last_y_pred is y_pred
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_get_model_summary_includes_stochastic_fields(self, mock_evaluator, mock_load_config):
        """Test get_model_summary includes stochastic-specific fields."""
        config_dict = {
            'model': {'dummy': {'param1': 10}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'mean',
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        hyperparameters = {'param1': 10}
        model = DummyStochasticModel("config.yaml", "logs_path", hyperparameters)
        model.is_fitted = True
        
        summary = model.get_model_summary()
        
        assert summary['model_type'] == 'DummyStochasticModel'
        assert summary['is_fitted'] == True
        assert summary['parameters'] == {'param1': 10}
        assert summary['num_samples'] == 5
        assert summary['point_forecast_statistic'] == 'mean'
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_inheritance_from_base_model(self, mock_evaluator, mock_load_config):
        """Test that BaseModel properly inherits from BaseModel."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'mean',
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyStochasticModel("config.yaml", "logs_path", {})
        
        # Test that BaseModel methods are available
        assert hasattr(model, 'get_params')
        assert hasattr(model, 'set_params')
        assert hasattr(model, 'set_scaler')
        assert hasattr(model, 'compute_metrics')
        assert hasattr(model, 'get_model_summary')
        assert hasattr(model, 'get_last_eval_true_pred')
        
        # Test that BaseModel methods are available
        assert hasattr(model, 'compute_point_forecast')
        assert hasattr(model, 'num_samples')
        assert hasattr(model, 'point_forecast_statistic')
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_predict_freq_validation_inherited(self, mock_evaluator, mock_load_config):
        """Test that freq validation is inherited from BaseModel."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {
                'tuning_loss': 'mae',
                'num_samples': 5,
                'point_forecast_statistic': 'mean',
                'metrics': ['mae', 'rmse']
            },
            'task': {'model_type': 'stochastic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyStochasticModel("config.yaml", "logs_path", {})
        
        with pytest.raises(ValueError, match="Frequency \\(freq\\) must be provided"):
            model.predict(freq=None)
        
        with pytest.raises(ValueError, match="Frequency \\(freq\\) must be provided"):
            model.predict(freq="")
    
    def test_abstract_methods_enforcement(self):
        """Test that abstract methods are properly enforced."""
        with pytest.raises(TypeError):
            BaseModel("config.yaml", "logs_path", {})
