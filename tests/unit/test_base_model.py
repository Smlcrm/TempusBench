"""
Unit tests for BaseModel class.

This test suite ensures that BaseModel correctly handles configuration extraction,
abstract method enforcement, and integration with the evaluation system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tempus_bench.models.base_model import BaseModel
from tempus_bench.metrics.metric_registry import MetricRegistry


class DummyModel(BaseModel):
    """Minimal concrete implementation of BaseModel for testing."""
    
    def train(self, y_context=None, y_target=None, timestamps_context=None, 
              timestamps_target=None, freq=None):
        self.is_fitted = True
        return self
    
    def predict(self, y_context=None, timestamps_context=None, 
                timestamps_target=None, freq=None):
        # Call parent's freq validation
        super().predict(y_context, timestamps_context, timestamps_target, freq)
        return np.array([[1.0, 2.0], [3.0, 4.0]])


class TestBaseModel:
    """Test suite for BaseModel class."""
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_init_with_valid_config(self, mock_evaluator, mock_load_config):
        """Test initialization with valid configuration."""
        # Create a simple dict-like mock
        config_dict = {
            'model': {'dummy': {'param1': 10, 'param2': 'value'}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        
        mock_load_config.return_value = config_dict
        
        # Provide hyperparameters that will be injected
        hyperparameters = {'param1': 10, 'param2': 'value'}
        model = DummyModel("dummy_config.yaml", "logs_path", hyperparameters)
        
        assert model.model_config == {'param1': 10, 'param2': 'value'}
        assert model.training_loss == 'mae'
        assert model.is_fitted == False
        assert model.scaler is None
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_extract_model_config_valid(self, mock_evaluator, mock_load_config):
        """Test _extract_model_config with valid config."""
        config_dict = {
            'model': {'dummy': {'param1': 5, 'param2': 'test'}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        hyperparameters = {'param1': 5, 'param2': 'test'}
        model = DummyModel("config.yaml", "logs_path", hyperparameters)
        assert model.model_config == {'param1': 5, 'param2': 'test'}
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_extract_model_config_missing_model_section(self, mock_evaluator, mock_load_config):
        """Test _extract_model_config raises error when model section is missing."""
        config_dict = {
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        # Create a model instance to access the method
        model = DummyModel("config.yaml", "logs_path", {})
        
        with pytest.raises(ValueError, match="Missing 'model' section"):
            model._extract_model_config(config_dict)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_extract_model_config_multiple_models(self, mock_evaluator, mock_load_config):
        """Test _extract_model_config raises error with multiple models."""
        config_dict = {
            'model': {
                'model1': {'param': 1},
                'model2': {'param': 2}
            },
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyModel("config.yaml", "logs_path", {})
        with pytest.raises(ValueError, match="exactly one model entry"):
            model._extract_model_config(config_dict)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_extract_model_config_model_params_as_list(self, mock_evaluator, mock_load_config):
        """Test _extract_model_config raises error when model params are a list."""
        config_dict = {
            'model': {
                'dummy': [{'param': 1}, {'param': 2}]
            },
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyModel("config.yaml", "logs_path", {})
        with pytest.raises(ValueError, match="must be a dict"):
            model._extract_model_config(config_dict)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_extract_model_config_model_params_as_none(self, mock_evaluator, mock_load_config):
        """Test _extract_model_config raises error when model params are None."""
        config_dict = {
            'model': {
                'dummy': None
            },
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyModel("config.yaml", "logs_path", {})
        with pytest.raises(ValueError, match="must be a dict"):
            model._extract_model_config(config_dict)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_compute_loss(self, mock_evaluator_class, mock_load_config):
        """Test compute_loss method integration with evaluator."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        # Mock the evaluator instance
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'mae': 0.5, 'rmse': 0.7}
        mock_evaluator_class.return_value = mock_evaluator
        
        model = DummyModel("config.yaml", "logs_path", {})
        
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1]])
        
        result = model.compute_loss(y_true, y_pred)
        
        # The actual result will be computed by the real evaluator, not our mock
        # So we need to check that the evaluator was called correctly
        mock_evaluator.evaluate.assert_called_once_with(y_true, y_pred)
        
        # Check that last eval values are stored
        assert model._last_y_true is y_true
        assert model._last_y_pred is y_pred
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_get_params(self, mock_evaluator, mock_load_config):
        """Test get_params returns model configuration."""
        config_dict = {
            'model': {'dummy': {'param1': 10, 'param2': 'value'}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        hyperparameters = {'param1': 10, 'param2': 'value'}
        model = DummyModel("config.yaml", "logs_path", hyperparameters)
        
        params = model.get_params()
        assert params == {'param1': 10, 'param2': 'value'}
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_set_params(self, mock_evaluator, mock_load_config):
        """Test set_params updates model configuration."""
        config_dict = {
            'model': {'dummy': {'param1': 10}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        hyperparameters = {'param1': 10}
        model = DummyModel("config.yaml", "logs_path", hyperparameters)
        model.is_fitted = True
        
        result = model.set_params(param1=20, param2='new')
        
        assert result is model
        assert model.model_config == {'param1': 20, 'param2': 'new'}
        assert model.is_fitted == False  # Should be reset when params change
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_set_scaler(self, mock_evaluator, mock_load_config):
        """Test set_scaler method."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        mock_scaler = Mock()
        
        model = DummyModel("config.yaml", "logs_path", {})
        
        result = model.set_scaler(mock_scaler)
        
        assert result is model
        assert model.scaler is mock_scaler
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_get_model_summary(self, mock_evaluator, mock_load_config):
        """Test get_model_summary returns correct information."""
        config_dict = {
            'model': {'dummy': {'param1': 10}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        hyperparameters = {'param1': 10}
        model = DummyModel("config.yaml", "logs_path", hyperparameters)
        model.is_fitted = True
        
        summary = model.get_model_summary()
        
        assert summary['model_type'] == 'DummyModel'
        assert summary['is_fitted'] == True
        assert summary['parameters'] == {'param1': 10}
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_get_last_eval_true_pred(self, mock_evaluator, mock_load_config):
        """Test get_last_eval_true_pred returns stored values."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyModel("config.yaml", "logs_path", {})
        
        y_true = np.array([[1.0, 2.0]])
        y_pred = np.array([[1.1, 2.1]])
        
        # Set the values directly
        model._last_y_true = y_true
        model._last_y_pred = y_pred
        
        result = model.get_last_eval_true_pred()
        
        assert result == (y_true, y_pred)
    
    @patch('tempus_bench.models.base_model.load_config')
    @patch('tempus_bench.metrics.metric_registry.MetricRegistry')
    def test_predict_freq_validation(self, mock_evaluator, mock_load_config):
        """Test predict method validates freq parameter."""
        config_dict = {
            'model': {'dummy': {}},
            'evaluation': {'tuning_loss': 'mae', 'metrics': ['mae', 'rmse']},
            'task': {'model_type': 'deterministic'}
        }
        mock_load_config.return_value = config_dict
        
        model = DummyModel("config.yaml", "logs_path", {})
        
        # Test with None freq
        with pytest.raises(ValueError, match="Frequency \\(freq\\) must be provided"):
            model.predict(freq=None)
        
        # Test with empty string freq
        with pytest.raises(ValueError, match="Frequency \\(freq\\) must be provided"):
            model.predict(freq="")
    
    def test_abstract_methods_enforcement(self):
        """Test that abstract methods are properly enforced."""
        with pytest.raises(TypeError):
            BaseModel("config.yaml", "logs_path", {})
