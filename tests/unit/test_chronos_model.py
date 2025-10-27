"""
Unit tests for Chronos model.

This test suite ensures that ChronosModel correctly implements the BaseModel interface,
returns samples with proper shape, and integrates properly with the evaluation system.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from tempus_bench.models.stochastic.chronos.chronos_model import ChronosModel


class TestChronosModel:
    """Test suite for ChronosModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'chronos': {
                    'model_size': 'tiny',
                    'context_length': 512
                }
            },
            'evaluation': {
                'tuning_loss': 'mae',
                'metrics': ['mae', 'rmse'],
                'num_samples': 5,
                'point_forecast_statistic': 'mean'
            }
        }
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    def test_init_with_valid_config(self, mock_evaluator, mock_load_config):
        """Test initialization with valid configuration."""
        mock_load_config.return_value = self.config
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        assert model.model_config['model_size'] == 'tiny'
        assert model.model_config['context_length'] == 512
        assert model.is_fitted == False
        assert model.num_samples == 5
        assert model.point_forecast_statistic == 'mean'
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    @patch('chronos.ChronosPipeline.from_pretrained')
    def test_train_loads_model(self, mock_from_pretrained, mock_evaluator, mock_load_config):
        """Test that train method loads the Chronos model."""
        mock_load_config.return_value = self.config
        
        # Mock Chronos pipeline
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        # Mock data
        y_context = np.random.randn(10, 2)  # (context_steps, num_targets)
        y_target = np.random.randn(5, 2)   # (forecast_horizon, num_targets)
        timestamps_context = np.arange(10)
        timestamps_target = np.arange(10, 15)
        
        result = model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        assert result is model
        assert model.is_fitted == True
        assert model.model is mock_pipeline
        
        # Verify model was loaded with correct parameters
        mock_from_pretrained.assert_called_once_with(
            "amazon/chronos-t5-tiny",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    @patch('chronos.ChronosPipeline.from_pretrained')
    def test_predict_returns_samples(self, mock_from_pretrained, mock_evaluator, mock_load_config):
        """Test that predict returns samples with correct shape."""
        mock_load_config.return_value = self.config
        
        # Mock Chronos pipeline
        mock_pipeline = Mock()
        # Mock forecast output: (num_targets, num_samples, forecast_horizon)
        mock_forecasts = np.random.randn(2, 5, 3)  # 2 targets, 5 samples, 3 horizon
        mock_pipeline.predict.return_value = mock_forecasts
        mock_from_pretrained.return_value = mock_pipeline
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        # Train first
        y_context = np.random.randn(10, 2)
        y_target = np.random.randn(3, 2)
        timestamps_context = np.arange(10)
        timestamps_target = np.arange(10, 13)
        
        model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        # Predict
        predictions = model.predict(y_context, timestamps_context, timestamps_target, "D")
        
        # Should return shape (num_samples, forecast_horizon, num_targets)
        assert predictions.shape == (5, 3, 2)
        
        # Verify the transpose was applied correctly
        expected = np.transpose(mock_forecasts, (1, 2, 0))
        np.testing.assert_array_equal(predictions, expected)
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    @patch('chronos.ChronosPipeline.from_pretrained')
    def test_predict_univariate(self, mock_from_pretrained, mock_evaluator, mock_load_config):
        """Test prediction with univariate data."""
        mock_load_config.return_value = self.config
        
        # Mock Chronos pipeline
        mock_pipeline = Mock()
        # Mock forecast output for univariate: (1, num_samples, forecast_horizon)
        mock_forecasts = np.random.randn(1, 5, 3)
        mock_pipeline.predict.return_value = mock_forecasts
        mock_from_pretrained.return_value = mock_pipeline
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        # Train with univariate data
        y_context = np.random.randn(10, 1)  # (context_steps, 1)
        y_target = np.random.randn(3, 1)    # (forecast_horizon, 1)
        timestamps_context = np.arange(10)
        timestamps_target = np.arange(10, 13)
        
        model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        # Predict
        predictions = model.predict(y_context, timestamps_context, timestamps_target, "D")
        
        # Should return shape (num_samples, forecast_horizon, 1)
        assert predictions.shape == (5, 3, 1)
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    @patch('chronos.ChronosPipeline.from_pretrained')
    def test_predict_context_padding(self, mock_from_pretrained, mock_evaluator, mock_load_config):
        """Test that context is padded when shorter than context_length."""
        mock_load_config.return_value = self.config
        
        # Mock Chronos pipeline
        mock_pipeline = Mock()
        mock_forecasts = np.random.randn(2, 5, 3)
        mock_pipeline.predict.return_value = mock_forecasts
        mock_from_pretrained.return_value = mock_pipeline
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        # Train first
        y_context = np.random.randn(10, 2)
        y_target = np.random.randn(3, 2)
        timestamps_context = np.arange(10)
        timestamps_target = np.arange(10, 13)
        
        model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        # Predict with short context (less than context_length=512)
        short_context = np.random.randn(100, 2)  # Much shorter than 512
        predictions = model.predict(short_context, timestamps_context, timestamps_target, "D")
        
        # Should still work and pad the context
        assert predictions.shape == (5, 3, 2)
        
        # Verify predict was called
        mock_pipeline.predict.assert_called()
        call_args = mock_pipeline.predict.call_args
        context_arg = call_args[1]['context']
        
        # Context should be padded to context_length
        assert context_arg.shape[1] == 512  # context_length
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    def test_predict_not_fitted_raises_error(self, mock_evaluator, mock_load_config):
        """Test that prediction without training raises error."""
        mock_load_config.return_value = self.config
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        y_context = np.random.randn(10, 2)
        timestamps_context = np.arange(10)
        timestamps_target = np.arange(10, 13)
        
        with pytest.raises(ValueError, match="Model not fitted. Call train\\(\\) first."):
            model.predict(y_context, timestamps_context, timestamps_target, "D")
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    def test_predict_freq_none_raises_error(self, mock_evaluator, mock_load_config):
        """Test that prediction with None freq raises error."""
        mock_load_config.return_value = self.config
        
        model = ChronosModel("config.yaml", "logs_path", {})
        model.is_fitted = True  # Mock fitted state
        
        y_context = np.random.randn(10, 2)
        timestamps_context = np.arange(10)
        timestamps_target = np.arange(10, 13)
        
        with pytest.raises(ValueError, match="Frequency \\(freq\\) must be provided from CSV data"):
            model.predict(y_context, timestamps_context, timestamps_target, None)
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    def test_get_model_summary(self, mock_evaluator, mock_load_config):
        """Test get_model_summary returns correct information."""
        mock_load_config.return_value = self.config
        
        model = ChronosModel("config.yaml", "logs_path", {})
        model.is_fitted = True
        
        summary = model.get_model_summary()
        
        assert summary['model_type'] == 'Chronos'
        assert summary['model_size'] == 'tiny'
        assert summary['context_length'] == 512
        assert summary['num_samples'] == 5
        assert summary['is_fitted'] == True
        assert 'device' in summary
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.metrics.evaluation.Evaluator')
    def test_compute_point_forecast_integration(self, mock_evaluator, mock_load_config):
        """Test integration with BaseModel's compute_point_forecast."""
        mock_load_config.return_value = self.config
        
        model = ChronosModel("config.yaml", "logs_path", {})
        
        # Create samples with shape (num_samples, forecast_horizon, num_targets)
        samples = np.array([
            [[1.0, 2.0], [3.0, 4.0]],  # Sample 1
            [[2.0, 3.0], [4.0, 5.0]],  # Sample 2
            [[0.0, 1.0], [2.0, 3.0]]   # Sample 3
        ])  # Shape: (3, 2, 2)
        
        point_forecast = model.compute_point_forecast(samples)
        
        # Should compute mean across samples
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])  # Mean across samples
        np.testing.assert_array_almost_equal(point_forecast, expected)
        assert point_forecast.shape == (2, 2)  # (forecast_horizon, num_targets)
