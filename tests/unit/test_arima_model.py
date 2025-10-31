"""
Unit tests for ARIMA model.

This test suite ensures that ArimaModel correctly implements the BaseModel interface,
handles both univariate and multivariate data, and integrates properly with the evaluation system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tempus_bench.models.arima.arima_model import ArimaModel


class TestArimaModel:
    """Test suite for ArimaModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'arima': {
                    'p': 1,
                    'd': 1,
                    'q': 1,
                    's': 1
                }
            },
            'evaluation': {
                'tuning_loss': 'mae',
                'metrics': ['mae', 'rmse']
            }
        }
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_init_with_valid_config(self, mock_evaluator, mock_load_config):
        """Test initialization with valid configuration."""
        mock_load_config.return_value = self.config
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        assert model.model_config['p'] == 1
        assert model.model_config['d'] == 1
        assert model.model_config['q'] == 1
        assert model.model_config['s'] == 1
        assert model.is_fitted == False
        assert model.model_ is None
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_init_missing_parameters_raises_error(self, mock_evaluator, mock_load_config):
        """Test that missing required parameters raise errors."""
        # Test missing 'p'
        config_missing_p = self.config.copy()
        config_missing_p['model']['arima'] = {'d': 1, 'q': 1, 's': 1}
        mock_load_config.return_value = config_missing_p
        
        with pytest.raises(ValueError, match="p must be specified in config"):
            ArimaModel("config.yaml", "logs_path", {})
        
        # Test missing 'd'
        config_missing_d = self.config.copy()
        config_missing_d['model']['arima'] = {'p': 1, 'q': 1, 's': 1}
        mock_load_config.return_value = config_missing_d
        
        with pytest.raises(ValueError, match="d must be specified in config"):
            ArimaModel("config.yaml", "logs_path", {})
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_train_univariate(self, mock_arima, mock_evaluator, mock_load_config):
        """Test training with univariate data."""
        mock_load_config.return_value = self.config
        
        # Mock ARIMA model and fit
        mock_fitted_model = Mock()
        mock_fitted_model.summary.return_value = "ARIMA Summary"
        mock_fitted_model.aic = 100.0
        mock_fitted_model.bic = 110.0
        mock_fitted_model.params = np.array([0.5, 0.3])
        
        mock_arima_instance = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_arima.return_value = mock_arima_instance
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        # Univariate data: shape (num_steps, 1)
        y_context = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_target = np.array([[6.0], [7.0]])
        timestamps_context = np.array([1, 2, 3, 4, 5])
        timestamps_target = np.array([6, 7])
        
        result = model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        assert result is model
        assert model.is_fitted == True
        assert model.model_ is mock_fitted_model
        mock_arima.assert_called_once()
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_train_multivariate(self, mock_arima, mock_evaluator, mock_load_config):
        """Test training with multivariate data."""
        mock_load_config.return_value = self.config
        
        # Mock ARIMA model and fit
        mock_fitted_model = Mock()
        mock_fitted_model.summary.return_value = "ARIMA Summary"
        mock_fitted_model.aic = 100.0
        mock_fitted_model.bic = 110.0
        mock_fitted_model.params = np.array([0.5, 0.3])
        
        mock_arima_instance = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_arima.return_value = mock_arima_instance
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        # Multivariate data: shape (num_steps, 2)
        y_context = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        y_target = np.array([[6.0, 7.0], [7.0, 8.0]])
        timestamps_context = np.array([1, 2, 3, 4, 5])
        timestamps_target = np.array([6, 7])
        
        result = model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        assert result is model
        assert model.is_fitted == True
        assert hasattr(model, 'models')
        assert len(model.models) == 2  # One model per variate
        assert model.model_ is mock_fitted_model  # First model for compatibility
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_predict_univariate(self, mock_arima, mock_evaluator, mock_load_config):
        """Test prediction with univariate data."""
        mock_load_config.return_value = self.config
        
        # Mock fitted model
        mock_fitted_model = Mock()
        mock_fitted_model.forecast.return_value = np.array([6.0, 7.0])
        
        mock_arima_instance = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_arima.return_value = mock_arima_instance
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        # Train first
        y_context = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_target = np.array([[6.0], [7.0]])
        timestamps_context = np.array([1, 2, 3, 4, 5])
        timestamps_target = np.array([6, 7])
        
        model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        # Predict
        predictions = model.predict(y_context, timestamps_context, timestamps_target, "D")
        
        assert predictions.shape == (2, 1)  # (forecast_horizon, num_targets)
        np.testing.assert_array_equal(predictions, np.array([[6.0], [7.0]]))
        mock_fitted_model.forecast.assert_called_once_with(steps=2, exog=None)
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_predict_multivariate(self, mock_arima, mock_evaluator, mock_load_config):
        """Test prediction with multivariate data."""
        mock_load_config.return_value = self.config
        
        # Mock fitted models
        mock_fitted_model1 = Mock()
        mock_fitted_model1.forecast.return_value = np.array([6.0, 7.0])
        
        mock_fitted_model2 = Mock()
        mock_fitted_model2.forecast.return_value = np.array([7.0, 8.0])
        
        mock_arima_instance1 = Mock()
        mock_arima_instance1.fit.return_value = mock_fitted_model1
        
        mock_arima_instance2 = Mock()
        mock_arima_instance2.fit.return_value = mock_fitted_model2
        
        # Mock ARIMA to return different instances
        mock_arima.side_effect = [mock_arima_instance1, mock_arima_instance2]
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        # Train with multivariate data
        y_context = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        y_target = np.array([[6.0, 7.0], [7.0, 8.0]])
        timestamps_context = np.array([1, 2, 3, 4, 5])
        timestamps_target = np.array([6, 7])
        
        model.train(y_context, y_target, timestamps_context, timestamps_target, "D")
        
        # Predict
        predictions = model.predict(y_context, timestamps_context, timestamps_target, "D")
        
        assert predictions.shape == (2, 2)  # (forecast_horizon, num_targets)
        expected = np.array([[6.0, 7.0], [7.0, 8.0]])
        np.testing.assert_array_equal(predictions, expected)
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_predict_not_fitted_raises_error(self, mock_evaluator, mock_load_config):
        """Test that prediction without training raises error."""
        mock_load_config.return_value = self.config
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        y_context = np.array([[1.0], [2.0], [3.0]])
        timestamps_context = np.array([1, 2, 3])
        timestamps_target = np.array([4, 5])
        
        with pytest.raises(ValueError, match="Model not fitted. Call train\\(\\) first."):
            model.predict(y_context, timestamps_context, timestamps_target, "D")
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_predict_freq_none_raises_error(self, mock_evaluator, mock_load_config):
        """Test that prediction with None freq raises error."""
        mock_load_config.return_value = self.config
        
        model = ArimaModel("config.yaml", "logs_path", {})
        model.is_fitted = True  # Mock fitted state
        
        y_context = np.array([[1.0], [2.0], [3.0]])
        timestamps_context = np.array([1, 2, 3])
        timestamps_target = np.array([4, 5])
        
        with pytest.raises(ValueError, match="Frequency \\(freq\\) must be provided from CSV data"):
            model.predict(y_context, timestamps_context, timestamps_target, None)
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_get_params(self, mock_evaluator, mock_load_config):
        """Test get_params returns model configuration."""
        mock_load_config.return_value = self.config
        
        model = ArimaModel("config.yaml", "logs_path", {})
        
        params = model.get_params()
        assert params == {'p': 1, 'd': 1, 'q': 1, 's': 1}
    
    @patch('tempus_bench.config.load_config')
    @patch('tempus_bench.pipeline.metric_registry.MetricRegistry')
    def test_set_params(self, mock_evaluator, mock_load_config):
        """Test set_params updates model configuration."""
        mock_load_config.return_value = self.config
        
        model = ArimaModel("config.yaml", "logs_path", {})
        model.is_fitted = True
        
        result = model.set_params(p=2, d=2)
        
        assert result is model
        assert model.model_config['p'] == 2
        assert model.model_config['d'] == 2
        assert model.is_fitted == False  # Should be reset when params change
