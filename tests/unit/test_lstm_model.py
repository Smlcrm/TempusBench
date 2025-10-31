"""
Unit tests for LSTM model.

This test suite ensures that LstmModel correctly implements the BaseModel interface,
handles both univariate and multivariate data, and trains and predicts without errors.
"""

import pytest
import numpy as np
from tempus_bench.models.lstm.lstm_model import LstmModel


class TestLstmModel:
    """Test suite for LstmModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Standard LSTM hyperparameters
        self.params = {
            'units': 32,
            'layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 2,  # Small number for testing
            'context_length': 10,
            'prediction_window': 4,
        }
        
        # Settings from settings.yaml
        self.settings = {
            'model_type': 'deterministic',
            'python_version': '3.11',
            'device': 'cpu',
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
        }
    
    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        model = LstmModel(self.params, self.settings)
        
        assert model.units == 32
        assert model.layers == 2
        assert model.dropout == 0.2
        assert model.learning_rate == 0.001
        assert model.is_fitted == False
        assert not hasattr(model, '_model')
    
    def test_train_univariate(self):
        """Test training with univariate data."""
        model = LstmModel(self.params, self.settings)
        
        # Create univariate data with enough length for context + prediction_window
        # We need at least context_length (10) + prediction_window (4) = 14 samples
        num_context = 30
        num_target = 8
        
        y_context = np.random.randn(num_context, 1).astype(np.float32)
        y_target = np.random.randn(num_target, 1).astype(np.float32)
        timestamps_context = np.arange(num_context)
        timestamps_target = np.arange(num_context, num_context + num_target)
        
        # Train with tuning_loss in kwargs
        result = model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            tuning_loss='mse'
        )
        
        assert result is model
        assert model.is_fitted == True
        assert hasattr(model, '_model')
        assert model._model is not None
    
    def test_train_multivariate(self):
        """Test training with multivariate data."""
        model = LstmModel(self.params, self.settings)
        
        # Create multivariate data (2 targets)
        num_context = 30
        num_target = 8
        num_targets = 2
        
        y_context = np.random.randn(num_context, num_targets).astype(np.float32)
        y_target = np.random.randn(num_target, num_targets).astype(np.float32)
        timestamps_context = np.arange(num_context)
        timestamps_target = np.arange(num_context, num_context + num_target)
        
        # Train with tuning_loss in kwargs
        result = model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            tuning_loss='mse'
        )
        
        assert result is model
        assert model.is_fitted == True
        assert hasattr(model, '_model')
    
    def test_train_minimum_data_length(self):
        """Test training with minimum required data length."""
        model = LstmModel(self.params, self.settings)
        
        # Minimum data: context_length (10) + prediction_window (4) = 14 samples
        min_length = model.context_length + model.prediction_window
        
        y_context = np.random.randn(model.context_length, 1).astype(np.float32)
        y_target = np.random.randn(model.prediction_window, 1).astype(np.float32)
        timestamps_context = np.arange(model.context_length)
        timestamps_target = np.arange(model.context_length, model.context_length + model.prediction_window)
        
        result = model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            tuning_loss='mse'
        )
        
        assert result is model
        assert model.is_fitted == True
    
    def test_train_insufficient_data_raises_error(self):
        """Test that training with insufficient data raises error."""
        model = LstmModel(self.params, self.settings)
        
        # Not enough data: only context_length samples (need context_length + prediction_window)
        y_context = np.random.randn(model.context_length - 1, 1).astype(np.float32)
        y_target = np.random.randn(model.prediction_window, 1).astype(np.float32)
        timestamps_context = np.arange(len(y_context))
        timestamps_target = np.arange(len(y_context), len(y_context) + len(y_target))
        
        with pytest.raises(ValueError, match="Input data must have at least"):
            model.train(
                y_context=y_context,
                y_target=y_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                tuning_loss='mse'
            )
    
    def test_predict_univariate(self):
        """Test prediction with univariate data."""
        model = LstmModel(self.params, self.settings)
        
        # Train first
        num_context = 30
        num_target = 8
        
        y_context_train = np.random.randn(num_context, 1).astype(np.float32)
        y_target_train = np.random.randn(num_target, 1).astype(np.float32)
        timestamps_context_train = np.arange(num_context)
        timestamps_target_train = np.arange(num_context, num_context + num_target)
        
        model.train(
            y_context=y_context_train,
            y_target=y_target_train,
            timestamps_context=timestamps_context_train,
            timestamps_target=timestamps_target_train,
            tuning_loss='mse'
        )
        
        # Predict
        forecast_length = 6
        y_context_predict = np.random.randn(model.context_length, 1).astype(np.float32)
        timestamps_context_predict = np.arange(model.context_length)
        timestamps_target_predict = np.arange(model.context_length, model.context_length + forecast_length)
        
        predictions = model.predict(
            y_context=y_context_predict,
            timestamps_context=timestamps_context_predict,
            timestamps_target=timestamps_target_predict
        )
        
        assert predictions.shape == (forecast_length, 1)
        assert isinstance(predictions, np.ndarray)
        assert np.all(np.isfinite(predictions))  # Check for NaN/Inf
    
    def test_predict_multivariate(self):
        """Test prediction with multivariate data."""
        model = LstmModel(self.params, self.settings)
        
        # Train first
        num_context = 30
        num_target = 8
        num_targets = 2
        
        y_context_train = np.random.randn(num_context, num_targets).astype(np.float32)
        y_target_train = np.random.randn(num_target, num_targets).astype(np.float32)
        timestamps_context_train = np.arange(num_context)
        timestamps_target_train = np.arange(num_context, num_context + num_target)
        
        model.train(
            y_context=y_context_train,
            y_target=y_target_train,
            timestamps_context=timestamps_context_train,
            timestamps_target=timestamps_target_train,
            tuning_loss='mse'
        )
        
        # Predict
        forecast_length = 6
        y_context_predict = np.random.randn(model.context_length, num_targets).astype(np.float32)
        timestamps_context_predict = np.arange(model.context_length)
        timestamps_target_predict = np.arange(model.context_length, model.context_length + forecast_length)
        
        predictions = model.predict(
            y_context=y_context_predict,
            timestamps_context=timestamps_context_predict,
            timestamps_target=timestamps_target_predict
        )
        
        assert predictions.shape == (forecast_length, num_targets)
        assert isinstance(predictions, np.ndarray)
        assert np.all(np.isfinite(predictions))  # Check for NaN/Inf
    
    def test_predict_not_fitted_raises_error(self):
        """Test that prediction without training raises error."""
        model = LstmModel(self.params, self.settings)
        
        y_context = np.random.randn(model.context_length, 1).astype(np.float32)
        timestamps_context = np.arange(model.context_length)
        timestamps_target = np.arange(model.context_length, model.context_length + 4)
        
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(
                y_context=y_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target
            )
    
    def test_predict_large_forecast_horizon(self):
        """Test prediction with forecast horizon larger than prediction_window."""
        model = LstmModel(self.params, self.settings)
        
        # Train first
        num_context = 30
        num_target = 8
        
        y_context_train = np.random.randn(num_context, 1).astype(np.float32)
        y_target_train = np.random.randn(num_target, 1).astype(np.float32)
        timestamps_context_train = np.arange(num_context)
        timestamps_target_train = np.arange(num_context, num_context + num_target)
        
        model.train(
            y_context=y_context_train,
            y_target=y_target_train,
            timestamps_context=timestamps_context_train,
            timestamps_target=timestamps_target_train,
            tuning_loss='mse'
        )
        
        # Predict with large forecast horizon (larger than prediction_window=4)
        forecast_length = 12
        y_context_predict = np.random.randn(model.context_length, 1).astype(np.float32)
        timestamps_context_predict = np.arange(model.context_length)
        timestamps_target_predict = np.arange(model.context_length, model.context_length + forecast_length)
        
        predictions = model.predict(
            y_context=y_context_predict,
            timestamps_context=timestamps_context_predict,
            timestamps_target=timestamps_target_predict
        )
        
        # Should predict exactly forecast_length steps
        assert predictions.shape == (forecast_length, 1)
        assert np.all(np.isfinite(predictions))
    
    def test_train_predict_full_cycle(self):
        """Test complete train-predict cycle."""
        model = LstmModel(self.params, self.settings)
        
        # Generate realistic time series data
        np.random.seed(42)
        num_context = 50
        num_target = 10
        num_targets = 1
        
        # Simple trend + noise
        trend = np.linspace(0, 10, num_context + num_target)
        noise = np.random.randn(num_context + num_target, num_targets) * 0.5
        data = (trend[:, np.newaxis] + noise).astype(np.float32)
        
        y_context = data[:num_context]
        y_target = data[num_context:num_context + num_target]
        timestamps_context = np.arange(num_context)
        timestamps_target = np.arange(num_context, num_context + num_target)
        
        # Train
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            tuning_loss='mse'
        )
        
        # Predict
        forecast_length = 8
        y_context_predict = data[:model.context_length]
        timestamps_context_predict = np.arange(model.context_length)
        timestamps_target_predict = np.arange(model.context_length, model.context_length + forecast_length)
        
        predictions = model.predict(
            y_context=y_context_predict,
            timestamps_context=timestamps_context_predict,
            timestamps_target=timestamps_target_predict
        )
        
        assert predictions.shape == (forecast_length, num_targets)
        assert np.all(np.isfinite(predictions))
        assert not np.all(predictions == 0)  # Should have some variation
    
    def test_get_params(self):
        """Test get_params returns model parameters."""
        model = LstmModel(self.params, self.settings)
        
        params = model.get_params()
        
        # Should return the validated params
        assert isinstance(params, type(model.params))
        assert params.units == 32
        assert params.layers == 2
    
    def test_set_params(self):
        """Test set_params updates model configuration."""
        model = LstmModel(self.params, self.settings)
        model.is_fitted = True  # Mock fitted state
        
        result = model.set_params(units=64, layers=3)
        
        assert result is model
        assert model.units == 64
        assert model.layers == 3
        assert model.is_fitted == False  # Should be reset when params change

