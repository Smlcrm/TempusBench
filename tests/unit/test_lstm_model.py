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
    
    def test_train_succeeds_when_concat_shorter_than_default_window_sum(self):
        """Runtime window clamp: n_ctx + n_tgt can be below hyperparam context_length + prediction_window."""
        model = LstmModel(self.params, self.settings)
        # Old implementation required len(context)+len(target) >= context_length + prediction_window (10+4=14).
        # With 9 + 4 = 13 rows, effective context becomes 9 and prediction 4 — training must work.
        y_context = np.random.randn(model.context_length - 1, 1).astype(np.float32)
        y_target = np.random.randn(model.prediction_window, 1).astype(np.float32)
        timestamps_context = np.arange(len(y_context))
        timestamps_target = np.arange(len(y_context), len(y_context) + len(y_target))
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            tuning_loss="mse",
        )
        assert model.is_fitted is True
        assert model._runtime_context_length == model.context_length - 1
        assert model._runtime_prediction_window == model.prediction_window
    
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

    def test_train_clamps_defaults_for_short_tasks(self):
        """
        Task context_window + forecast_horizon can be smaller than default 32+8
        (e.g. solar_100, gdp_years, patient_sparse, power_consumption_years).
        """
        params = {**self.params, "context_length": 32, "prediction_window": 8}
        model = LstmModel(params, self.settings)
        # Defaults 32 / 8 from merged hyperparams, runtime clamps to available rows.
        y_context = np.random.randn(12, 1).astype(np.float32)
        y_target = np.random.randn(4, 1).astype(np.float32)
        ts_c = np.arange(12)
        ts_t = np.arange(12, 16)
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_c,
            timestamps_target=ts_t,
            tuning_loss="mse",
        )
        assert model.is_fitted is True
        assert model._runtime_context_length == 12
        assert model._runtime_prediction_window == 4

    def test_mape_tuning_with_near_zero_labels_finishes(self):
        """MAPE-style loss must not produce NaNs when targets include zeros (sparse series)."""
        model = LstmModel(self.params, self.settings)
        y_context = np.linspace(0.0, 0.5, 20, dtype=np.float32).reshape(-1, 1)
        y_target = np.zeros((8, 1), dtype=np.float32)
        ts_c = np.arange(20)
        ts_t = np.arange(20, 28)
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_c,
            timestamps_target=ts_t,
            tuning_loss="mape",
        )
        preds = model.predict(
            y_context=y_context[-model._runtime_context_length :],
            timestamps_context=ts_c[-model._runtime_context_length :],
            timestamps_target=np.arange(8),
        )
        assert preds.shape == (8, 1)
        assert np.all(np.isfinite(preds))

    def test_covariate_train_and_predict(self):
        """Past + future covariates are concatenated; only targets are predicted."""
        model = LstmModel(self.params, self.settings)
        n_ctx, n_tgt = 18, 5
        n_cov = 1
        y_context = np.random.randn(n_ctx, 1).astype(np.float32)
        y_target = np.random.randn(n_tgt, 1).astype(np.float32)
        x_context = np.random.randn(n_ctx, n_cov).astype(np.float32)
        x_target = np.random.randn(n_tgt, n_cov).astype(np.float32)
        ts_c = np.arange(n_ctx)
        ts_t = np.arange(n_ctx, n_ctx + n_tgt)
        model.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_c,
            timestamps_target=ts_t,
            x_context=x_context,
            x_target=x_target,
            tuning_loss="mse",
        )
        hist_y = np.concatenate([y_context, y_target], axis=0)
        hist_x = np.concatenate([x_context, x_target], axis=0)
        fut_h = 5
        pred = model.predict(
            y_context=hist_y,
            timestamps_context=np.arange(len(hist_y)),
            timestamps_target=np.arange(len(hist_y), len(hist_y) + fut_h),
            x_context=hist_x,
            x_target=np.random.randn(fut_h, n_cov).astype(np.float32),
        )
        assert pred.shape == (fut_h, 1)
        assert np.all(np.isfinite(pred))

    def test_train_raises_if_only_one_covariate_array(self):
        model = LstmModel(self.params, self.settings)
        with pytest.raises(ValueError, match="both x_context and x_target"):
            model.train(
                y_context=np.random.randn(10, 1).astype(np.float32),
                y_target=np.random.randn(4, 1).astype(np.float32),
                timestamps_context=np.arange(10),
                timestamps_target=np.arange(10, 14),
                x_context=np.random.randn(10, 1).astype(np.float32),
                x_target=None,
                tuning_loss="mse",
            )

    def test_predict_requires_covariates_after_covariate_train(self):
        model = LstmModel(self.params, self.settings)
        n_ctx, n_tgt = 10, 4
        model.train(
            y_context=np.random.randn(n_ctx, 1).astype(np.float32),
            y_target=np.random.randn(n_tgt, 1).astype(np.float32),
            timestamps_context=np.arange(n_ctx),
            timestamps_target=np.arange(n_ctx, n_ctx + n_tgt),
            x_context=np.random.randn(n_ctx, 2).astype(np.float32),
            x_target=np.random.randn(n_tgt, 2).astype(np.float32),
            tuning_loss="mse",
        )
        hist = np.random.randn(n_ctx + n_tgt, 1).astype(np.float32)
        with pytest.raises(ValueError, match="requires x_context and x_target"):
            model.predict(
                y_context=hist,
                timestamps_context=np.arange(len(hist)),
                timestamps_target=np.arange(4),
            )

    def test_predict_rejects_covariates_when_trained_without(self):
        model = LstmModel(self.params, self.settings)
        model.train(
            y_context=np.random.randn(20, 1).astype(np.float32),
            y_target=np.random.randn(8, 1).astype(np.float32),
            timestamps_context=np.arange(20),
            timestamps_target=np.arange(20, 28),
            tuning_loss="mse",
        )
        hist = np.random.randn(28, 1).astype(np.float32)
        with pytest.raises(ValueError, match="trained without covariates"):
            model.predict(
                y_context=hist,
                timestamps_context=np.arange(28),
                timestamps_target=np.arange(3),
                x_context=np.random.randn(28, 1).astype(np.float32),
                x_target=np.random.randn(3, 1).astype(np.float32),
            )

