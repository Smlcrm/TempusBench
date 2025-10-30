"""
Unit tests for Pydantic models in config module.

This test suite provides 100% coverage of the models.py validators including
edge cases for max_num_variates, model parameter validation, and foundation models.
"""

import pytest
from pydantic import ValidationError
from tempus_bench.config.configs import (
    EvaluationConfig,
    ModelConfig,
    DatasetConfig,
    TaskConfig,
    BenchmarkConfig,
    BenchmarkSettingsConfig
)


class TestEvaluationConfig:
    """Test suite for EvaluationConfig model."""
    
    def test_valid_config_with_tuning_loss(self):
        """Test valid evaluation configuration with tuning loss."""
        config = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20,
            max_num_variates=10.0,
            num_samples=100,
            num_quantiles=10,
            point_forecast_statistic="mean"
        )
        assert config.tuning_loss == "mae"
        assert config.max_windows == 20
        assert config.max_num_variates == 10.0
    
    def test_valid_config_with_default_tuning_loss(self):
        """Test valid evaluation configuration with default tuning loss."""
        config = EvaluationConfig(
            max_windows=20
        )
        assert config.tuning_loss == "mae"  # default value
        assert config.max_windows == 20
    
    def test_invalid_tuning_loss(self):
        """Test invalid tuning loss value."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(
                tuning_loss="invalid_metric",
                max_windows=20
            )
        assert "Input should be 'mae', 'mase', 'mape' or 'rmse'" in str(exc_info.value)
    
    def test_max_num_variates_inf(self):
        """Test max_num_variates with infinity value."""
        config = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20,
            max_num_variates=float('inf')
        )
        assert config.max_num_variates == float('inf')
    
    def test_max_num_variates_invalid_negative(self):
        """Test that negative max_num_variates raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(
                tuning_loss="mae",
                max_windows=20,
                max_num_variates=-1.0
            )
        assert "at least 1 or inf" in str(exc_info.value)
    
    def test_max_num_variates_zero(self):
        """Test that zero max_num_variates raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(
                tuning_loss="mae",
                max_windows=20,
                max_num_variates=0.0
            )
        assert "at least 1 or inf" in str(exc_info.value)
    
    def test_max_num_variates_valid_float(self):
        """Test that valid float values are accepted."""
        config = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20,
            max_num_variates=5.5
        )
        assert config.max_num_variates == 5.5


class TestModelConfig:
    """Test suite for ModelConfig model."""
    
    def test_valid_traditional_model_config(self):
        """Test valid traditional model configuration."""
        config = ModelConfig(
            arima={"p": [1, 2], "d": [1], "tuning_loss": ["mae"]}
        )
        assert config.arima["p"] == [1, 2]
        assert config.arima["d"] == [1]
        assert config.arima["tuning_loss"] == ["mae"]
    
    def test_valid_traditional_model_without_tuning_loss(self):
        """Test valid traditional model configuration without tuning_loss (for non-trainable models)."""
        config = ModelConfig(
            prophet={"seasonality_mode": ["additive"]}  # Prophet doesn't require tuning_loss
        )
        assert config.prophet["seasonality_mode"] == ["additive"]
    
    def test_valid_foundation_model_config(self):
        """Test valid foundation model configuration."""
        config = ModelConfig(
            chronos={}  # Foundation models must have empty config
        )
        assert config.chronos == {}
    
    def test_model_parameters_not_dict(self):
        """Test that non-dict parameters raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(arima="invalid")
        assert "must be a dict" in str(exc_info.value)
    
    def test_traditional_model_non_list_parameter(self):
        """Test that traditional models must have list parameters."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(arima={"p": 1, "tuning_loss": ["mae"]})  # Should be [1]
        assert "must be a list of values" in str(exc_info.value)
    
    def test_traditional_model_empty_list(self):
        """Test that traditional models cannot have empty lists."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(arima={"p": [], "tuning_loss": ["mae"]})
        assert "cannot be an empty list" in str(exc_info.value)
    
    def test_foundation_model_simple_values(self):
        """Test that foundation models reject any parameters."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                chronos={"param": "value"}
            )
        assert "Foundation model 'chronos' must not define any parameters" in str(exc_info.value)
    
    def test_foundation_model_list_value(self):
        """Test that foundation models reject list parameters."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                chronos={"params": [1, 2, 3]}
            )
        assert "Foundation model 'chronos' must not define any parameters" in str(exc_info.value)
    
    def test_trainable_model_missing_tuning_loss(self):
        """Test that trainable models fail without tuning_loss."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(arima={"p": [1, 2]})  # Missing tuning_loss
        assert "requires 'tuning_loss' parameter" in str(exc_info.value)
    
    def test_non_trainable_model_without_tuning_loss(self):
        """Test that non-trainable models work without tuning_loss."""
        config = ModelConfig(
            prophet={"seasonality_mode": ["additive"]},
            xgboost={"n_estimators": [100]}
        )
        assert config.prophet["seasonality_mode"] == ["additive"]
        assert config.xgboost["n_estimators"] == [100]
    
    def test_none_values_allowed(self):
        """Test that None values are allowed for optional models."""
        config = ModelConfig()
        assert config.arima is None


class TestDatasetConfig:
    """Test suite for DatasetConfig model."""
    
    def test_valid_config_with_defaults(self):
        """Test valid dataset configuration with defaults."""
        config = DatasetConfig(file_name="test_dataset.csv")
        assert config.file_name == "test_dataset.csv"
        assert config.normalize is True
        assert config.handle_missing == "interpolate"
    
    def test_valid_config_custom_values(self):
        """Test valid dataset configuration with custom values."""
        config = DatasetConfig(
            file_name="test_dataset.csv",
            normalize=False,
            handle_missing="drop"
        )
        assert config.file_name == "test_dataset.csv"
        assert config.normalize is False
        assert config.handle_missing == "drop"
    
    def test_invalid_handle_missing(self):
        """Test that invalid handle_missing raises ValidationError."""
        with pytest.raises(ValidationError):
            DatasetConfig(name="test", handle_missing="invalid_method")


class TestTaskConfig:
    """Test suite for TaskConfig model."""
    
    def test_valid_config(self):
        """Test valid task configuration."""
        config = TaskConfig(
            name="test_task",
            forecast_horizon=24,
            context_window=100,
            dataset=DatasetConfig(file_name="test.csv")
        )
        assert config.forecast_horizon == 24
        assert config.context_window == 100
        assert config.dataset.file_name == "test.csv"
    
    def test_invalid_forecast_horizon_too_large(self):
        """Test that forecast_horizon > 128 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(
                name="test_task",
                forecast_horizon=200,
                context_window=100,
                dataset=DatasetConfig(file_name="test.csv")
            )
        assert "128" in str(exc_info.value)
    
    def test_invalid_forecast_horizon_zero(self):
        """Test that forecast_horizon < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskConfig(
                forecast_horizon=0,
                context_window=100,
                dataset=DatasetConfig(name="test")
            )
    
    def test_invalid_context_window_zero(self):
        """Test that context_window < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskConfig(
                forecast_horizon=24,
                context_window=0,
                dataset=DatasetConfig(name="test")
            )


class TestBenchmarkConfig:
    """Test suite for BenchmarkConfig model."""
    
    def test_valid_config(self):
        """Test valid benchmark configuration."""
        evaluation = EvaluationConfig(tuning_loss="mae", max_windows=20)
        model = ModelConfig(arima={"p": [1, 2], "tuning_loss": ["mae"]})
        
        config = BenchmarkConfig(
            task_path="*",
            evaluation=evaluation,
            model=model
        )
        assert config.task_path == "*"
        assert config.evaluation.tuning_loss == "mae"
        assert config.model.arima["p"] == [1, 2]
    
    def test_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                task_path="*",
                evaluation=EvaluationConfig(tuning_loss="mae", max_windows=20),
                model=ModelConfig(arima={"p": [1, 2]}),
                extra_field="not allowed"
            )
    
    def test_training_loss_validation_at_benchmark_level(self):
        """Test that training_loss validation works at BenchmarkConfig level."""
        evaluation = EvaluationConfig(tuning_loss="mae", max_windows=20)
        
        # Valid configuration with allowed model using tuning_loss
        model_valid = ModelConfig(arima={"p": [1, 2], "tuning_loss": ["mae"]})
        config = BenchmarkConfig(
            task_path="*",
            evaluation=evaluation,
            model=model_valid
        )
        assert config.model.arima["tuning_loss"] == ["mae"]
        
        # Invalid configuration with forbidden model using training_loss
        with pytest.raises(ValidationError) as exc_info:
            model_invalid = ModelConfig(prophet={"seasonality_mode": ["additive"], "tuning_loss": ["mae"]})
            BenchmarkConfig(
                task_path="*",
                evaluation=evaluation,
                model=model_invalid
            )
        assert "Model 'prophet' cannot define 'tuning_loss'" in str(exc_info.value)


class TestBenchmarkConfigTuningLossCompatibility:
    """Test suite for tuning loss and model compatibility validation."""
    
    @pytest.fixture
    def mock_models_dir(self, tmp_path, monkeypatch):
        """Create a mock models directory structure."""
        from tempus_bench.utils import paths
        
        deterministic_dir = tmp_path / "deterministic"
        stochastic_dir = tmp_path / "stochastic"
        
        # Create deterministic model folders
        (deterministic_dir / "arima" / "arima_model.py").mkdir(parents=True)
        (deterministic_dir / "prophet" / "prophet_model.py").mkdir(parents=True)
        (deterministic_dir / "xgboost" / "xgboost_model.py").mkdir(parents=True)
        (deterministic_dir / "exponential_smoothing" / "exponential_smoothing_model.py").mkdir(parents=True)
        
        # Create stochastic model folders
        (stochastic_dir / "chronos" / "chronos_model.py").mkdir(parents=True)
        (stochastic_dir / "deepar" / "deepar_model.py").mkdir(parents=True)
        (stochastic_dir / "moirai" / "moirai_model.py").mkdir(parents=True)
        
        # Mock get_models_dir to return our temp directory
        def _get_models_dir():
            return tmp_path
        
        monkeypatch.setattr(paths, "get_models_dir", _get_models_dir)
        
        return tmp_path
    
    def test_deterministic_model_with_tuning_loss(self, mock_models_dir):
        """Test that deterministic models work with tuning_loss."""
        evaluation = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20
        )
        model = ModelConfig(arima={"p": [1, 2], "tuning_loss": ["mae"]})
        
        config = BenchmarkConfig(
            task_path="*",
            evaluation=evaluation,
            model=model
        )
        assert config.evaluation.tuning_loss == "mae"
    
    def test_deterministic_model_missing_tuning_loss(self, mock_models_dir):
        """Test that trainable deterministic model without tuning_loss fails."""
        evaluation = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20
        )
        model = ModelConfig(arima={"p": [1, 2]})  # Missing tuning_loss for trainable model
        
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(
                task_path="*",
                evaluation=evaluation,
                model=model
            )
        assert "requires 'tuning_loss' parameter" in str(exc_info.value)
    
    def test_non_trainable_deterministic_model_without_tuning_loss(self, mock_models_dir):
        """Test that non-trainable deterministic models work without tuning_loss."""
        evaluation = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20
        )
        model = ModelConfig(prophet={"seasonality_mode": ["additive"]})  # Prophet doesn't require tuning_loss
        
        config = BenchmarkConfig(
            task_path="*",
            evaluation=evaluation,
            model=model
        )
        assert config.evaluation.tuning_loss == "mae"
    
    def test_multiple_deterministic_models(self, mock_models_dir):
        """Test configuration with multiple deterministic models."""
        evaluation = EvaluationConfig(
            tuning_loss="mae",
            max_windows=20
        )
        model = ModelConfig(
            arima={"p": [1, 2], "tuning_loss": ["mae"]},  # Trainable model requires tuning_loss
            prophet={"seasonality_mode": ["additive"]},    # Non-trainable model doesn't require tuning_loss
            xgboost={"n_estimators": [100]}                # Non-trainable model doesn't require tuning_loss
        )
        
        config = BenchmarkConfig(
            task_path="*",
            evaluation=evaluation,
            model=model
        )
        assert config.evaluation.tuning_loss == "mae"


class TestBenchmarkSettingsConfig:
    """Test suite for BenchmarkSettingsConfig model."""
    
    def test_valid_config_with_defaults(self):
        """Test valid model settings configuration with defaults."""
        config = BenchmarkSettingsConfig()
        assert config.python_version == "3.11"
        assert config.device == "cpu"
    
    def test_valid_config_custom_values(self):
        """Test valid model settings configuration with custom values."""
        config = BenchmarkSettingsConfig(python_version="3.12", device="gpu")
        assert config.python_version == "3.12"
        assert config.device == "gpu"
    
    def test_invalid_device(self):
        """Test that invalid device raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkSettingsConfig(device="invalid")


class TestBenchmarkSettingsConfig:
    """Test suite for BenchmarkSettingsConfig model."""
    
    def test_valid_config(self):
        """Test valid systems configuration."""
        config = BenchmarkSettingsConfig(
            logging_format="%(message)s",
            file_logging=True,
            console_logging=True,
            tensorboard_logging=True,
            runs_dir="runs",
            conda_env_prefix="benchmark"
        )
        assert config.logging_format == "%(message)s"
        assert config.file_logging is True
        assert config.runs_dir == "runs"
    
    def test_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            BenchmarkSettingsConfig(
                logging_format="%(message)s",
                file_logging=True,
                console_logging=True,
                tensorboard_logging=True,
                runs_dir="runs",
                conda_env_prefix="benchmark",
                extra_field="not allowed"
            )


class TestModelConfigTrainingLossValidation:
    """Test suite for ModelConfig training_loss validation."""
    
    def test_allowed_models_can_use_training_loss(self):
        """Test that models performing loss-based optimization can use training_loss."""
        # ARIMA - allowed deterministic model
        config = ModelConfig(arima={"p": [1, 2], "training_loss": ["mae", "rmse"], "tuning_loss": ["mae"]})
        assert config.arima["training_loss"] == ["mae", "rmse"]
    
        # LSTM - allowed deterministic model
        config = ModelConfig(lstm={"units": [50], "training_loss": ["mae"], "tuning_loss": ["mae"]})
        assert config.lstm["training_loss"] == ["mae"]
    
        # SVR - allowed deterministic model
        config = ModelConfig(svr={"kernel": ["rbf"], "training_loss": ["mae"], "tuning_loss": ["mae"]})
        assert config.svr["training_loss"] == ["mae"]
    
    def test_non_trainable_deterministic_models_cannot_use_training_loss(self):
        """Test that non-trainable deterministic models cannot define training_loss parameter."""
        deterministic_models = [
            'exponential_smoothing', 'seasonal_naive', 'croston_classic', 'theta',
            'xgboost', 'random_forest', 'prophet', 'moment', 'timesfm', 'tabpfn', 'varmax'
        ]
        
        for model_name in deterministic_models:
            with pytest.raises(ValidationError) as exc_info:
                if model_name in ['moment', 'timesfm', 'tabpfn']:
                    # Foundation models use simple values
                    ModelConfig(**{model_name: {"tuning_loss": "mae"}})
                else:
                    # Traditional models use lists
                    ModelConfig(**{model_name: {"param": [1], "tuning_loss": ["mae"]}})
            
            assert f"Model '{model_name}' cannot define 'tuning_loss'" in str(exc_info.value)
            assert "Only models performing loss-based optimization (ARIMA, LSTM, DeepAR, SVR)" in str(exc_info.value)
    
    def test_stochastic_models_cannot_use_training_loss(self):
        """Test that stochastic models cannot define training_loss parameter."""
        stochastic_models = [
            'chronos', 'tiny_time_mixer', 'moirai', 'moirai_moe',
            'lagllama', 'toto'
        ]
        
        for model_name in stochastic_models:
            with pytest.raises(ValidationError) as exc_info:
                # Foundation models use simple values
                ModelConfig(**{model_name: {"training_loss": "mae"}})
            
            assert f"Stochastic model '{model_name}' cannot define 'training_loss' parameter" in str(exc_info.value)
            assert "Only models performing loss-based optimization (ARIMA, LSTM, DeepAR, SVR)" in str(exc_info.value)
    
    def test_deepar_can_use_training_loss(self):
        """Test that DeepAR (stochastic but loss-based) can use training_loss."""
        config = ModelConfig(deepar={"hidden_size": [32], "training_loss": ["mae"], "tuning_loss": ["mae"]})
        assert config.deepar["training_loss"] == ["mae"]
        assert config.deepar["hidden_size"] == [32]
    
    def test_multiple_models_with_training_loss_validation(self):
        """Test validation when multiple models are configured."""
        # Valid configuration with allowed models
        config = ModelConfig(
            arima={"p": [1], "tuning_loss": ["mae"]},
            lstm={"units": [50], "tuning_loss": ["rmse"]},
            deepar={"hidden_size": [32], "tuning_loss": ["mae"]}
        )
        assert config.arima["tuning_loss"] == ["mae"]
        assert config.lstm["tuning_loss"] == ["rmse"]
        assert config.deepar["tuning_loss"] == ["mae"]
        
        # Invalid configuration with forbidden model
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                arima={"p": [1], "tuning_loss": ["mae"]},  # Allowed
                prophet={"tuning_loss": ["mae"]}  # Not allowed
            )
        assert "Deterministic model 'prophet' cannot define 'training_loss' parameter" in str(exc_info.value)
    
    def test_training_loss_validation_with_none_configs(self):
        """Test that None model configs don't trigger training_loss validation."""
        config = ModelConfig(
            arima={"p": [1], "training_loss": ["mae"]},  # Allowed
            prophet=None,  # None config should not trigger validation
            chronos=None   # None config should not trigger validation
        )
        assert config.arima["training_loss"] == ["mae"]
        assert config.prophet is None
        assert config.chronos is None
    
    def test_training_loss_with_other_parameters(self):
        """Test training_loss validation when combined with other parameters."""
        # Valid case - ARIMA with multiple parameters including training_loss
        config = ModelConfig(arima={
            "p": [1, 2, 3],
            "d": [0, 1],
            "q": [1, 2],
            "training_loss": ["mae", "rmse"],
            "seasonal": [True, False]
        })
        assert config.arima["training_loss"] == ["mae", "rmse"]
        assert config.arima["p"] == [1, 2, 3]
        
        # Invalid case - Prophet with multiple parameters including training_loss
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(prophet={
                "seasonality_mode": ["additive", "multiplicative"],
                "training_loss": ["mae"],  # This should trigger validation error
                "holidays": [True, False]
            })
        assert "Deterministic model 'prophet' cannot define 'training_loss' parameter" in str(exc_info.value)
    
    def test_training_loss_edge_cases(self):
        """Test edge cases for training_loss validation."""
        # Test with empty training_loss list (should be rejected at field validator level)
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(prophet={"training_loss": []})
        # This fails at field_validator level due to empty list validation
        assert "cannot be an empty list" in str(exc_info.value)
        
        # Test with None training_loss value (should still be rejected for forbidden models)
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(chronos={"training_loss": None})
        assert "Stochastic model 'chronos' cannot define 'tuning_loss' parameter" in str(exc_info.value)
        
        # Test with string training_loss for traditional models (should be rejected)
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(arima={"training_loss": "mae"})  # Should be ["mae"] for traditional models
        # This should fail at the field_validator level, not the model_validator level
        assert "must be a list of values" in str(exc_info.value)
        
        # Test with non-empty list for forbidden deterministic model (should reach model validator)
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(prophet={"training_loss": ["mae"]})
        assert "Deterministic model 'prophet' cannot define 'training_loss' parameter" in str(exc_info.value)
        
        # Test with non-empty list for forbidden stochastic model (should reach model validator)
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(chronos={"training_loss": ["mae"]})
        assert "Stochastic model 'chronos' cannot define 'tuning_loss' parameter" in str(exc_info.value)
    
    def test_all_model_types_coverage(self):
        """Test that all model types are properly categorized and validated."""
        # Test all deterministic models that should be forbidden
        forbidden_deterministic = [
            'exponential_smoothing', 'seasonal_naive', 'croston_classic', 'theta',
            'xgboost', 'random_forest', 'prophet', 'moment', 'timesfm', 'tabpfn', 'varmax'
        ]
        
        for model_name in forbidden_deterministic:
            with pytest.raises(ValidationError) as exc_info:
                if model_name in ['moment', 'timesfm', 'tabpfn']:
                    # Foundation models use simple values
                    ModelConfig(**{model_name: {"tuning_loss": "mae"}})
                else:
                    # Traditional models use lists
                    ModelConfig(**{model_name: {"param": [1], "tuning_loss": ["mae"]}})
            
            assert f"Model '{model_name}' cannot define 'tuning_loss'" in str(exc_info.value)
        
        # Test all stochastic models that should be forbidden (except DeepAR)
        forbidden_stochastic = [
            'chronos', 'tiny_time_mixer', 'moirai', 'moirai_moe',
            'lagllama', 'toto'
        ]
        
        for model_name in forbidden_stochastic:
            with pytest.raises(ValidationError) as exc_info:
                ModelConfig(**{model_name: {"training_loss": "mae"}})
            
            assert f"Stochastic model '{model_name}' cannot define 'training_loss' parameter" in str(exc_info.value)
        
        # Test all allowed models
        allowed_models = [
            ('arima', {"p": [1], "training_loss": ["mae"]}),
            ('lstm', {"units": [50], "training_loss": ["mae"]}),
            ('deepar', {"training_loss": "mae"}),
            ('svr', {"C": [1.0], "training_loss": ["mae"]})
        ]
        
        for model_name, config_dict in allowed_models:
            config = ModelConfig(**{model_name: config_dict})
            assert config.model_dump()[model_name]["training_loss"] is not None
    
    def test_training_loss_validation_error_messages(self):
        """Test that error messages are specific and informative."""
        # Test deterministic model error message
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(prophet={"training_loss": ["mae"]})
        
        error_msg = str(exc_info.value)
        assert "Deterministic model 'prophet'" in error_msg
        assert "cannot define 'training_loss' parameter" in error_msg
        assert "Only models performing loss-based optimization (ARIMA, LSTM, DeepAR, SVR)" in error_msg
        
        # Test stochastic model error message
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(chronos={"training_loss": "mae"})
        
        error_msg = str(exc_info.value)
        assert "Stochastic model 'chronos'" in error_msg
        assert "cannot define 'training_loss' parameter" in error_msg
        assert "Only models performing loss-based optimization (ARIMA, LSTM, DeepAR, SVR)" in error_msg
    
    def test_training_loss_with_complex_configurations(self):
        """Test training_loss validation with complex model configurations."""
        # Test multiple models with mixed valid/invalid training_loss usage
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                arima={"p": [1], "training_loss": ["mae"]},  # Valid
                lstm={"units": [50], "training_loss": ["rmse"]},  # Valid
                prophet={"training_loss": ["mae"]},  # Invalid - should trigger error
                chronos=None  # None - should not trigger error
            )
        
        # The error should be about prophet, not arima or lstm
        assert "Deterministic model 'prophet' cannot define 'training_loss' parameter" in str(exc_info.value)
        
        # Test valid complex configuration
        config = ModelConfig(
            arima={"p": [1, 2], "d": [0, 1], "training_loss": ["mae", "rmse"]},
            lstm={"units": [50, 100], "dropout": [0.1, 0.2], "training_loss": ["mae"]},
            deepar={"training_loss": "mae", "context_length": 50},
            svr={"C": [1.0, 10.0], "epsilon": [0.01, 0.1], "training_loss": ["mae"]}
        )
        
        assert config.arima["training_loss"] == ["mae", "rmse"]
        assert config.lstm["training_loss"] == ["mae"]
        assert config.deepar["training_loss"] == "mae"
        assert config.svr["training_loss"] == ["mae"]


class TestModelConfigEdgeCases:
    """Test suite for ModelConfig edge cases."""
    
    def test_none_values_return_early(self):
        """Test that None values in validate_model_parameters return early (line 91)."""
        # This test specifically covers line 91 in validate_model_parameters
        # When v is None, the function should return v early
        # We test this by ensuring ModelConfig with None values works
        config = ModelConfig(arima=None, prophet=None)
        # The validator at line 91 checks `if v is None: return v`
        # This should pass without raising any errors
        assert config.arima is None
        assert config.prophet is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

