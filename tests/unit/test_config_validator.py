"""
Tests for the configuration validator utility.
"""

import pytest
import tempfile
import os
import yaml
from benchmarking_pipeline.utils.config_validator import (
    ConfigValidator, 
    ConfigValidationError, 
    validate_config_file, 
    validate_config_dict
)


class TestConfigValidator:
    """Test the ConfigValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConfigValidator instance."""
        return ConfigValidator()
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration dictionary."""
        return {
            'test_type': 'deterministic',
            'dataset': {
                'name': 'test_dataset',
                'path': 'test/path',
                'frequency': 'D',
                'split_ratio': [0.8, 0.1, 0.1],
                'normalize': True,
                'handle_missing': 'interpolate'
            },
            'model': {
                'arima': {
                    'p': [1],
                    'd': [1],
                    'q': [1],
                    'training_loss': ['mae'],
                    'forecast_horizon': [10]
                }
            },
            'evaluation': {
                'type': 'deterministic',
                'metrics': ['mae']
            }
        }
    
    @pytest.mark.unit
    def test_valid_config(self, validator, valid_config):
        """Test that a valid configuration passes validation."""
        assert validator.validate_config(valid_config) is True
    
    @pytest.mark.unit
    def test_missing_required_field(self, validator, valid_config):
        """Test that missing required fields cause validation errors."""
        del valid_config['test_type']
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "Required field 'test_type' is missing" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_test_type(self, validator, valid_config):
        """Test that invalid test_type values cause validation errors."""
        valid_config['test_type'] = 'invalid_type'
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "not allowed" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_split_ratio(self, validator, valid_config):
        """Test that invalid split ratios cause validation errors."""
        valid_config['dataset']['split_ratio'] = [0.5, 0.5, 0.0]  # Sums to 1.0 but has 0
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_model_name(self, validator, valid_config):
        """Test that invalid model names cause validation errors."""
        # Add an invalid model to the model section
        valid_config['model']['invalid_model'] = {'some_param': [1]}
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "Unknown model" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_forbidden_unknown_fields(self, validator, valid_config):
        """Test that unknown fields (like target_cols) are forbidden."""
        valid_config['dataset']['target_cols'] = ['y']
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        assert "Unknown field 'target_cols' is not allowed" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_forecast_horizon(self, validator, valid_config):
        """Test that invalid forecast_horizon types cause validation errors."""
        valid_config['model']['arima']['forecast_horizon'] = "invalid"  # Must be a list
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "Expected list" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_handle_missing(self, validator, valid_config):
        """Test that invalid handle_missing values cause validation errors."""
        valid_config['dataset']['handle_missing'] = 'invalid_method'
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "not allowed" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_evaluation_type(self, validator, valid_config):
        """Test that invalid evaluation types cause validation errors."""
        valid_config['evaluation']['type'] = 'invalid_type'
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "not allowed" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_invalid_metrics(self, validator, valid_config):
        """Test that invalid metrics cause validation errors."""
        valid_config['evaluation']['metrics'] = ['invalid_metric']
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_config(valid_config)
        
        assert "Unknown metrics" in str(exc_info.value)


class TestConfigValidationFunctions:
    """Test the convenience validation functions."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration dictionary."""
        return {
            'test_type': 'deterministic',
            'dataset': {
                'name': 'test_dataset',
                'path': 'test/path',
                'frequency': 'D',
                'split_ratio': [0.8, 0.1, 0.1],
                'normalize': True,
                'handle_missing': 'interpolate'
            },
            'model': {
                'arima': {
                    'p': [1],
                    'd': [1],
                    'q': [1],
                    'training_loss': ['mae'],
                    'forecast_horizon': [10]
                }
            },
            'evaluation': {
                'type': 'deterministic',
                'metrics': ['mae']
            }
        }
    
    @pytest.mark.unit
    def test_validate_config_dict(self, valid_config):
        """Test the validate_config_dict convenience function."""
        assert validate_config_dict(valid_config) is True
    
    @pytest.mark.unit
    def test_validate_config_file(self, valid_config):
        """Test the validate_config_file convenience function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_file = f.name
        
        try:
            assert validate_config_file(temp_file) is True
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.unit
    def test_validate_config_file_not_found(self):
        """Test that validate_config_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            validate_config_file('/nonexistent/file.yaml')
    
    @pytest.mark.unit
    def test_validate_config_file_invalid_yaml(self):
        """Test that validate_config_file handles invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_config_file(temp_file)
            assert "Invalid YAML format" in str(exc_info.value)
        finally:
            os.unlink(temp_file)


class TestConfigValidationError:
    """Test the ConfigValidationError exception class."""
    
    @pytest.mark.unit
    def test_error_with_field_path(self):
        """Test ConfigValidationError with field path."""
        error = ConfigValidationError("Invalid value", "dataset.forecast_horizon")
        assert "dataset.forecast_horizon" in str(error)
        assert "Invalid value" in str(error)
    
    @pytest.mark.unit
    def test_error_without_field_path(self):
        """Test ConfigValidationError without field path."""
        error = ConfigValidationError("Missing required field")
        assert "Missing required field" in str(error)
        assert "field_path" not in str(error)
    
    @pytest.mark.unit
    def test_error_with_details(self):
        """Test ConfigValidationError with additional details."""
        error = ConfigValidationError("Invalid value", "test_field", "Additional context")
        assert "test_field" in str(error)
        assert "Invalid value" in str(error)
