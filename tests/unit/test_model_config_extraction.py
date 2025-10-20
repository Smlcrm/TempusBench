"""
Unit tests for model configuration extraction in BaseModel.

This test ensures that BaseModel strictly enforces that config['model']
contains exactly one selected hyperparameter dictionary, not a grid or list.
"""

import pytest
import numpy as np
from benchmarking_pipeline.models.base_model import BaseModel


class DummyModel(BaseModel):
    """Minimal concrete implementation of BaseModel for testing."""
    
    def train(self, y_context=None, y_target=None, timestamps_context=None, 
              timestamps_target=None, freq=None):
        self.is_fitted = True
        return self
    
    def predict(self, y_context=None, timestamps_context=None, 
                timestamps_target=None, freq=None):
        return np.array([[1.0]])


class TestModelConfigExtraction:
    """Test suite for model config extraction validation."""
    
    def test_valid_single_model_config(self):
        """Test that a valid single model config is accepted."""
        config = {
            'model': {
                'dummy': {
                    'param1': 10,
                    'param2': 'value'
                }
            },
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        model = DummyModel(config)
        assert model.model_config == {'param1': 10, 'param2': 'value'}
    
    def test_empty_model_config_accepted(self):
        """Test that an empty model config dict (for foundation models) is accepted."""
        config = {
            'model': {
                'foundation_model': {}
            },
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        model = DummyModel(config)
        assert model.model_config == {}
    
    def test_missing_model_section_raises(self):
        """Test that missing 'model' section raises ValueError."""
        config = {
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        with pytest.raises(ValueError, match="Missing 'model' section"):
            DummyModel(config)
    
    def test_multiple_models_raises(self):
        """Test that multiple model entries raise ValueError."""
        config = {
            'model': {
                'model1': {'param': 1},
                'model2': {'param': 2}
            },
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        with pytest.raises(ValueError, match="exactly one model entry"):
            DummyModel(config)
    
    def test_model_params_as_list_raises(self):
        """Test that model params as a list (grid) raises ValueError."""
        config = {
            'model': {
                'dummy': [{'param': 1}, {'param': 2}]
            },
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        with pytest.raises(ValueError, match="must be a dict"):
            DummyModel(config)
    
    def test_model_params_as_none_raises(self):
        """Test that model params as None raises ValueError."""
        config = {
            'model': {
                'dummy': None
            },
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        with pytest.raises(ValueError, match="must be a dict"):
            DummyModel(config)
    
    def test_model_section_not_dict_raises(self):
        """Test that model section not being a dict raises ValueError."""
        config = {
            'model': ['list', 'of', 'models'],
            'task': {'tuning_loss': 'mae'},
            'evaluation': {'metrics': ['mae', 'rmse']}
        }
        
        with pytest.raises(ValueError, match="exactly one model entry"):
            DummyModel(config)

