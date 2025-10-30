"""
Unit tests for ModelRouter class.

This test suite ensures that ModelRouter correctly discovers models,
validates model structure, and routes requests appropriately.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from tempus_bench.models.model_router import ModelRouter


class TestModelRouter:
    """Test suite for ModelRouter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_models_dir = Path("/mock/models")
        self.mock_deterministic_dir = self.mock_models_dir / "deterministic"
        self.mock_stochastic_dir = self.mock_models_dir / "stochastic"
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_init_discovers_models(self, mock_get_logger, mock_get_models_dir):
        """Test that initialization discovers available models."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock directory structure
        deterministic_items = [
            Mock(is_dir=lambda: True, name="arima"),
            Mock(is_dir=lambda: True, name="prophet"),
            Mock(is_dir=lambda: True, name="lstm"),
        ]
        stochastic_items = [
            Mock(is_dir=lambda: True, name="chronos"),
            Mock(is_dir=lambda: True, name="moirai"),
            Mock(is_dir=lambda: True, name="toto"),
        ]
        
        # Mock model files exist
        for item in deterministic_items + stochastic_items:
            item.__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = deterministic_items
        self.mock_stochastic_dir.iterdir.return_value = stochastic_items
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        assert "arima" in router.deterministic_models
        assert "prophet" in router.deterministic_models
        assert "lstm" in router.deterministic_models
        assert "chronos" in router.stochastic_models
        assert "moirai" in router.stochastic_models
        assert "toto" in router.stochastic_models
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_init_missing_deterministic_folder(self, mock_get_logger, mock_get_models_dir):
        """Test that missing deterministic folder raises SystemError."""
        mock_get_models_dir.return_value = self.mock_models_dir
        self.mock_deterministic_dir.exists.return_value = False
        
        with pytest.raises(SystemError, match="The models/deterministic folder is missing"):
            ModelRouter("logs_path")
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_init_missing_stochastic_folder(self, mock_get_logger, mock_get_models_dir):
        """Test that missing stochastic folder raises SystemError."""
        mock_get_models_dir.return_value = self.mock_models_dir
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = False
        
        with pytest.raises(SystemError, match="The models/stochastic folder is missing"):
            ModelRouter("logs_path")
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_validate_models_overlap_raises_error(self, mock_get_logger, mock_get_models_dir):
        """Test that overlapping models raise ValueError."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock overlapping models
        deterministic_items = [Mock(is_dir=lambda: True, name="overlap_model")]
        stochastic_items = [Mock(is_dir=lambda: True, name="overlap_model")]
        
        for item in deterministic_items + stochastic_items:
            item.__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = deterministic_items
        self.mock_stochastic_dir.iterdir.return_value = stochastic_items
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        with pytest.raises(ValueError, match="Models cannot be both deterministic and stochastic"):
            ModelRouter("logs_path")
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_available_models(self, mock_get_logger, mock_get_models_dir):
        """Test get_available_models returns correct structure."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock models
        deterministic_items = [
            Mock(is_dir=lambda: True, name="arima"),
            Mock(is_dir=lambda: True, name="prophet"),
        ]
        stochastic_items = [
            Mock(is_dir=lambda: True, name="chronos"),
            Mock(is_dir=lambda: True, name="moirai"),
        ]
        
        for item in deterministic_items + stochastic_items:
            item.__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = deterministic_items
        self.mock_stochastic_dir.iterdir.return_value = stochastic_items
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        available = router.get_available_models()
        
        assert "deterministic" in available
        assert "stochastic" in available
        assert available["deterministic"] == ["arima", "prophet"]
        assert available["stochastic"] == ["chronos", "moirai"]
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_model_path_by_model_type_deterministic_model_type_deterministic_model(self, mock_get_logger, mock_get_models_dir):
        """Test routing for deterministic task with deterministic model."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock deterministic model
        deterministic_items = [Mock(is_dir=lambda: True, name="arima")]
        deterministic_items[0].__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = deterministic_items
        self.mock_stochastic_dir.iterdir.return_value = []
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        folder_path, file_name, class_name = router.get_model_path_by_model_type("arima", "deterministic")
        
        assert folder_path == str(self.mock_models_dir / "deterministic" / "arima")
        assert file_name == "arima_model"
        assert class_name == "ArimaModel"
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_model_path_by_model_type_stochastic_model_type_stochastic_model(self, mock_get_logger, mock_get_models_dir):
        """Test routing for stochastic task with stochastic model."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock stochastic model
        stochastic_items = [Mock(is_dir=lambda: True, name="chronos")]
        stochastic_items[0].__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = []
        self.mock_stochastic_dir.iterdir.return_value = stochastic_items
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        folder_path, file_name, class_name = router.get_model_path_by_model_type("chronos", "stochastic")
        
        assert folder_path == str(self.mock_models_dir / "stochastic" / "chronos")
        assert file_name == "chronos_model"
        assert class_name == "ChronosModel"
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_model_path_by_model_type_deterministic_model_type_stochastic_model(self, mock_get_logger, mock_get_models_dir):
        """Test routing for deterministic task with stochastic model (extract point forecast)."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock stochastic model
        stochastic_items = [Mock(is_dir=lambda: True, name="chronos")]
        stochastic_items[0].__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = []
        self.mock_stochastic_dir.iterdir.return_value = stochastic_items
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        folder_path, file_name, class_name = router.get_model_path_by_model_type("chronos", "deterministic")
        
        assert folder_path == str(self.mock_models_dir / "stochastic" / "chronos")
        assert file_name == "chronos_model"
        assert class_name == "ChronosModel"
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_model_path_by_model_type_stochastic_model_type_deterministic_model_raises_error(self, mock_get_logger, mock_get_models_dir):
        """Test that stochastic task with deterministic model raises error."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        # Mock deterministic model
        deterministic_items = [Mock(is_dir=lambda: True, name="arima")]
        deterministic_items[0].__truediv__ = lambda x: Mock(exists=lambda: True)
        
        self.mock_deterministic_dir.iterdir.return_value = deterministic_items
        self.mock_stochastic_dir.iterdir.return_value = []
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        with pytest.raises(ValueError, match="Model 'arima' is deterministic but task requires stochastic forecasting"):
            router.get_model_path_by_model_type("arima", "stochastic")
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_model_path_by_model_type_invalid_model_type(self, mock_get_logger, mock_get_models_dir):
        """Test that invalid task type raises error."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        self.mock_deterministic_dir.iterdir.return_value = []
        self.mock_stochastic_dir.iterdir.return_value = []
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        with pytest.raises(ValueError, match="Model .* not found"):
            router.get_model_path_by_model_type("arima", "invalid")
    
    @patch('tempus_bench.models.model_router.get_models_dir')
    @patch('tempus_bench.models.model_router.get_logger')
    def test_get_model_path_by_model_type_unknown_model(self, mock_get_logger, mock_get_models_dir):
        """Test that unknown model raises error."""
        mock_get_models_dir.return_value = self.mock_models_dir
        
        self.mock_deterministic_dir.iterdir.return_value = []
        self.mock_stochastic_dir.iterdir.return_value = []
        self.mock_deterministic_dir.exists.return_value = True
        self.mock_stochastic_dir.exists.return_value = True
        
        router = ModelRouter("logs_path")
        
        with pytest.raises(ValueError, match="Unknown model 'unknown_model'"):
            router.get_model_path_by_model_type("unknown_model", "deterministic")
    
    def test_generate_class_name(self):
        """Test _generate_class_name generates correct class names."""
        with patch('tempus_bench.models.model_router.get_models_dir'):
            with patch('tempus_bench.models.model_router.get_logger'):
                router = ModelRouter("logs_path")
                
                assert router._generate_class_name("arima", "deterministic") == "ArimaModel"
                assert router._generate_class_name("moirai_moe", "stochastic") == "MoiraiMoeModel"
                assert router._generate_class_name("lag_llama", "stochastic") == "LagLlamaModel"
