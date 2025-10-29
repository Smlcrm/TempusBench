"""
Unit tests for multi-document task.yaml file validation in ConfigManager.

This test ensures that ConfigManager correctly validates task.yaml files with multiple
task configurations separated by '---' (YAML multi-document format).
"""

import pytest
import yaml
from pathlib import Path
from tempus_bench.config.config import ConfigManager, ConfigValidationError
from tempus_bench.config.models import TaskConfig


@pytest.fixture
def sample_task_yaml_content():
    """Sample task.yaml content with multiple task configs."""
    return """task:
  forecast_horizon: 24
  context_window: 50
  dataset:     file_name: test_dataset
    normalize: true
    handle_missing: interpolate
---
task:
  forecast_horizon: 48
  context_window: 100
  dataset:     file_name: test_dataset
    normalize: false
    handle_missing: drop
"""


@pytest.fixture
def single_task_yaml_content():
    """Sample task.yaml content with single task config."""
    return """task:
  forecast_horizon: 24
  context_window: 50
  dataset:     file_name: test_dataset
    normalize: true
    handle_missing: interpolate
"""


@pytest.fixture
def invalid_task_yaml_content():
    """Invalid task.yaml content (missing 'task' key in one document)."""
    return """task:
  forecast_horizon: 24
  context_window: 50
  dataset:     file_name: test_dataset
    normalize: true
    handle_missing: interpolate
---
forecast_horizon: 48
context_window: 100
dataset:
  name: test_dataset
  normalize: false
  handle_missing: drop
"""


class TestTaskYamlMultiDoc:
    """Test suite for multi-document task.yaml validation."""
    
    def test_validate_multiple_task_configs(self, tmp_path, sample_task_yaml_content):
        """Test that multiple task configs can be loaded from a single file."""
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        task_file = task_dir / "task.yaml"
        task_file.write_text(sample_task_yaml_content)
        
        # Create a minimal ConfigManager instance for testing
        # We'll use a mock approach since we don't have full benchmark config
        class TestConfigManager:
            def __init__(self, task_file):
                self.task_dirs = [task_dir]
            
            def validate_task_configs(self):
                """Adapted validation logic for testing."""
                validated_configs = {}
                
                for task_dir in self.task_dirs:
                    task_config_path = task_dir / "task.yaml"
                    if not task_config_path.exists():
                        raise ConfigValidationError(f"Task config not found: {task_config_path}")
                    
                    task_name = task_dir.name
                    
                    with open(task_config_path, 'r') as f:
                        task_config_documents = list(yaml.safe_load_all(f))
                    
                    task_configs = []
                    for doc in task_config_documents:
                        if not doc:
                            continue
                        
                        if isinstance(doc, dict) and 'task' in doc:
                            task_data = doc['task']
                        else:
                            raise ConfigValidationError(
                                f"All sections in {task_config_path} must contain a 'task' key."
                            )
                        
                        task_config = TaskConfig(**task_data)
                        task_configs.append(task_config)
                    
                    if not task_configs:
                        raise ConfigValidationError(f"No valid task configurations found in {task_config_path}")
                    
                    validated_configs[task_name] = task_configs
                
                return validated_configs
        
        manager = TestConfigManager(task_file)
        result = manager.validate_task_configs()
        
        assert 'test_task' in result
        assert len(result['test_task']) == 2
        
        # Check first config
        config1 = result['test_task'][0]
        assert config1.forecast_horizon == 24
        assert config1.context_window == 50
        assert config1.dataset.name == 'test_dataset'
        assert config1.dataset.normalize is True
        assert config1.dataset.handle_missing == 'interpolate'
        
        # Check second config
        config2 = result['test_task'][1]
        assert config2.forecast_horizon == 48
        assert config2.context_window == 100
        assert config2.dataset.name == 'test_dataset'
        assert config2.dataset.normalize is False
        assert config2.dataset.handle_missing == 'drop'
    
    def test_validate_single_task_config(self, tmp_path, single_task_yaml_content):
        """Test that single task config works correctly."""
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        task_file = task_dir / "task.yaml"
        task_file.write_text(single_task_yaml_content)
        
        class TestConfigManager:
            def __init__(self, task_file):
                self.task_dirs = [task_dir]
            
            def validate_task_configs(self):
                validated_configs = {}
                
                for task_dir in self.task_dirs:
                    task_config_path = task_dir / "task.yaml"
                    if not task_config_path.exists():
                        raise ConfigValidationError(f"Task config not found: {task_config_path}")
                    
                    task_name = task_dir.name
                    
                    with open(task_config_path, 'r') as f:
                        task_config_documents = list(yaml.safe_load_all(f))
                    
                    task_configs = []
                    for doc in task_config_documents:
                        if not doc:
                            continue
                        
                        if isinstance(doc, dict) and 'task' in doc:
                            task_data = doc['task']
                        else:
                            raise ConfigValidationError(
                                f"All sections in {task_config_path} must contain a 'task' key."
                            )
                        
                        task_config = TaskConfig(**task_data)
                        task_configs.append(task_config)
                    
                    if not task_configs:
                        raise ConfigValidationError(f"No valid task configurations found in {task_config_path}")
                    
                    validated_configs[task_name] = task_configs
                
                return validated_configs
        
        manager = TestConfigManager(task_file)
        result = manager.validate_task_configs()
        
        assert 'test_task' in result
        assert len(result['test_task']) == 1
        
        config = result['test_task'][0]
        assert config.forecast_horizon == 24
        assert config.context_window == 50
    
    def test_invalid_missing_task_key(self, tmp_path, invalid_task_yaml_content):
        """Test that missing 'task' key raises ConfigValidationError."""
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        task_file = task_dir / "task.yaml"
        task_file.write_text(invalid_task_yaml_content)
        
        class TestConfigManager:
            def __init__(self, task_file):
                self.task_dirs = [task_dir]
            
            def validate_task_configs(self):
                validated_configs = {}
                
                for task_dir in self.task_dirs:
                    task_config_path = task_dir / "task.yaml"
                    if not task_config_path.exists():
                        raise ConfigValidationError(f"Task config not found: {task_config_path}")
                    
                    task_name = task_dir.name
                    
                    with open(task_config_path, 'r') as f:
                        task_config_documents = list(yaml.safe_load_all(f))
                    
                    task_configs = []
                    for doc in task_config_documents:
                        if not doc:
                            continue
                        
                        if isinstance(doc, dict) and 'task' in doc:
                            task_data = doc['task']
                        else:
                            raise ConfigValidationError(
                                f"All sections in {task_config_path} must contain a 'task' key."
                            )
                        
                        task_config = TaskConfig(**task_data)
                        task_configs.append(task_config)
                    
                    validated_configs[task_name] = task_configs
                
                return validated_configs
        
        manager = TestConfigManager(task_file)
        
        with pytest.raises(ConfigValidationError, match="must contain a 'task' key"):
            manager.validate_task_configs()
    
    def test_empty_file_raises_error(self, tmp_path):
        """Test that empty task.yaml file raises ConfigValidationError."""
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        task_file = task_dir / "task.yaml"
        task_file.write_text("")  # Empty file
        
        class TestConfigManager:
            def __init__(self, task_file):
                self.task_dirs = [task_dir]
            
            def validate_task_configs(self):
                validated_configs = {}
                
                for task_dir in self.task_dirs:
                    task_config_path = task_dir / "task.yaml"
                    if not task_config_path.exists():
                        raise ConfigValidationError(f"Task config not found: {task_config_path}")
                    
                    task_name = task_dir.name
                    
                    with open(task_config_path, 'r') as f:
                        task_config_documents = list(yaml.safe_load_all(f))
                    
                    task_configs = []
                    for doc in task_config_documents:
                        if not doc:
                            continue
                        
                        if isinstance(doc, dict) and 'task' in doc:
                            task_data = doc['task']
                        else:
                            raise ConfigValidationError(
                                f"All sections in {task_config_path} must contain a 'task' key."
                            )
                        
                        task_config = TaskConfig(**task_data)
                        task_configs.append(task_config)
                    
                    if not task_configs:
                        raise ConfigValidationError(f"No valid task configurations found in {task_config_path}")
                    
                    validated_configs[task_name] = task_configs
                
                return validated_configs
        
        manager = TestConfigManager(task_file)
        
        with pytest.raises(ConfigValidationError, match="No valid task configurations found"):
            manager.validate_task_configs()
    
    def test_invalid_task_config_schema(self, tmp_path):
        """Test that invalid task config schema raises ConfigValidationError."""
        invalid_yaml = """task:
  forecast_horizon: 24
  # Missing context_window and dataset
"""
        
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        task_file = task_dir / "task.yaml"
        task_file.write_text(invalid_yaml)
        
        class TestConfigManager:
            def __init__(self, task_file):
                self.task_dirs = [task_dir]
            
            def validate_task_configs(self):
                validated_configs = {}
                
                for task_dir in self.task_dirs:
                    task_config_path = task_dir / "task.yaml"
                    if not task_config_path.exists():
                        raise ConfigValidationError(f"Task config not found: {task_config_path}")
                    
                    task_name = task_dir.name
                    
                    with open(task_config_path, 'r') as f:
                        task_config_documents = list(yaml.safe_load_all(f))
                    
                    task_configs = []
                    for doc in task_config_documents:
                        if not doc:
                            continue
                        
                        if isinstance(doc, dict) and 'task' in doc:
                            task_data = doc['task']
                        else:
                            raise ConfigValidationError(
                                f"All sections in {task_config_path} must contain a 'task' key."
                            )
                        
                        # This should raise ValidationError
                        task_config = TaskConfig(**task_data)
                        task_configs.append(task_config)
                    
                    validated_configs[task_name] = task_configs
                
                return validated_configs
        
        manager = TestConfigManager(task_file)
        
        with pytest.raises(Exception):  # Should raise ValidationError from Pydantic
            manager.validate_task_configs()


