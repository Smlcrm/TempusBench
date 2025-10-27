"""
Utility functions for loading model-specific configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .paths import get_models_dir

logger = logging.getLogger(__name__)


def load_model_settings(model_name: str) -> Dict[str, Any]:
    """
    Load model settings from settings.yaml file.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing model settings
        
    Raises:
        FileNotFoundError: If settings.yaml not found
        ValueError: If invalid model settings
    """
    models_dir = get_models_dir()
    
    # Find the model directory
    model_dir = None
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('__'):
            for model_folder in subdir.iterdir():
                if model_folder.is_dir() and model_folder.name == model_name:
                    model_dir = model_folder
                    break
            if model_dir:
                break
    
    if not model_dir:
        raise FileNotFoundError(f"Model directory not found for {model_name}")
    
    model_settings_path = model_dir / "settings.yaml"
    if not model_settings_path.exists():
        raise FileNotFoundError(f"Model settings not found: {model_settings_path}")
    
    try:
        with open(model_settings_path, 'r') as f:
            model_settings = yaml.safe_load(f)
        
        # Validate required fields
        if 'python_version' not in model_settings:
            model_settings['python_version'] = "3.11"
        if 'device' not in model_settings:
            model_settings['device'] = "cpu"
            
        return model_settings
        
    except Exception as e:
        raise ValueError(f"Invalid model settings at {model_settings_path}: {e}")


def load_task_config(task_path: str) -> Dict[str, Any]:
    """
    Load the first task configuration from task.yaml file.
    
    Args:
        task_path: Path to the task directory
        
    Returns:
        Dictionary containing the first task configuration
        
    Raises:
        FileNotFoundError: If task.yaml not found
        ValueError: If invalid task configuration
    """
    task_configs = load_all_task_configs(task_path)
    if not task_configs:
        raise ValueError("No task configurations found")
    return task_configs[0]


def load_all_task_configs(task_path: str) -> List[Dict[str, Any]]:
    """
    Load all task configurations from task.yaml file.
    
    Args:
        task_path: Path to the task directory
        
    Returns:
        List of dictionaries containing task configurations
        
    Raises:
        FileNotFoundError: If task.yaml not found
        ValueError: If invalid task configuration
    """
    task_dir = Path(task_path)
    task_config_path = task_dir / "task.yaml"
    
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_config_path}")
    
    try:
        with open(task_config_path, 'r') as f:
            # Load all YAML documents from the file
            task_configs = list(yaml.safe_load_all(f))
        
        # Filter out None values and validate each task config
        valid_configs = []
        for i, task_config in enumerate(task_configs):
            if task_config is None:
                continue
                
            # Validate required fields
            if 'task' not in task_config:
                logger.warning(f"Skipping invalid task config {i+1}: missing 'task' section")
                continue
            
            task_data = task_config['task']
            if 'dataset' not in task_data:
                logger.warning(f"Skipping invalid task config {i+1}: missing 'dataset' section")
                continue
            
            dataset_config = task_data['dataset']
            if 'name' not in dataset_config:
                logger.warning(f"Skipping invalid task config {i+1}: missing 'name' field in dataset config")
                continue
            
            # Set defaults for missing fields
            if 'normalize' not in dataset_config:
                dataset_config['normalize'] = True
            if 'handle_missing' not in dataset_config:
                dataset_config['handle_missing'] = 'interpolate'
            
            valid_configs.append(task_data)
        
        if not valid_configs:
            raise ValueError("No valid task configurations found")
            
        return valid_configs
        
    except Exception as e:
        raise ValueError(f"Invalid task configuration at {task_config_path}: {e}")


def get_python_version_for_model(model_name: str) -> str:
    """
    Get the Python version for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Python version string (e.g., "3.11")
    """
    try:
        model_settings = load_model_settings(model_name)
        return model_settings.get('python_version', '3.11')
    except Exception as e:
        logger.warning(f"Failed to load model settings for {model_name}: {e}, using default Python 3.11")
        return '3.11'


def get_device_for_model(model_name: str) -> str:
    """
    Get the device for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Device string ("cpu" or "gpu")
    """
    try:
        model_settings = load_model_settings(model_name)
        return model_settings.get('device', 'cpu')
    except Exception as e:
        logger.warning(f"Failed to load model settings for {model_name}: {e}, using default device cpu")
        return 'cpu'
