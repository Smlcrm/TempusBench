"""
Configuration module for the benchmarking pipeline.

This module provides Pydantic-based configuration models for all global settings
including task configuration, model parameters, evaluation metrics, and system settings.
"""

from typing import Dict, Any, Optional

from .models import (
    BenchmarkConfig,
    TaskConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    ModelSettingsConfig,
    SystemsConfig,
)
from .manager import ConfigManager, ConfigValidationError

# Global ConfigManager instance for singleton pattern
_config_manager: Optional[ConfigManager] = None
_current_config_path: Optional[str] = None

def load_config(config_path: str, logs_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration using ConfigManager singleton.

    This function ensures that configuration is validated before any execution.
    It maintains a global ConfigManager instance to avoid redundant validation
    of the same config file.

    Args:
        config_path: Path to the main benchmark configuration YAML file
        logs_path: Directory for log files

    Returns:
        Dictionary containing validated configuration with backward-compatible structure:
        - logging: System logging settings (from settings)
        - evaluation: Evaluation configuration (from main.evaluation)
        - task: Task configuration (from main.task - first task's config)
        - model: Model hyperparameters (from main.model)
        - settings: System settings (from settings)
        - model_settings: Model execution settings
        - task_configs: All task configurations

    Raises:
        ConfigValidationError: If configuration validation fails
    """
    global _config_manager, _current_config_path

    # Check if we need to create a new ConfigManager instance
    if _config_manager is None or _current_config_path != config_path:
        _config_manager = ConfigManager(config_path, logs_path)
        _current_config_path = config_path

    # Return backward-compatible dict structure
    config_dict = {
        # Backward compatibility - flatten main config to top level
        "evaluation": _config_manager.benchmark_config.evaluation.model_dump(),
        "model": _config_manager.model,

        # System settings (flattened for backward compatibility)
        "logging": {
            "console_logging": _config_manager.benchmark_settings.console_logging,
            "file_logging": _config_manager.benchmark_settings.file_logging,
            "tensorboard_logging": _config_manager.benchmark_settings.tensorboard_logging,
        },

        # Additional validated configs
        "settings": _config_manager.benchmark_settings.model_dump(),
        "model_settings": {name: config.model_dump() for name, config in _config_manager.model_settings.items()},
        "task_configs": {name: [task.model_dump() for task in tasks] for name, tasks in _config_manager.task.items()},
    }

    return config_dict

def get_config_manager() -> Optional[ConfigManager]:
    """
    Get the current global ConfigManager instance.

    Returns:
        The current ConfigManager instance, or None if not initialized
    """
    return _config_manager

__all__ = [
    "BenchmarkConfig",
    "TaskConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "ModelSettingsConfig",
    "SystemsConfig",
    "ConfigManager",
    "ConfigValidationError",
    "load_config",
    "get_config_manager",
]
