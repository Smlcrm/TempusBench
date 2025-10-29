"""
Configuration module for the benchmarking pipeline.

This module provides Pydantic-based configuration models for all global settings
including task configuration, model parameters, evaluation metrics, and system settings.
"""

from typing import Optional

from .models import (
    BenchmarkConfig,
    TaskConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    BenchmarkSettingsConfig,
    JobConfig,
)
from .config import ConfigManager, ConfigValidationError

# Global ConfigManager instance for singleton pattern
_config_manager: Optional[ConfigManager] = None
_current_config_path: Optional[str] = None

def load_config_manager(config_path: str, logs_path: str) -> "ConfigManager":
    """
    Load and validate benchmark configuration as a singleton.

    This function uses a global ConfigManager instance to ensure that configuration files
    are parsed, validated, and cached efficiently. Only re-initializes the configuration
    manager if `config_path` changes.

    Args:
        config_path (str): Absolute or relative path to the benchmark YAML configuration file.
        logs_path (str): Path to the logs directory to associate with this configuration.

    Returns:
        ConfigManager: Singleton instance with validated configs and accessors
            for all loaded configs and helper methods.

    Raises:
        ConfigValidationError: If configuration schema validation fails, or required files are missing.
    """
    global _config_manager, _current_config_path

    # Check if we need to create a new ConfigManager instance
    if _config_manager is None or _current_config_path != config_path:
        _config_manager = ConfigManager(config_path, logs_path)
        _current_config_path = config_path

    return _config_manager

__all__ = [
    "BenchmarkConfig",
    "TaskConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "BenchmarkSettingsConfig",
    "JobConfig",
    "ConfigManager",
    "ConfigValidationError",
    "load_config",
    "get_config_manager",
]
