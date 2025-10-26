"""
Configuration module for the benchmarking pipeline.

This module provides Pydantic-based configuration models for all global settings
including task configuration, model parameters, evaluation metrics, and system settings.
"""

from .models import (
    BenchmarkConfig,
    TaskConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    SystemConfig,
    LoggingConfig,
    PathsConfig
)
from .validator import ConfigValidator
from .manager import ConfigManager

# Backward compatibility functions
def load_config(config_path: str):
    """Load configuration with automatic merging with defaults."""
    manager = ConfigManager()
    config = manager.load_from_file(config_path)
    return config.model_dump()

def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file."""
    manager = ConfigManager()
    return manager.validate_config_file(config_path)

__all__ = [
    "BenchmarkConfig",
    "TaskConfig", 
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "SystemConfig",
    "LoggingConfig",
    "PathsConfig",
    "ConfigValidator",
    "ConfigManager",
    "load_config",
    "validate_config_file"
]
