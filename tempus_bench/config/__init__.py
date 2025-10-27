"""
Configuration module for the benchmarking pipeline.

This module provides Pydantic-based configuration models for all global settings
including task configuration, model parameters, evaluation metrics, and system settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from .models import (
    BenchmarkConfig,
    TaskConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    ModelSettingsConfig,
    SystemsConfig,
)
from .validator import ConfigValidator


def validate_config_file(config_path: str, logs_dir: str) -> bool:
    """
    Validate a configuration file using the ConfigValidator.
    
    Args:
        config_path: Path to the configuration file
        logs_dir: Directory for log files
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator(logs_dir)
    config = load_config(config_path)
    validator.validate_benchmark_config(BenchmarkConfig(**config))
    return True

__all__ = [
    "BenchmarkConfig",
    "TaskConfig", 
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "ModelSettingsConfig",
    "SystemsConfig",
    "ConfigValidator",
    "load_config",
    "validate_config_file"
]
