"""
Configuration module for the benchmarking pipeline.

This module provides configuration models, validation, and management for the
benchmarking pipeline. All configuration-related functionality is centralized here.
"""

from .configs import (
    TaskConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelHParams,
    EvaluationSettings,
    JobConfig,
    convert_pydantic_errors,
    load_and_validate_task_configs,
    validate_model_settings_dict,
    validate_task_config_with_context,
)
from .manager import Manager, ValidationError


__all__ = [
    # Configuration models
    "TaskConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelHParams",
    "EvaluationSettings",
    "JobConfig",
    # Manager and exceptions
    "Manager",
    "ValidationError",
    # Validation utility functions
    "convert_pydantic_errors",
    "load_and_validate_task_configs",
    "validate_model_settings_dict",
    "validate_task_config_with_context",
]
