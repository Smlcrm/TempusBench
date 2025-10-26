"""
Configuration validator for the benchmarking pipeline.

This module provides comprehensive validation of configuration files using Pydantic
to ensure they comply with the expected schema before execution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from .models import BenchmarkConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ConfigValidator:
    """Configuration validator with comprehensive validation rules."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the configuration validator.
        
        Args:
            models_dir: Directory containing model implementations
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / "models"
    
    def validate_config(self, config_data: Dict[str, Any]) -> BenchmarkConfig:
        """
        Validate configuration data and return a BenchmarkConfig instance.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            Validated BenchmarkConfig instance
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Validate using Pydantic model
            config = BenchmarkConfig(**config_data)
            
            # Additional custom validations
            self._validate_model_availability(config)
            self._validate_paths_exist(config)
            
            logger.info("Configuration validation passed")
            return config
            
        except ValidationError as e:
            error_msg = self._convert_pydantic_errors(e)
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ConfigValidationError(error_msg)
    
    def validate_config_file(self, config_path: str) -> bool:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load and validate
        from .loader import ConfigLoader
        loader = ConfigLoader()
        loader.load_from_file(config_path)
        
        return True
    
    def _validate_model_availability(self, config: BenchmarkConfig) -> None:
        """Validate that all specified models are available."""
        if not self.models_dir.exists():
            raise ConfigValidationError(f"Models directory not found: {self.models_dir}")
        
        # Get all available models
        available_models = self._get_available_models()
        
        # Check each model in the configuration
        model_dict = config.model.model_dump()
        for model_name, model_params in model_dict.items():
            if model_params is not None:
                if model_name not in available_models:
                    raise ConfigValidationError(
                        f"Model '{model_name}' is not available. "
                        f"Available models: {sorted(available_models)}"
                    )
    
    def _validate_paths_exist(self, config: BenchmarkConfig) -> None:
        """Validate that required paths exist."""
        paths = config.paths
        
        # Check datasets directory
        datasets_path = Path(paths.datasets_dir)
        if not datasets_path.exists():
            logger.warning(f"Datasets directory not found: {datasets_path}")
        
        # Check models directory
        models_path = Path(paths.models_dir)
        if not models_path.exists():
            raise ConfigValidationError(f"Models directory not found: {models_path}")
        
        # Check configs directory
        configs_path = Path(paths.configs_dir)
        if not configs_path.exists():
            logger.warning(f"Configs directory not found: {configs_path}")
    
    def _get_available_models(self) -> set:
        """Get all available model names from the models directory."""
        available_models = set()
        
        if not self.models_dir.exists():
            return available_models
        
        # Look for model folders in any subdirectory
        for subdir in self.models_dir.iterdir():
            if subdir.is_dir():
                for model_folder in subdir.iterdir():
                    if model_folder.is_dir():
                        # Check if it has a model file
                        model_file = model_folder / f"{model_folder.name}_model.py"
                        if model_file.exists():
                            available_models.add(model_folder.name)
        
        return available_models
    
    def _convert_pydantic_errors(self, validation_error: ValidationError) -> str:
        """Convert Pydantic validation errors to a readable string."""
        error_messages = []
        for error in validation_error.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_messages.append(f"{field_path}: {error['msg']}")
        return "; ".join(error_messages)
    
    def get_validation_summary(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Get a summary of configuration validation results.
        
        Args:
            config: Validated configuration
            
        Returns:
            Dictionary with validation summary
        """
        return {
            "task_type": config.task.task_type,
            "forecast_horizon": config.task.forecast_horizon,
            "context_window": config.task.context_window,
            "max_windows": config.evaluation.max_windows,
            "num_models": len([m for m in config.model.model_dump().values() if m is not None]),
            "evaluation_metrics": config.evaluation.metrics,
            "tuning_loss": config.evaluation.tuning_loss,
            "datasets_pattern": config.task.dataset.name,
            "normalize_data": config.task.dataset.normalize,
            "handle_missing": config.task.dataset.handle_missing,
            "logging_enabled": config.logging.console_logging,
            "available_models": sorted(self._get_available_models())
        }
