"""
Configuration manager for the benchmarking pipeline.

This module provides essential configuration loading and validation functionality.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .models import BenchmarkConfig
from .validator import ConfigValidator


class ConfigManager:
    """Configuration manager for loading and validating configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.validator = ConfigValidator()

    def load_from_file(self, config_path: Union[str, Path]) -> BenchmarkConfig:
        """
        Load configuration from a file, always merging with config/default.yaml

        Args:
            config_path: Path to the configuration file

        Returns:
            Validated BenchmarkConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML file
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            user_config_data = self._load_yaml(config_path)
        else:
            raise ValueError(f"Only YAML configuration files are supported. Found: {config_path.suffix}")

        # Always merge with defaults
        config_data = self._merge_with_defaults(user_config_data)

        # Validate and return
        return self.validator.validate_config(config_data)

    def validate_config_file(self, config_path: str) -> bool:
        """
        Validate a configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            True if validation passes

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load and validate
        self.load_from_file(config_path)
        return True

    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ValueError("Configuration file is empty or invalid YAML")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

    def _merge_with_defaults(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with default configuration.

        Args:
            user_config: User-provided configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        # Load default configuration
        default_config_path = self.config_dir / "default.yaml"

        if not default_config_path.exists():
            # If no default.yaml, return user config as-is
            return user_config

        try:
            default_config = self._load_yaml(default_config_path)
        except Exception:
            # If we can't load defaults, return user config as-is
            return user_config

        # Deep merge user config over defaults
        merged_config = self._deep_merge(default_config, user_config)

        return merged_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
