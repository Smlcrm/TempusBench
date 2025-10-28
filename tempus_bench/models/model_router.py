"""
Model Router for handling model routing based on deterministic vs stochastic categorization.

This module provides routing for models based on their location in deterministic/ or stochastic/ folders.
All models must handle both univariate and multivariate datasets internally.
"""

import os
from typing import Tuple, Dict, Any, Set
from pathlib import Path
from ..utils.paths import get_models_dir
from ..utils.logger import get_logger

class ModelRouter:
    """
    Routes model requests to appropriate folders based on deterministic vs stochastic categorization.

    Handles two cases:
    1. deterministic models: Route to deterministic implementation (single point forecasts)
    2. stochastic models: Route to stochastic implementation (probabilistic forecasts with samples/quantiles)

    All models must handle both univariate and multivariate datasets internally.
    """

    def __init__(self):
        """Initialize the router and discover available models."""
        self.deterministic_models: Set[str] = set()
        self.stochastic_models: Set[str] = set()
        self.logger = get_logger()

        self._discover_models()
        self._validate_models()

    def _discover_models(self):
        """Discover available models by examining the folder structure."""
        models_dir = get_models_dir()

        # Discover deterministic models
        deterministic_dir = models_dir / "deterministic"
        if deterministic_dir.exists():
            for item in deterministic_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}_model.py").exists():
                    self.deterministic_models.add(item.name)
        else:
            raise SystemError("The models/deterministic folder is missing")

        # Discover stochastic models
        stochastic_dir = models_dir / "stochastic"
        if stochastic_dir.exists():
            for item in stochastic_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}_model.py").exists():
                    self.stochastic_models.add(item.name)
        else:
            raise SystemError("The models/stochastic folder is missing")

    def _validate_models(self):
        """Validate that all models are properly categorized and raise errors for inconsistencies."""
        # Check for models that appear in multiple categories
        overlap = self.deterministic_models & self.stochastic_models
        if overlap:
            raise ValueError(
                f"Models cannot be both deterministic and stochastic: {overlap}"
            )

        # Check for models that are not in any category
        all_models = self.deterministic_models | self.stochastic_models
        models_dir = get_models_dir()

        # Find all model folders
        all_model_folders = set()
        for category_dir in [
            models_dir / "deterministic",
            models_dir / "stochastic",
        ]:
            if category_dir.exists():
                for item in category_dir.iterdir():
                    if item.is_dir() and (item / f"{item.name}_model.py").exists():
                        all_model_folders.add(item.name)

        uncategorized_models = all_model_folders - all_models
        if uncategorized_models:
            raise ValueError(
                f"Found models that are not properly categorized: {uncategorized_models}. "
                f"Each model must be in exactly one of: deterministic or stochastic folders."
            )

        # Check for models that are in categories but don't have proper structure
        for model_name in all_models:
            if model_name in self.deterministic_models:
                model_path = models_dir / "deterministic" / model_name / f"{model_name}_model.py"
                if not model_path.exists():
                    raise ValueError(
                        f"Deterministic model '{model_name}' missing required file: {model_path}"
                    )

            elif model_name in self.stochastic_models:
                model_path = models_dir / "stochastic" / model_name / f"{model_name}_model.py"
                if not model_path.exists():
                    raise ValueError(
                        f"Stochastic model '{model_name}' missing required file: {model_path}"
                    )

    def _generate_class_name(self, model_name: str, category: str) -> str:
        """
        Generate the class name for a model using naming convention.

        Args:
            model_name: Name of the model
            category: The category to use ('deterministic' or 'stochastic')

        Returns:
            The class name following the convention: {ModelName}Model
        """
        # Convert model_name to PascalCase and append "Model"
        # e.g., "arima" -> "ArimaModel", "moirai_moe" -> "MoiraiMoeModel"
        return ''.join(word.capitalize() for word in model_name.split('_')) + 'Model'

    def get_available_models(self) -> Dict[str, list]:
        """
        Get all available models and their supported variants.

        Returns:
            Dictionary mapping model categories to lists of models
        """
        return {
            "deterministic": sorted(self.deterministic_models),
            "stochastic": sorted(self.stochastic_models),
        }

    def get_model_path_by_task_type(
        self, model_name: str, task_type: str
    ) -> Tuple[str, str, str]:
        """
        Get the model path, file name, and class name based on model name and task type.

        Args:
            model_name: Name of the model
            task_type: Type of task - "deterministic" or "stochastic"

        Returns:
            Tuple of (folder_path, file_name, class_name)

        Note:
            All models handle both univariate and multivariate datasets internally.
            Routing rules:
            - Stochastic task + Stochastic model → Use stochastic model
            - Deterministic task + Deterministic model → Use deterministic model
            - Deterministic task + Stochastic model → Use stochastic model (extract point forecast)
            - Stochastic task + Deterministic model → ERROR (can't generate samples)
        """
        # Get the absolute path to the models directory
        models_dir = get_models_dir()

        # Validate inputs
        if task_type not in ["deterministic", "stochastic"]:
            raise ValueError(f"task_type must be 'deterministic' or 'stochastic', got '{task_type}'")

        # Check if model exists
        if model_name not in (self.deterministic_models | self.stochastic_models):
            available_models = sorted(
                self.deterministic_models | self.stochastic_models
            )
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )

        # Determine model type
        is_deterministic_model = model_name in self.deterministic_models
        is_stochastic_model = model_name in self.stochastic_models

        # Apply routing rules
        if task_type == "stochastic" and is_deterministic_model:
            raise ValueError(
                f"Model '{model_name}' is deterministic but task requires stochastic forecasting. "
                f"Deterministic models cannot generate samples/quantiles. "
                f"Use a stochastic model instead."
            )

        elif task_type == "deterministic" and is_stochastic_model:
            # Use stochastic model but extract point forecast (mean/median)
            folder_path = str(models_dir / "stochastic" / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name, "stochastic")
            return folder_path, file_name, class_name

        elif task_type == "stochastic" and is_stochastic_model:
            # Use stochastic model for stochastic task
            folder_path = str(models_dir / "stochastic" / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name, "stochastic")
            return folder_path, file_name, class_name

        elif task_type == "deterministic" and is_deterministic_model:
            # Use deterministic model for deterministic task
            folder_path = str(models_dir / "deterministic" / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name, "deterministic")
            return folder_path, file_name, class_name
