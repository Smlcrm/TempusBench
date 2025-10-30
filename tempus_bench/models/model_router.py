"""
Model Router for handling model routing based on deterministic vs stochastic categorization.

This module provides routing for models based on their model_type in settings.yaml.
All models must handle both univariate and multivariate datasets internally.
"""

import os
import yaml

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from ..utils.logger import Logger
from ..utils.paths import get_models_dir


class ModelRouter:
    """
    Routes model requests to appropriate folders based on deterministic vs stochastic categorization.

    Handles three cases:
    1. deterministic models: Route to deterministic implementation (single point forecasts)
    2. stochastic models: Route to stochastic implementation (probabilistic forecasts with samples/quantiles)
    3. hybrid models: Route to hybrid implementation (both point forecasts and samples)

    All models must handle both univariate and multivariate datasets internally.
    """

    def __init__(self, logger: Logger):
        """Initialize the router and discover available models.

        Args:
            logger: Optional Logger instance to use for logging.
        """
        self.deterministic_models: List[str] = []
        self.stochastic_models: List[str] = []
        self.hybrid_models: List[str] = []
        self.logger = logger

        self._discover_models()

    def _discover_models(self):
        """Discover available models by examining the folder structure and reading settings.yaml."""
        models_dir = get_models_dir()

        # Iterate through all subdirectories in the models folder
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue

            # Skip special directories
            if item.name.startswith("__") or item.name in ["__pycache__"]:
                continue

            # Check if this directory contains a model file
            model_file = item / f"{item.name}_model.py"
            if not model_file.exists():
                continue

            # Read the settings.yaml to get model_type
            settings_file = item / "settings.yaml"
            if not settings_file.exists():
                self.logger.warning(
                    "ModelRouter",
                    f"Model '{item.name}' is missing settings.yaml file. Skipping.",
                )
                continue

            try:
                with open(settings_file, "r") as f:
                    settings = yaml.safe_load(f)

                model_type = settings.get("model_type")

                if model_type is None:
                    self.logger.warning(
                        "ModelRouter",
                        f"Model '{item.name}' is missing 'model_type' in settings.yaml. Skipping.",
                    )
                    continue

                # Add model to appropriate list based on model_type
                if model_type == "deterministic":
                    if item.name not in self.deterministic_models:
                        self.deterministic_models.append(item.name)
                    else:
                        raise ValueError(
                            f"Model '{item.name}' is duplicated in the deterministic models"
                        )
                elif model_type == "stochastic":
                    if item.name not in self.stochastic_models:
                        self.stochastic_models.append(item.name)
                    else:
                        raise ValueError(
                            f"Model '{item.name}' is duplicated in the stochastic models"
                        )
                elif model_type == "hybrid":
                    if item.name not in self.hybrid_models:
                        self.hybrid_models.append(item.name)
                    else:
                        raise ValueError(
                            f"Model '{item.name}' is duplicated in the hybrid models"
                        )
                else:
                    self.logger.warning(
                        "ModelRouter",
                        f"Model '{item.name}' has invalid model_type '{model_type}' in settings.yaml. "
                        f"Expected 'deterministic', 'stochastic', or 'hybrid'. Skipping.",
                    )

            except yaml.YAMLError as e:
                self.logger.error(
                    "ModelRouter",
                    f"Error parsing settings.yaml for model '{item.name}': {e}. Skipping.",
                )
                continue
            except Exception as e:
                self.logger.error(
                    "ModelRouter",
                    f"Unexpected error while processing model '{item.name}': {e}. Skipping.",
                )
                continue

        # Check for overlap between model lists after discovery
        overlaps = [
            (
                "deterministic & stochastic",
                set(self.deterministic_models) & set(self.stochastic_models),
            ),
            (
                "deterministic & hybrid",
                set(self.deterministic_models) & set(self.hybrid_models),
            ),
            (
                "stochastic & hybrid",
                set(self.stochastic_models) & set(self.hybrid_models),
            ),
        ]
        msg = [f"{label}: {sorted(list(items))}" for label, items in overlaps if items]
        if msg:
            raise ValueError(
                "Model(s) appear in multiple categories:\n  " + "\n  ".join(msg)
            )

    def _generate_class_name(self, model_name: str) -> str:
        """
        Generate the class name for a model using naming convention.

        Args:
            model_name: Name of the model

        Returns:
            The class name following the convention: {ModelName}Model
        """
        # Convert model_name to PascalCase and append "Model"
        # e.g., "arima" -> "ArimaModel", "moirai_moe" -> "MoiraiMoeModel"
        return "".join(word.capitalize() for word in model_name.split("_")) + "Model"

    def get_available_models(self) -> Dict[str, list]:
        """
        Get all available models and their supported variants.

        Returns:
            Dictionary mapping model categories to lists of models
        """
        return {
            "deterministic": sorted(self.deterministic_models),
            "stochastic": sorted(self.stochastic_models),
            "hybrid": sorted(self.hybrid_models),
        }

    def get_model_path_by_model_type(
        self, model_name: str, model_type: str
    ) -> Tuple[str, str, str]:
        """
        Get the model path, file name, and class name based on model name and model type.

        Args:
            model_name: Name of the model
            model_type: Type of model - "deterministic", "stochastic", or "hybrid"

        Returns:
            Tuple of (folder_path, file_name, class_name)

        Note:
            All models handle both univariate and multivariate datasets internally.
            Routing rules:
            - Stochastic model type + Stochastic model → Use stochastic model
            - Deterministic model type + Deterministic model → Use deterministic model
            - Deterministic model type + Stochastic model → Use stochastic model (extract point forecast)
            - Stochastic model type + Deterministic model → ERROR (can't generate samples)
        """
        # Get the absolute path to the models directory
        models_dir = get_models_dir()

        # Determine model type
        is_deterministic_model = model_name in self.deterministic_models
        is_stochastic_model = model_name in self.stochastic_models
        is_hybrid_model = model_name in self.hybrid_models

        if model_type == "deterministic" and is_deterministic_model:
            # Use deterministic model for deterministic task
            folder_path = str(models_dir / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name

        elif model_type == "deterministic" and is_stochastic_model:
            # Use stochastic model but extract point forecast (mean/median)
            folder_path = str(models_dir / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name

        elif model_type == "stochastic" and is_stochastic_model:
            # Use stochastic model for stochastic task
            folder_path = str(models_dir / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name

        elif model_type == "hybrid" and is_hybrid_model:
            # Use hybrid model for hybrid task
            folder_path = str(models_dir / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name

        elif model_type == "stochastic" and is_deterministic_model:
            # Cannot generate samples from deterministic model
            raise ValueError(
                f"Cannot use deterministic model '{model_name}' for stochastic model type. "
                f"Stochastic model types require models that can generate forecast samples."
            )

        else:
            # Model not found in any category
            available = self.get_available_models()
            raise ValueError(
                f"Model '{model_name}' not found. Available models:\n"
                f"  Deterministic: {available['deterministic']}\n"
                f"  Stochastic: {available['stochastic']}\n"
                f"  Hybrid: {available['hybrid']}"
            )
