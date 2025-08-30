"""
Model Router for handling model routing based on inferred target count.

This module provides routing for models based on their location and whether the dataset
has one target (univariate) or multiple targets (multivariate).
All data is treated as multivariate where univariate is just num_targets == 1.

The router automatically discovers models from the folder structure and categorizes them into:
- anyvariate: Models that can handle both univariate and multivariate data
- multivariate: Models with separate implementations for univariate/multivariate
- univariate_only: Models that only support univariate data
"""

import os
from typing import Tuple, Dict, Any, Optional, Set
from pathlib import Path


class ModelRouter:
    """
    Routes model requests to appropriate folders based on inferred target count.
    
    Handles three cases:
    1. anyvariate models: Route to anyvariate implementation (handles both variants)
    2. models with separate implementations: Automatically detect variant based on target count:
       - num_targets == 1: Use univariate variant (fails if not available)
       - num_targets > 1: Use multivariate variant
    3. univariate-only models: Route to univariate implementation
    
    All data is treated as multivariate where univariate is just num_targets == 1.
    """
    
    def __init__(self):
        """Initialize the router and discover available models."""
        self.anyvariate_models: Set[str] = set()
        self.multivariate_models: Set[str] = set()
        self.univariate_models: Set[str] = set()
        
        self._discover_models()
        self._validate_model_categorization()
    
    def _discover_models(self):
        """Discover available models by examining the folder structure."""
        models_dir = Path(__file__).parent
        
        # Discover anyvariate models
        anyvariate_dir = models_dir / "anyvariate"
        if anyvariate_dir.exists():
            for item in anyvariate_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}_model.py").exists():
                    self.anyvariate_models.add(item.name)
        
        # Discover models that have separate multivariate implementations
        multivariate_dir = models_dir / "multivariate"
        if multivariate_dir.exists():
            for item in multivariate_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}_model.py").exists():
                    self.multivariate_models.add(item.name)
        
        # Discover univariate-only models (models that only exist in univariate folder)
        univariate_dir = models_dir / "univariate"
        if univariate_dir.exists():
            for item in univariate_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}_model.py").exists():
                    # Only add to univariate_models if it's not already in multivariate_models
                    if item.name not in self.multivariate_models:
                        self.univariate_models.add(item.name)
    
    def _validate_model_categorization(self):
        """Validate that all models are properly categorized and raise errors for inconsistencies."""
        # Check for models that appear in multiple categories
        anyvariate_multivariate_overlap = self.anyvariate_models & self.multivariate_models
        anyvariate_univariate_overlap = self.anyvariate_models & self.univariate_models
        
        if anyvariate_multivariate_overlap:
            raise ValueError(f"Models cannot be both anyvariate and multivariate: {anyvariate_multivariate_overlap}")
        
        if anyvariate_univariate_overlap:
            raise ValueError(f"Models cannot be both anyvariate and univariate: {anyvariate_univariate_overlap}")
        
        # Note: multivariate_models and univariate_models can overlap because:
        # - multivariate_models = models that have separate multivariate implementations
        # - univariate_models = models that only exist in univariate folder
        # So a model can be in both if it has both implementations
        
        # Check for models that are not in any category
        all_models = self.anyvariate_models | self.multivariate_models | self.univariate_models
        models_dir = Path(__file__).parent
        
        # Find all model folders
        all_model_folders = set()
        for category_dir in [models_dir / "anyvariate", models_dir / "multivariate", models_dir / "univariate"]:
            if category_dir.exists():
                for item in category_dir.iterdir():
                    if item.is_dir() and (item / f"{item.name}_model.py").exists():
                        all_model_folders.add(item.name)
        
        uncategorized_models = all_model_folders - all_models
        if uncategorized_models:
            raise ValueError(f"Found models that are not properly categorized: {uncategorized_models}. "
                           f"Each model must be in exactly one of: anyvariate, multivariate, or univariate folders.")
        
        # Check for models that are in categories but don't have proper structure
        for model_name in all_models:
            if model_name in self.anyvariate_models:
                model_path = models_dir / "anyvariate" / model_name / f"{model_name}_model.py"
                if not model_path.exists():
                    raise ValueError(f"Anyvariate model '{model_name}' missing required file: {model_path}")
            elif model_name in self.multivariate_models:
                # Check both multivariate and univariate implementations
                multivariate_path = models_dir / "multivariate" / model_name / f"{model_name}_model.py"
                univariate_path = models_dir / "univariate" / model_name / f"{model_name}_model.py"
                
                if not multivariate_path.exists():
                    raise ValueError(f"Multivariate model '{model_name}' missing required file: {multivariate_path}")
                if not univariate_path.exists():
                    raise ValueError(f"Multivariate model '{model_name}' missing univariate implementation: {univariate_path}")
            elif model_name in self.univariate_models:
                model_path = models_dir / "univariate" / model_name / f"{model_name}_model.py"
                if not model_path.exists():
                    raise ValueError(f"Univariate model '{model_name}' missing required file: {model_path}")
    
    def parse_model_spec(self, model_spec: str) -> str:
        """
        Parse model specification to extract the model name.
        
        Args:
            model_spec: Model specification string (e.g., 'arima', 'chronos', 'prophet')
            
        Returns:
            Model name string
            
        Examples:
            >>> router = ModelRouter()
            >>> router.parse_model_spec('arima')
            'arima'
            >>> router.parse_model_spec('chronos')
            'chronos'
        """
        return model_spec.strip()
    
    def get_model_path_by_target_count(self, model_name: str, num_targets: int) -> Tuple[str, str, str]:
        """
        Get the model path, file name, and class name based on target count.
        
        Args:
            model_name: Name of the model
            num_targets: Number of targets (1 for univariate, >1 for multivariate)
            
        Returns:
            Tuple of (folder_path, file_name, class_name)
            
        Note:
            All data is treated as multivariate where univariate is just num_targets == 1.
            The routing logic automatically detects variant:
            - num_targets == 1: univariate variant (fails if not available)
            - num_targets > 1: multivariate variant
        """
        # Get the absolute path to the models directory
        models_dir = Path(__file__).parent
        
        # Validate num_targets
        if num_targets < 1:
            raise ValueError(f"num_targets must be >= 1, got {num_targets}")
        
        # Handle anyvariate models (can handle both univariate and multivariate)
        if model_name in self.anyvariate_models:
            folder_path = str(models_dir / "anyvariate" / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name, 'anyvariate')
            return folder_path, file_name, class_name
        
        # Handle models with separate multivariate implementations
        elif model_name in self.multivariate_models:
            # Automatically detect variant based on num_targets
            if num_targets == 1:
                # Check if univariate implementation exists
                univariate_path = models_dir / "univariate" / model_name
                if univariate_path.exists() and (univariate_path / f"{model_name}_model.py").exists():
                    folder_path = str(models_dir / "univariate" / model_name)
                    class_name = self._generate_class_name(model_name, 'univariate')
                    print(f"[ROUTER] Routing {model_name} with {num_targets} target(s) -> univariate variant")
                else:
                    # Fail fast: no fallback to multivariate
                    raise ValueError(f"Model '{model_name}' requested for univariate data (num_targets=1) but univariate implementation not found at {univariate_path}")
            else:
                # Use multivariate for actual multivariate data
                folder_path = str(models_dir / "multivariate" / model_name)
                class_name = self._generate_class_name(model_name, 'multivariate')
                print(f"[ROUTER] Routing {model_name} with {num_targets} target(s) -> multivariate variant")
            
            file_name = f"{model_name}_model"
            return folder_path, file_name, class_name
        
        # Handle univariate-only models
        elif model_name in self.univariate_models:
            # Univariate models only support univariate data
            if num_targets > 1:
                raise ValueError(f"Model '{model_name}' does not support multivariate variant. "
                               f"Available variants: univariate")
            
            folder_path = str(models_dir / "univariate" / model_name)
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name, 'univariate')
            return folder_path, file_name, class_name
        
        else:
            available_models = sorted(self.anyvariate_models | self.multivariate_models | self.univariate_models)
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    def _generate_class_name(self, model_name: str, variant: str) -> str:
        """
        Generate the class name for a model by reading the actual model file.
        
        Args:
            model_name: Name of the model
            variant: The variant to use ('univariate', 'multivariate', or 'anyvariate')
            
        Returns:
            The class name found in the model file
        """
        models_dir = Path(__file__).parent
        
        # Determine which file to read based on variant
        if variant == 'anyvariate' or model_name in self.anyvariate_models:
            model_file = models_dir / "anyvariate" / model_name / f"{model_name}_model.py"
        elif variant == 'multivariate':
            # Always read from multivariate folder when variant is explicitly multivariate
            model_file = models_dir / "multivariate" / model_name / f"{model_name}_model.py"
        elif variant == 'univariate':
            # Always read from univariate folder when variant is explicitly univariate
            model_file = models_dir / "univariate" / model_name / f"{model_name}_model.py"
        elif model_name in self.multivariate_models:
            # For models in multivariate_models set, use the specified variant
            model_file = models_dir / variant / model_name / f"{model_name}_model.py"
        elif model_name in self.univariate_models:
            # For models in univariate_models set, default to univariate
            model_file = models_dir / "univariate" / model_name / f"{model_name}_model.py"
        else:
            raise ValueError(f"Unknown model '{model_name}'")
        
        if not model_file.exists():
            raise ValueError(f"Model file not found: {model_file}")
        
        # Read the model file and extract the class name
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for class definitions
            import re
            class_pattern = r'class\s+(\w+)(?:\(|:)'
            class_matches = re.findall(class_pattern, content)
            
            if not class_matches:
                raise ValueError(f"No class definition found in {model_file}")
            
            # Find the main model class (usually the one ending with 'Model')
            model_classes = [cls for cls in class_matches if cls.endswith('Model')]
            
            if model_classes:
                return model_classes[0]  # Return the first Model class found
            else:
                # If no class ends with 'Model', return the first class found
                return class_matches[0]
                
        except Exception as e:
            raise ValueError(f"Failed to read class name from {model_file}: {e}")
    
    def validate_model_spec(self, model_spec: str) -> bool:
        """
        Validate that a model specification is valid.
        
        Args:
            model_spec: Model specification string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            model_name = self.parse_model_spec(model_spec)
            
            # Check if model exists in any category
            if model_name not in (self.anyvariate_models | self.multivariate_models | self.univariate_models):
                return False
            
            # Check if folder exists
            folder_path, _, _ = self.get_model_path_by_target_count(model_name, 1)
            return os.path.exists(folder_path)
            
        except (ValueError, Exception):
            return False
    
    def get_available_models(self) -> Dict[str, list]:
        """
        Get all available models and their supported variants.
        
        Returns:
            Dictionary mapping model categories to lists of models
        """
        return {
            'anyvariate': sorted(self.anyvariate_models),
            'multivariate': sorted(self.multivariate_models),
            'univariate_only': sorted(self.univariate_models)
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        # Get the absolute path to the models directory
        models_dir = Path(__file__).parent
        
        info = {
            'name': model_name,
            'category': None,
            'supported_variants': [],
            'folder_paths': {}
        }
        
        if model_name in self.anyvariate_models:
            info['category'] = 'anyvariate'
            info['supported_variants'] = ['univariate', 'multivariate']
            info['folder_paths'] = {
                'anyvariate': str(models_dir / "anyvariate" / model_name)
            }
        elif model_name in self.multivariate_models:
            info['category'] = 'multivariate'
            info['supported_variants'] = ['univariate', 'multivariate']
            info['folder_paths'] = {
                'univariate': str(models_dir / "univariate" / model_name),
                'multivariate': str(models_dir / "multivariate" / model_name)
            }
        elif model_name in self.univariate_models:
            info['category'] = 'univariate_only'
            info['supported_variants'] = ['univariate']
            info['folder_paths'] = {
                'univariate': str(models_dir / "univariate" / model_name)
            }

        return info
    
    def get_model_path_with_auto_detection(self, model_name: str, num_targets: int) -> Tuple[str, str, str]:
        """
        Backward-compatible wrapper for automatic variant detection based on inferred num_targets.

        Args:
            model_name: Name of the model
            num_targets: Number of target variables in the dataset

        Returns:
            Tuple of (folder_path, file_name, class_name)
        """
        return self.get_model_path_by_target_count(model_name, num_targets)


# Global router instance
model_router = ModelRouter()
