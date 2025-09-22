"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import random
from .data_types import Dataset, DatasetSplit, PreprocessedData


class MinMaxScalerSafe:
    def __init__(self, feature_range=(1e-4, 1.0)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X_scaled):
        return self.scaler.inverse_transform(X_scaled)


class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary. Preprocessing-related keys are under 'dataset'.
        """
        self.config = config
        self.scalers = {}
        self.preprocessing_config = self._extract_preprocessing_config()

    def _extract_preprocessing_config(self) -> Dict[str, Any]:
        """
        Extract preprocessing options from the dataset section.
        """
        dataset_cfg = self.config.get("dataset", {})
        return {
            "normalize": dataset_cfg.get("normalize", True),
            "normalization_method": dataset_cfg.get("normalization_method", "minmax"),
            "handle_missing": dataset_cfg.get("handle_missing", "interpolate"),
            "remove_outliers": dataset_cfg.get("remove_outliers", False),
            "outlier_threshold": dataset_cfg.get("outlier_threshold", 3)
        }

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        """
        strategy = self.preprocessing_config.get('handle_missing', 'interpolate')

        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif strategy == 'interpolate':
            # Fill NaNs over time per target (rows are time, columns are targets)
            return df.interpolate(axis=0).ffill(axis=0).bfill(axis=0)
        elif strategy == 'forward_fill':
            return df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            return df.fillna(method='bfill')
        else:
            return df.fillna(method='ffill').fillna(method='bfill')

    def _normalize_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using standard or min-max scaling.
        """
        if not self.preprocessing_config.get('normalize', True):
            return df

        # Handle empty DataFrames
        if df.empty:
            return df

        method = self.preprocessing_config.get('normalization_method', 'standard')
        df_normalized = df.copy()

        numerical_cols = df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            # Skip columns with no valid data
            if df[col].isna().all() or len(df[col].dropna()) == 0:
                continue
                
            # Handle columns with some NaN values by filling them before scaling
            col_data = df[col].fillna(df[col].mean())  # Fill NaN with mean for scaling
            
            if is_training:
                if method == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScalerSafe()
                df_normalized[col] = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler
            else:
                scaler = self.scalers.get(col)
                if scaler:
                    df_normalized[col] = scaler.transform(col_data.values.reshape(-1, 1)).flatten()

        return df_normalized

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using the z-score method.
        """
        if not self.preprocessing_config.get('remove_outliers', False):
            return df

        threshold = self.preprocessing_config.get('outlier_threshold', 3)
        df_clean = df.copy()

        for col in df.select_dtypes(include=[np.number]).columns:
            # Only calculate z-scores on non-null values
            non_null_mask = df[col].notna()
            if non_null_mask.sum() > 0:  # Only process if we have non-null values
                non_null_data = df[col][non_null_mask]
                z_scores = np.abs((non_null_data - non_null_data.mean()) / non_null_data.std())
                outlier_mask = z_scores >= threshold
                
                # Replace outliers with NaN (they will be handled by interpolation later)
                df_clean.loc[non_null_mask, col] = non_null_data.mask(outlier_mask)

        return df_clean

    def _generate_artificial_timestamps(self, split: DatasetSplit, metadata: Dict[str, Any]) -> DatasetSplit:
        """Generate artificial timestamps for unknown/irregular frequency data."""
        freq = str(metadata.get("freq", "unknown")).lower()
        
        # Check if we need artificial timestamps
        if freq not in ["unknown", "irregular", "none", ""]:
            return split
            
        # Get data length
        data_length = len(split.targets) if hasattr(split.targets, '__len__') else split.targets.shape[0]
        
        # Random frequency >= 1 minute
        frequencies = ["1min", "5min", "15min", "1h", "3h", "1D", "1W"]
        freq = random.choice(frequencies)
        
        # Generate timestamps from current time
        timestamps = pd.date_range(start=datetime.now(), periods=data_length, freq=freq)
        
        return DatasetSplit(
            targets=split.targets,
            features=split.features, 
            metadata=split.metadata,
            timestamps=timestamps.values
        )

    def preprocess(self, data: Dataset) -> PreprocessedData:
        """
        Apply missing value handling, outlier removal, normalization, and timestamp generation to all splits.

        Args:
            data: Dataset with train/validation/test splits

        Returns:
            PreprocessedData: Transformed dataset and preprocessing metadata
        """
        def process_split(split: DatasetSplit, is_training: bool) -> DatasetSplit:
            # Generate artificial timestamps if needed
            processed_split = self._generate_artificial_timestamps(split, data.metadata or {})
            
            # Process targets
            targets_df = processed_split.targets.copy() if isinstance(processed_split.targets, pd.DataFrame) else pd.DataFrame(processed_split.targets)
            targets_df = self._handle_missing_values(targets_df)
            if is_training:
                targets_df = self._remove_outliers(targets_df)
            targets_df = self._normalize_features(targets_df, is_training=is_training)
            
            # Process features (exogenous variables) if present
            features_df = None
            if processed_split.features is not None:
                features_df = processed_split.features.copy() if isinstance(processed_split.features, pd.DataFrame) else pd.DataFrame(processed_split.features)
                features_df = self._handle_missing_values(features_df)
                if is_training:
                    features_df = self._remove_outliers(features_df)
                features_df = self._normalize_features(features_df, is_training=is_training)
            
            return DatasetSplit(
                targets=targets_df, 
                features=features_df, 
                metadata=processed_split.metadata, 
                timestamps=processed_split.timestamps
            )

        train_split = process_split(data.train, is_training=True)
        val_split = process_split(data.validation, is_training=False)
        test_split = process_split(data.test, is_training=False)

        preprocessed_dataset = Dataset(
            train=train_split,
            validation=val_split,
            test=test_split,
            name=data.name,
            metadata=data.metadata
        )

        preprocessing_info = {
            'normalize': self.preprocessing_config.get('normalize', True),
            'normalization_method': self.preprocessing_config.get('normalization_method', 'standard'),
            'remove_outliers': self.preprocessing_config.get('remove_outliers', False),
            'outlier_threshold': self.preprocessing_config.get('outlier_threshold', 3),
            'handle_missing': self.preprocessing_config.get('handle_missing', 'interpolate'),
            'scalers': {
                col: {
                    'type': type(scaler).__name__,
                    'mean': getattr(scaler, 'mean_', None),
                    'scale': getattr(scaler, 'scale_', None)
                } for col, scaler in self.scalers.items()
            },
            'target_columns': list(train_split.targets.columns) if hasattr(train_split.targets, 'columns') else None,
            'feature_columns': list(train_split.features.columns) if train_split.features is not None and hasattr(train_split.features, 'columns') else None
        }

        return PreprocessedData(data=preprocessed_dataset, preprocessing_info=preprocessing_info)
