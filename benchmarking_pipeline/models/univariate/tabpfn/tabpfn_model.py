import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import os

from tabpfn import TabPFNRegressor 
from benchmarking_pipeline.models.foundation_model import FoundationModel
import torch


def make_time_features(n: int) -> pd.DataFrame:
    """
    Produce basic cyclic time features for positions 0..n-1.
    Mirrors TabPFN-TS style feature engineering for univariate forecasting.
    """
    t = np.arange(n)
    features = {
        "t": t,
        "sin_1": np.sin(2 * np.pi * t / max(1, n)),
        "cos_1": np.cos(2 * np.pi * t / max(1, n)),
        "sin_2": np.sin(4 * np.pi * t / max(1, n)),
        "cos_2": np.cos(4 * np.pi * t / max(1, n)),
    }
    return pd.DataFrame(features)


class TabpfnModel(FoundationModel):

    def __init__(
        self, config: Dict[str, Any] = None, config_file: str = None
    ):
        """
        Initializes a TabPFN-TS forecaster

        Args:
            n_ensemble_configs (int): Number of ensemble configurations (kept in signature).
            device (str): 'cpu' or 'cuda' for the underlying TabPFN model.
            allow_large_cpu_dataset (bool): If True, bypasses the default CPU sample limit by
                setting ignore_pretraining_limits=True. Otherwise will error if >1000 samples.
        """
        super().__init__(config, config_file)
        
        # Extract model-specific config
        model_config = self._extract_model_config(self.config)
        
        if 'allow_large_cpu_dataset' not in model_config:
            raise ValueError("allow_large_cpu_dataset must be specified in config")
        if 'max_sequence_length' not in model_config:
            raise ValueError("max_sequence_length must be specified in config")
        
        self.allow_large_cpu_dataset = model_config['allow_large_cpu_dataset']
        self.max_sequence_length = model_config['max_sequence_length']
        
        # Set device - default to CPU for TabPFN
        self.device = model_config.get('device', 'cpu')
        
        self.model = None
        self.is_fitted = False

    def _build_tabular(self, y_history: np.ndarray, X_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given history of target and exogenous features, build tabular training matrix and target vector.
        """
        n = len(y_history)
        
        # Handle edge cases for TabPFN
        if n == 0:
            raise ValueError("y_history cannot be empty")
        
        # Ensure y_history has reasonable values for TabPFN
        y_history = np.array(y_history, dtype=np.float64)
        
        # Check for very small or zero values that could cause TabPFN issues
        if np.any(np.abs(y_history) < 1e-10):
            # Add small noise to prevent numerical issues
            y_history = y_history + np.random.normal(0, 1e-8, y_history.shape)
        
        # Normalize y_history to prevent TabPFN numerical issues
        y_mean = np.mean(y_history)
        y_std = np.std(y_history)
        if y_std < 1e-10:
            y_std = 1.0  # Prevent division by zero
        
        y_normalized = (y_history - y_mean) / y_std
        
        time_feats = make_time_features(n).reset_index(drop=True)
        if X_history is None or X_history.size == 0:
            df = time_feats
        else:
            exog_df = pd.DataFrame(X_history, index=range(n)).reset_index(drop=True)
            df = pd.concat([time_feats, exog_df], axis=1)
        X = df.to_numpy()
        
        # Store normalization parameters for later use
        self._y_mean = y_mean
        self._y_std = y_std
        
        return X, y_normalized
    
    def set_params(self, **params: Dict[str, Any]) -> 'TabpfnModel':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def train(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        y_context_timestamps = None,
        y_target_timestamps = None,
        **kwargs
    ) -> 'TabpfnModel':
        # Zero-shot TabPFN uses context during predict; mark as fitted
        self.is_fitted = True
        return self

    def get_params(self) -> Dict[str, Any]:
        return {
            'allow_large_cpu_dataset': self.allow_large_cpu_dataset,
            'max_sequence_length': self.max_sequence_length,
            'device': self.device,
        }

    def predict(self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_context_timestamps = None,
        y_target_timestamps = None,
        **kwargs):
        #print("HUH")
        #print(y_target)
        #print("YUHUHU?")
        #print(y_target_timestamps[0].strftime('%Y-%m-%d %X'))
        #raise Exception("UNgas")
        #timestamp_strings = [ts.strftime('%Y-%m-%d %X') for ts in y_target_timestamps]
        
        # Construct DataFrame - handle case where y_context might already be a DataFrame
        if isinstance(y_context, pd.DataFrame):
            # Extract the actual values from the DataFrame
            y_context_values = y_context.iloc[:, 0].values  # Get first column values
            # Use the original column name if available
            column_name = y_context.columns[0] if len(y_context.columns) > 0 else 'target'
        else:
            # y_context is already a numpy array or series
            y_context_values = y_context
            column_name = 'target'
            
        df = pd.DataFrame(y_context_values, index=y_context_timestamps, columns=[column_name])
        
        # Fail-fast: y_target is required to determine prediction length
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")
        
        # Determine prediction length from y_target dimensions
        if hasattr(y_target, 'shape'):
            if y_target.ndim == 1:
                prediction_length = len(y_target)
            elif y_target.ndim == 2:
                prediction_length = y_target.shape[0]
            else:
                raise ValueError(f"y_target has unexpected dimensions: {y_target.shape}. Expected 1D or 2D array.")
        else:
            # Handle case where y_target might be a list or other iterable
            prediction_length = len(y_target)
        
        if prediction_length <= 0:
            raise ValueError(f"y_target has invalid length: {prediction_length}. Must be > 0.")
        
        self.ctx = len(df)
        results = self._sub_predict(df, column_name, prediction_length)
        return np.array(results)

    def _sub_predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        prediction_length: int
    ) -> List[float]:
        """
        Multi-step forecast for the target column.

        Args:
            df (pd.DataFrame): Historical time series with target and optional features.
            target_col (str): Column to forecast.
            prediction_length (int): Number of steps to predict.

        Returns:
            List[float]: Forecasted values.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        y_history = df[target_col].astype(float).values
        
        # Handle exogenous inputs - if only target column exists, X_full should be empty
        if len(df.columns) == 1:
            X_full = np.array([]).reshape(0, 0)  # Empty 2D array
        else:
            X_full = df.drop(columns=[target_col]).values  # exogenous inputs

        if len(y_history) > self.max_sequence_length:
            warnings.warn(
                f"Data has more than {self.max_sequence_length} points. "
                f"Using the last {self.max_sequence_length} points as context.",
                UserWarning
            )
            y_history = y_history[-self.max_sequence_length:]
            X_full = X_full[-self.max_sequence_length:, :]

        forecasts = []

        for _ in range(prediction_length):
            # Build training data from current history
            X_train, y_train = self._build_tabular(y_history, X_full)

            if self.device == "cpu" and not self.allow_large_cpu_dataset and X_train.shape[0] > 1000:
                raise RuntimeError(
                    "Running on CPU with more than 1000 samples is disallowed by default. "
                    "Either set allow_large_cpu_dataset=True when constructing the forecaster to "
                    "bypass this (which will set ignore_pretraining_limits), or use a GPU."
                )

            # Prepare next time-step features
            next_pos = len(y_history)
            next_time_feats = make_time_features(next_pos + 1).iloc[[-1]].reset_index(drop=True)
            if X_full is None or X_full.size == 0:
                query_df = next_time_feats
            else:
                last_exog = pd.DataFrame(X_full, columns=[f"feat_{i}" for i in range(X_full.shape[1])])
                last_exog_row = last_exog.iloc[[-1]].reset_index(drop=True)
                query_df = pd.concat([next_time_feats, last_exog_row], axis=1)

            X_query = query_df.to_numpy()

            # Instantiate and use a fresh TabPFNRegressor (zero-shot / in-context)
            # Pass ignore_pretraining_limits if allowed and on CPU
            if self.device == "cpu" and self.allow_large_cpu_dataset:
                model = TabPFNRegressor(device=self.device, ignore_pretraining_limits=True)
            else:
                model = TabPFNRegressor(device=self.device)

            model.fit(X_train, y_train)  # provide context
            pred = model.predict(X_query)  # only test features passed here

            next_val = float(np.asarray(pred).ravel()[0])
            forecasts.append(next_val)

            # Denormalize the prediction back to original scale
            next_val_denorm = next_val * self._y_std + self._y_mean
            
            # Append the denormalized predicted value to history (autoregressive)
            y_history = np.concatenate([y_history, [next_val_denorm]])
            if X_full is not None and X_full.size != 0:
                X_full = np.vstack([X_full, X_full[-1]])

            # Truncate history if exceeding max length
            if len(y_history) > self.max_sequence_length:
                y_history = y_history[-self.max_sequence_length:]
                if X_full is not None and X_full.size != 0:
                    X_full = X_full[-self.max_sequence_length:, :]

        # Denormalize all forecasts back to original scale
        forecasts_denorm = [f * self._y_std + self._y_mean for f in forecasts]
        return forecasts_denorm