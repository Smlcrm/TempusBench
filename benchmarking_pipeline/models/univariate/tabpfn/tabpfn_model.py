import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import os
import math
from pathlib import Path
from tabpfn import TabPFNRegressor
from benchmarking_pipeline.models.base_model import BaseModel
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


class TabpfnModel(BaseModel):

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes a TabPFN-TS forecaster

        Args:
            n_ensemble_configs (int): Number of ensemble configurations (kept in signature).
            device (str): 'cpu' or 'cuda' for the underlying TabPFN model.
            allow_large_cpu_dataset (bool): If True, bypasses the default CPU sample limit by
                setting ignore_pretraining_limits=True. Otherwise will error if >1000 samples.
        """
        super().__init__(config)

        # Store the full config for creating variate models
        self.full_config = config

        # Detect if this is anyvariate mode based on config filename
        self.is_anyvariate = False
        if hasattr(self, 'model_config') and 'original_config_path' in config:
            config_filename = Path(config['original_config_path']).name.lower()
            if "univariate" in config_filename:
                self.is_anyvariate = True
                print(f"[TABPFN] Anyvariate mode enabled for config: {config_filename}")

        # Set training loss to one of the configured metrics if mae is not available
        if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'metrics_to_calculate'):
            if 'mae' not in self.evaluator.metrics_to_calculate:
                # Use the first available metric for training loss optimization
                available_metrics = self.evaluator.metrics_to_calculate
                if 'rmse' in available_metrics:
                    self.training_loss = 'rmse'
                elif 'mse' in available_metrics:
                    self.training_loss = 'mse'
                elif len(available_metrics) > 0:
                    self.training_loss = available_metrics[0]

        # self.model_config["allow_large_cpu_dataset"]
        # self.model_config["max_sequence_length"]

        # Set device - default to CPU for TabPFN
        # self.device = model_config.get("device", "cpu")
        self.model = None
        self.models = []  # For anyvariate mode
        self.is_fitted = False

    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "TabpfnModel":
        # Zero-shot TabPFN uses context during predict; mark as fitted

        self.is_fitted = True
        return self

    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):
        forecast_horizon = timestamps_target.shape[0]
        context_window = self.model_config["context_window"]
        forecast_window = self.model_config["forecast_window"]

        # Fit the model on the current context window
        regressor = TabPFNRegressor()

        timestamps_context = timestamps_context[-context_window:]
        y_context = y_context[-context_window:]

        print("Fitting TabFPN")
        regressor.fit(timestamps_context, y_context)

        y_pred = []
        steps_left = forecast_horizon

        for step in range(math.ceil(forecast_horizon / forecast_window)):

            timestamps_curr = timestamps_target[
                forecast_window * step : forecast_window * (step + 1), :
            ]

            y_pred_curr = regressor.predict(timestamps_curr)
            y_pred_curr = np.expand_dims(y_pred_curr, axis=1)
            y_pred.append(y_pred_curr)

            # Update the context and target for the next iteration
            timestamps_context = np.concatenate(
                [timestamps_context, timestamps_curr], axis=0
            )
            y_context = np.concatenate([y_context, y_pred_curr], axis=0)

        forecasts = np.concatenate(y_pred, axis=0)

        return forecasts

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ):
        """
        Train the TabPFN model. Supports anyvariate mode for handling multivariate data.
        
        Args:
            y_context: Past target values - training data
            y_target: Future target values (for validation)
            timestamps_context: Timestamps for y_context
            timestamps_target: Timestamps for y_target
            freq: Frequency string
            **kwargs: Additional keyword arguments
            
        Returns:
            self: The fitted model instance
        """
        # Ensure y_context is 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
        if y_target is not None and y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)

        n_variates = y_context.shape[1]

        # For univariate case or non-anyvariate mode, use the original method
        if n_variates == 1 or not self.is_anyvariate:
            return self._train(y_context, y_target, timestamps_context, timestamps_target, freq, **kwargs)

        # For anyvariate mode with multivariate data, train separate TabPFN models for each variate
        self.models = []  # Store individual models for each variate

        for variate_idx in range(n_variates):
            print(f"[INFO] Training TabPFN model for variate {variate_idx + 1}/{n_variates}")

            # Create a new TabPFN model for this variate
            variate_model = TabpfnModel(self.full_config)

            # Extract data for this specific variate
            variate_context = y_context[:, variate_idx:variate_idx+1]
            variate_target = y_target[:, variate_idx:variate_idx+1] if y_target is not None else None

            # Train the model for this variate
            variate_model._train(
                y_context=variate_context,
                y_target=variate_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs
            )

            # Store the trained model
            self.models.append(variate_model)

        # Mark the main model as fitted for anyvariate case
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray],
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained TabPFN model. Supports anyvariate mode.
        
        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for y_context
            timestamps_target: Timestamps for the prediction horizon
            freq: Frequency string
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Model predictions
        """
        # Ensure y_context is 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        n_variates = y_context.shape[1]

        # For univariate case or non-anyvariate mode, use the original method
        if n_variates == 1 or not self.is_anyvariate:
            return self._predict(y_context, timestamps_context, timestamps_target, freq, **kwargs)

        # For anyvariate mode with multivariate data, predict using each variate's model
        if not self.models:
            raise ValueError("No variate models found. Train the model first.")

        all_predictions = []
        for variate_idx in range(n_variates):
            print(f"[INFO] Predicting with TabPFN model for variate {variate_idx + 1}/{n_variates}")
            
            # Extract data for this specific variate
            variate_context = y_context[:, variate_idx:variate_idx+1]
            
            # Make predictions for this variate
            variate_pred = self.models[variate_idx]._predict(
                y_context=variate_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs
            )
            
            all_predictions.append(variate_pred)

        # Combine predictions from all variates
        combined_predictions = np.concatenate(all_predictions, axis=1)
        
        # Store for TensorBoard logging (use first variate as representative)
        self._last_y_pred = all_predictions[0]
        
        return combined_predictions

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        loss_function: str = None,
        y_train: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute loss metrics, supporting per-variate metrics in anyvariate mode.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Loss function to use (inherited from parent if None)
            y_train: Training data for metrics like MASE
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        # Ensure inputs are 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_train is not None and y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        n_variates = y_true.shape[1]

        # For univariate case or non-anyvariate mode, use the parent method
        if n_variates == 1 or not self.is_anyvariate:
            return super().compute_loss(y_true, y_pred, loss_function, y_train)

        # For anyvariate mode, compute metrics per variate
        all_metrics = {}
        aggregated_metrics = {}

        for variate_idx in range(n_variates):
            # Extract data for this variate
            y_true_variate = y_true[:, variate_idx:variate_idx+1]
            y_pred_variate = y_pred[:, variate_idx:variate_idx+1]
            y_train_variate = y_train[:, variate_idx:variate_idx+1] if y_train is not None else None

            # Compute metrics for this variate using the evaluator
            variate_metrics = self.evaluator.evaluate(y_pred_variate, y_true_variate, y_train=y_train_variate)

            # Add variate index to metric names for per-variate tracking
            for metric_name, metric_value in variate_metrics.items():
                all_metrics[f"{metric_name}_variate_{variate_idx + 1}"] = metric_value

                # Also accumulate for aggregated metrics
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = []
                aggregated_metrics[metric_name].append(metric_value)

        # Add aggregated metrics (mean across variates) for hyperparameter optimization
        for metric_name, values in aggregated_metrics.items():
            all_metrics[metric_name] = np.mean(values)

        # Store for TensorBoard logging (use first variate as representative)
        self._last_y_true = y_true[:, 0]
        self._last_y_pred = y_pred[:, 0]

        return all_metrics
