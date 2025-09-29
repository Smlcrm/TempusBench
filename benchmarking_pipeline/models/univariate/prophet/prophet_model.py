import os
import json
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from benchmarking_pipeline.models.base_model import BaseModel


class ProphetModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Prophet model with a given configuration.

        Args:
            config: Configuration dictionary for Prophet parameters.
                    e.g., {'model_params': {'seasonality_mode': 'multiplicative'}}
            config_file: Path to a JSON configuration file.
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
                print(f"[PROPHET] Anyvariate mode enabled for config: {config_filename}")

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

        # Initialize model state
        self.models = []  # For anyvariate mode
        
        self._build_model()

    def _build_model(self):
        # Extract Prophet-specific parameters from the model config section
        # Get the actual prophet model config (either from model_config for hyperparameter tuning or from full config)
        if hasattr(self, 'model_config') and isinstance(self.model_config, dict) and 'seasonality_mode' in self.model_config:
            prophet_config = self.model_config
        elif 'model' in self.full_config and 'prophet' in self.full_config['model']:
            prophet_config = self.full_config['model']['prophet']
        else:
            prophet_config = self.model_config

        # Filter out any non-Prophet parameters that might have been included
        valid_prophet_params = [
            'growth', 'changepoints', 'n_changepoints', 'changepoint_range',
            'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
            'holidays', 'seasonality_mode', 'seasonality_prior_scale',
            'holidays_prior_scale', 'changepoint_prior_scale', 'mcmc_samples',
            'interval_width', 'uncertainty_samples', 'stan_backend'
        ]
        
        filtered_config = {k: v for k, v in prophet_config.items() if k in valid_prophet_params}
        
        self.model = Prophet(**filtered_config)
        self.is_fitted = False

    @staticmethod
    def ensure_series_with_datetimeindex(y, start_date, freq):
        """
        Ensure the input series has a proper datetime index.

        Args:
            y: Input series (can be numpy array, pandas series, or already indexed)
            start_date: Start date to use for the index
            freq: Frequency from CSV data - MUST be provided

        Returns:
            pd.Series: Series with proper datetime index

        Raises:
            ValueError: If freq is None or empty
        """
        if freq is None or freq == "":
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )

        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            return y
        return pd.Series(
            y.values if hasattr(y, "values") else y,
            index=pd.date_range(start_date, periods=len(y), freq=freq),
        )

    def convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                min_ts = np.min(timestamps)
                max_ts = np.max(timestamps)

                # Pandas datetime bounds for 64-bit ns: 1677-09-21 to 2262-04-11
                # 1677-09-21T00:12:43.145224192Z = -9223372036854775808 ns
                # 2262-04-11T23:47:16.854775807Z = 9223372036854775807 ns
                NS_LOWER = -9223372036854775808
                NS_UPPER = 9223372036854775807
                US_LOWER = NS_LOWER // 1000
                US_UPPER = NS_UPPER // 1000
                MS_LOWER = NS_LOWER // 1_000_000
                MS_UPPER = NS_UPPER // 1_000_000
                S_LOWER = NS_LOWER // 1_000_000_000
                S_UPPER = NS_UPPER // 1_000_000_000

                def in_bounds(val, lower, upper):
                    return lower <= val <= upper

                # Try to classify the likely unit and check bounds
                unit = None
                if isinstance(min_ts, (int, np.integer)):
                    # Try nanoseconds
                    if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(
                        max_ts, NS_LOWER, NS_UPPER
                    ):
                        unit = "ns"
                    # Try microseconds
                    elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(
                        max_ts, US_LOWER, US_UPPER
                    ):
                        unit = "us"
                    # Try milliseconds
                    elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(
                        max_ts, MS_LOWER, MS_UPPER
                    ):
                        unit = "ms"
                    # Try seconds
                    elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(
                        max_ts, S_LOWER, S_UPPER
                    ):
                        unit = "s"
                    else:
                        raise ValueError(
                            f"Timestamps are out of bounds for pandas datetime64[ns] (min={min_ts}, max={max_ts})."
                        )
                    timestamps = pd.to_datetime(timestamps, unit=unit)
                else:
                    timestamps = pd.to_datetime(timestamps)

        return timestamps

    def _train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ):

        if not self.is_fitted:
            # Extract Prophet-specific parameters from the model config section
            # Get the actual prophet model config (either from model_config for hyperparameter tuning or from full config)
            if hasattr(self, 'model_config') and isinstance(self.model_config, dict) and 'seasonality_mode' in self.model_config:
                prophet_config = self.model_config
            elif 'model' in self.full_config and 'prophet' in self.full_config['model']:
                prophet_config = self.full_config['model']['prophet']
            else:
                prophet_config = self.model_config

            # Filter out any non-Prophet parameters that might have been included
            valid_prophet_params = [
                'growth', 'changepoints', 'n_changepoints', 'changepoint_range',
                'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
                'holidays', 'seasonality_mode', 'seasonality_prior_scale',
                'holidays_prior_scale', 'changepoint_prior_scale', 'mcmc_samples',
                'interval_width', 'uncertainty_samples', 'stan_backend'
            ]
            
            filtered_config = {k: v for k, v in prophet_config.items() if k in valid_prophet_params}
            
            self.model = Prophet(**filtered_config)
        # Use the provided timestamps to create a DatetimeIndex for y_context
        # Ensure 1D series indexed by provided timestamps
        y_context = y_context.squeeze()
        y_target = y_target.squeeze()

        timestamps_context = self.convert_to_datetimeindex(timestamps_context)
        timestamps_target = self.convert_to_datetimeindex(timestamps_target)

        train_df = pd.DataFrame({"ds": timestamps_context, "y": y_context})

        self.model.fit(train_df)
        # Store training statistics for fallback predictions
        self.is_fitted = True
        return self

    def _predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained Prophet model.

        Args:
            y_context: Recent/past target values
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            y_context_timestamps: Timestamps for context data
            y_target: Target values for evaluation (optional)
            y_target_timestamps: Timestamps for target data (optional)
            freq: Frequency string from CSV data - MUST be provided

        Returns:
            np.ndarray: Model predictions

        Raises:
            ValueError: If freq is None or if required data is missing
        """

        # Default forecast horizon if neither y_target nor y_target_timestamps provided

        # Create future dataframe with the correct timestamps
        future_df = pd.DataFrame(
            {"ds": self.convert_to_datetimeindex(timestamps_target)}
        )

        # Make predictions
        forecast = self.model.predict(future_df)

        forecast = np.asarray(forecast["yhat"])

        if len(forecast.shape) == 1:
            forecast = np.expand_dims(forecast, axis=1)
        # Return the predicted values

        return forecast

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
        Train the Prophet model. Supports anyvariate mode for handling multivariate data.
        
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

        # For anyvariate mode with multivariate data, train separate Prophet models for each variate
        self.models = []  # Store individual models for each variate

        for variate_idx in range(n_variates):
            print(f"[INFO] Training Prophet model for variate {variate_idx + 1}/{n_variates}")

            # Create a new Prophet model for this variate
            variate_model = ProphetModel(self.full_config)

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
        Make predictions using the trained Prophet model. Supports anyvariate mode.
        
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
            print(f"[INFO] Predicting with Prophet model for variate {variate_idx + 1}/{n_variates}")
            
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
