import os
import json
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
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
        self._build_model()

    def _build_model(self):
        self.model = Prophet(**self.model_config)
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
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ):

        if not self.is_fitted:
            self.model = Prophet(**self.model_config)
        # Use the provided timestamps to create a DatetimeIndex for y_context
        # Ensure 1D series indexed by provided timestamps
        y_context = y_context.squeeze()
        y_target = y_target.squeeze()

        timestamps_context = self.convert_to_datetimeindex(timestamps_context)
        timestamps_target = self.convert_to_datetimeindex(timestamps_target)

        train_df = pd.DataFrame({"ds": timestamps_context, "y": y_context})
        
        # Add covariates as additional regressors if provided
        if x_context is not None and x_target is not None:
            # Concatenate context and target covariates for training
            x_all = np.concatenate([x_context, x_target], axis=0)
            
            # Ensure x_all is 2D
            if x_all.ndim == 1:
                x_all = x_all.reshape(-1, 1)
            
            # Add each covariate as a regressor
            num_covariates = x_all.shape[1]
            self.num_covariates = num_covariates
            
            for i in range(num_covariates):
                regressor_name = f"covariate_{i}"
                # Add only context covariates to training dataframe
                train_df[regressor_name] = x_context[:, i] if x_context.ndim > 1 else x_context
                # Add regressor to model
                self.model.add_regressor(regressor_name)

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
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained Prophet model.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            freq: Frequency string from CSV data - MUST be provided
            x_context: Covariates for context period (not used in prediction)
            x_target: Covariates for target period (used for forecasting)

        Returns:
            np.ndarray: Model predictions

        Raises:
            ValueError: If freq is None or if required data is missing
        """

        # Create future dataframe with the correct timestamps
        future_df = pd.DataFrame(
            {"ds": self.convert_to_datetimeindex(timestamps_target)}
        )
        
        # Add covariates to future dataframe if they were used in training
        if x_target is not None and hasattr(self, 'num_covariates'):
            # Ensure x_target is 2D
            if x_target.ndim == 1:
                x_target = x_target.reshape(-1, 1)
            
            for i in range(self.num_covariates):
                regressor_name = f"covariate_{i}"
                future_df[regressor_name] = x_target[:, i] if x_target.ndim > 1 else x_target

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
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Train the Prophet model on multivariate time series data.
        For multivariate data, this method applies the univariate Prophet model
        to each variate separately.

        Args:
            y_context: Training target values (shape: [n_samples, n_variates])
            y_target: Target values for evaluation (optional)
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            freq: Frequency string from CSV data
            x_context: Covariates for context period (optional)
            x_target: Covariates for target period (optional)
            **kwargs: Additional arguments

        Returns:
            self: Trained model instance
        """
        # Handle multivariate data by training separate Prophet models for each variate
        if y_context.ndim > 1 and y_context.shape[1] > 1:
            # Multivariate case: train separate models for each variate
            self.models = []  # Store list of Prophet models for each variate
            
            for variate_idx in range(y_context.shape[1]):
                # Extract single variate data
                y_context_variate = y_context[:, variate_idx]
                y_target_variate = y_target[:, variate_idx] if y_target is not None else None
                
                # Create a temporary model instance for this variate
                # Use the full config but replace the model section with the specific model config
                inner_config = self.config.copy()
                inner_config["model"] = {"prophet": self.model_config}
                temp_model = ProphetModel(inner_config)
                
                # Train on this variate using the univariate method
                temp_model._train(
                    y_context=y_context_variate,
                    y_target=y_target_variate,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                    x_context=x_context,
                    x_target=x_target,
                    **kwargs
                )
                
                self.models.append(temp_model)
            
            # Set the main model to the first variate for compatibility
            self.model = self.models[0].model
            self.is_fitted = True
            
        else:
            # Univariate case: use the original univariate method
            self._train(
                y_context=y_context,
                y_target=y_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                x_context=x_context,
                x_target=x_target,
                **kwargs
            )
        
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Make predictions using the trained Prophet model(s).
        For multivariate data, this method applies the univariate Prophet model
        to each variate separately and combines the results.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            freq: Frequency string from CSV data
            x_context: Covariates for context period (optional)
            x_target: Covariates for target period (optional)

        Returns:
            np.ndarray: Model predictions (shape: [n_forecast_steps, n_variates])
        """
        # Handle multivariate data by predicting with separate Prophet models for each variate
        if hasattr(self, 'models') and self.models:
            # Multivariate case: predict with each variate model and combine
            predictions = []
            
            for variate_idx, model in enumerate(self.models):
                # Extract single variate context if provided
                y_context_variate = y_context[:, variate_idx] if y_context is not None else None
                
                # Predict for this variate
                variate_pred = model._predict(
                    y_context=y_context_variate,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                    x_context=x_context,
                    x_target=x_target
                )
                
                predictions.append(variate_pred)
            
            # Combine predictions from all variates
            combined_predictions = np.column_stack(predictions)
            
            # Inverse transform predictions if scaler is available
            if self.scaler is not None:
                combined_predictions = self.scaler.inverse_transform(combined_predictions)
            
            return combined_predictions
            
        else:
            # Univariate case: use the original univariate method
            return self._predict(
                y_context=y_context,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                x_context=x_context,
                x_target=x_target
            )