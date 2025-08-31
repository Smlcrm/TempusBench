"""
Multivariate LSTM model implementation.

This model extends the univariate LSTM to handle multiple target variables simultaneously.
Design choices (all Option A):
- Single output layer that predicts forecast_horizon * n_targets values (flattened)
- Concatenate all targets into a single input sequence (context_length, n_targets)
- Sum/average losses across all targets
- Predict all targets simultaneously in one forward pass
- Keep same LSTM architecture, just change input/output dimensions
"""

import numpy as np
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel
import time
from tensorflow.keras.callbacks import TensorBoard


class MultivariateLSTMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Multivariate LSTM model with given configuration.

        Args:
            config: Configuration dictionary containing model parameters
                - units: int, number of LSTM units
                - layers: int, number of LSTM layers
                - dropout: float, dropout rate
                - learning_rate: float, learning rate for optimizer
                - batch_size: int, batch size for training
                - epochs: int, number of training epochs
                - context_length: int, length of input sequences
                - target_cols: list of str, names of target columns (for multivariate)
                - feature_cols: list of str, names of feature columns
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config)
        if "units" not in self.model_config:
            raise ValueError("units must be specified in config")
        if "layers" not in self.model_config:
            raise ValueError("layers must be specified in config")
        if "dropout" not in self.model_config:
            raise ValueError("dropout must be specified in config")
        if "learning_rate" not in self.model_config:
            raise ValueError("learning_rate must be specified in config")
        if "batch_size" not in self.model_config:
            raise ValueError("batch_size must be specified in config")
        if "epochs" not in self.model_config:
            raise ValueError("epochs must be specified in config")
        if "context_length" not in self.model_config:
            raise ValueError("context_length must be specified in config")
        if "prediction_window" not in self.model_config:
            raise ValueError("prediction_window must be specified in config")
        self.model = None
        # num_targets will be calculated from data during training

    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the Multivariate LSTM model architecture.

        Args:
            input_shape: Shape of input data (context_length, num_targets)
        """

        context_length, num_targets = input_shape
        self.model = Sequential()

        # Add LSTM layers
        for i in range(self.model_config["layers"]):
            return_sequences = i < self.model_config["layers"] - 1
            self.model.add(
                LSTM(
                    units=self.model_config["units"],
                    return_sequences=return_sequences,
                    input_shape=input_shape if i == 0 else None,
                ),
            )
            if self.model_config["dropout"] > 0:
                self.model.add(Dropout(self.model_config["dropout"]))

        # Add output layer - predicts prediction_window * num_targets values (flattened)
        self.model.add(Dense(self.model_config["prediction_window"] * num_targets))

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.model_config["learning_rate"]),
            loss=self.training_loss,
        )

    def _prepare_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for Multivariate LSTM.

        Args:
            X: Input features (2D array with shape (num_timesteps, num_targets))

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared sequences and targets
        """
        # # Ensure X is 2D: (num_timesteps, num_targets) for multivariate
        # if X.ndim == 1:
        #     # If univariate input, reshape to (num_timesteps, 1)
        #     X = X.reshape(-1, 1)
        # elif X.ndim > 2:
        #     # If 3D or higher, flatten to 2D
        #     X = X.reshape(X.shape[0], -1)

        X_seq, y_seq = [], []
        for i in range(
            len(X)
            - self.model_config["context_length"]
            - self.model_config["prediction_window"]
            + 1
        ):
            curr_X = X[i : (i + self.model_config["context_length"])]
            # curr_X = curr_X.flatten()

            X_seq.append(curr_X)
            # y_seq: flatten to 1D array of length prediction_window * num_targets
            future_values = X[
                i
                + self.model_config["context_length"] : i
                + self.model_config["context_length"]
                + self.model_config["prediction_window"]
            ]
            future_values = future_values.flatten()
            y_seq.append(future_values)
        return np.array(X_seq), np.array(y_seq)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "MultivariateLSTMModel":
        """
        Train the Multivariate LSTM model on given data.

        TECHNIQUE: Sliding Window Multi-Step Learning for Multiple Targets
        - Creates overlapping sequences of length context_length from historical data
        - Each sequence predicts prediction_window future values for all targets simultaneously
        - Training pairs: Input [t, t+1, ..., t+context_length-1] → Target [t+context_length, ..., t+context_length+prediction_window-1]
        - Model learns to predict multiple future steps for multiple targets in a single forward pass
        - Captures temporal dependencies across multiple future time steps and target variables

        Args:
            y_context: Past target values (time series) - used for training (can be DataFrame for multivariate)
            y_target: Future target values (optional, for validation)
            y_start_date: The start date timestamp for y_context and y_target in string form
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Convert input to numpy array and handle multivariate data
        forecast_length, num_targets = y_target.shape

        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(y_context)

        print("X shape:", X_seq.shape)
        print("y shape:", y_seq.shape)

        # Build model if not already built
        if self.model is None:
            self._build_model(
                input_shape=(self.model_config["context_length"], num_targets)
            )

        # Train model
        self.model.fit(
            X_seq,
            y_seq,
            batch_size=self.model_config["batch_size"],
            epochs=self.model_config["epochs"],
            verbose=0,
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Make predictions using the trained Multivariate LSTM model.
        Predicts len(y_target) steps ahead for all targets using non-overlapping windows.

        TECHNIQUE: Non-overlapping Multi-Step Windows for Multiple Targets
        - Starts with last context_length historical values for all targets
        - Predicts prediction_window steps at once for all targets (no autoregressive feedback)
        - Advances window by prediction_window steps for next prediction
        - Repeats until len(y_target) steps are predicted for all targets
        - Fair comparison with non-autoregressive models (ARIMA, Exponential Smoothing)
        - No data leakage from using own predictions as input

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """

        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")

        forecast_length, num_targets = y_context.shape

        # Calculate how many prediction windows we need
        num_windows = math.ceil(
            forecast_length / self.model_config["prediction_window"]
        )

        all_predictions = []
        current_sequence = y_context[-self.model_config["context_length"] :].reshape(
            1, self.model_config["context_length"], num_targets
        )

        for window in range(num_windows):
            # Predict prediction_window steps at once for all targets
            predictions = self.model.predict(current_sequence, verbose=0)
            # Reshape predictions from (1, prediction_window * num_targets) to (prediction_window, num_targets)
            predictions_reshaped = predictions[0].reshape(
                self.model_config["prediction_window"], num_targets
            )
            all_predictions.extend(predictions_reshaped)

            # Advance window by prediction_window steps and use predictions as input
            if window < num_windows - 1:  # Don't update on last iteration
                # Move window forward by prediction_window steps
                current_sequence = np.roll(
                    current_sequence, -self.model_config["prediction_window"], axis=1
                )
                # Only update as many as fit
                n = min(
                    self.model_config["prediction_window"],
                    self.model_config["context_length"],
                )
                current_sequence[0, -n:, :] = predictions_reshaped[:n, :]

        # Return only the requested number of predictions
        result = np.array(all_predictions[:forecast_length])

        if len(result.shape) == 1:
            result = np.expand_dims(result, axis=1)

        return result
