"""
LSTM model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
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


class LstmModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM model with given configuration.

        Args:
            config: Configuration dictionary containing model parameters
                - units: int, number of LSTM units
                - layers: int, number of LSTM layers
                - dropout: float, dropout rate
                - learning_rate: float, learning rate for optimizer
                - batch_size: int, batch size for training
                - epochs: int, number of training epochs
                - sequence_length: int, length of input sequences
                - feature_cols: list of str, names of feature columns
                - loss_function: str, loss function for training
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
        if "sequence_length" not in self.model_config:
            raise ValueError("sequence_length must be specified in config")
        if "training_loss" not in self.model_config:
            raise ValueError("training_loss must be specified in config")

        self.model = None

    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        self.model = Sequential()

        # Add LSTM layers
        for i in range(self.model_config["layers"]):
            return_sequences = i < self.model_config["layers"] - 1
            self.model.add(
                LSTM(
                    units=self.model_config["units"],
                    return_sequences=return_sequences,
                    input_shape=input_shape if i == 0 else None,
                )
            )
            if self.model_config["dropout"] > 0:
                self.model.add(Dropout(self.model_config["dropout"]))

        # Add output layer
        self.model.add(Dense(self.model_config["forecast_horizon"]))

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.model_config["learning_rate"]),
            loss=self.training_loss,
        )

    def _prepare_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for LSTM.

        Args:
            X: Input features (1D or 2D array)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared sequences and targets
        """
        # Ensure X is 2D: (num_timesteps, 1) for univariate
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_seq, y_seq = [], []
        for i in range(
            len(X)
            - self.model_config["sequence_length"]
            - self.model_config["forecast_horizon"]
            + 1
        ):
            X_seq.append(X[i : (i + self.model_config["sequence_length"])])
            # y_seq: flatten to 1D if forecast_horizon == 1, else keep as 1D array
            y_seq.append(
                X[
                    i
                    + self.model_config["sequence_length"] : i
                    + self.model_config["sequence_length"]
                    + self.model_config["forecast_horizon"],
                    0,
                ]
            )
        return np.array(X_seq), np.array(y_seq)

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> "LstmModel":
        """
        Train the LSTM model on given data.

        TECHNIQUE: Sliding Window Multi-Step Learning
        - Creates overlapping sequences of length sequence_length from historical data
        - Each sequence predicts forecast_horizon future values simultaneously
        - Training pairs: Input [t, t+1, ..., t+sequence_length-1] → Target [t+sequence_length, ..., t+sequence_length+forecast_horizon-1]
        - Model learns to predict multiple future steps in a single forward pass
        - Captures temporal dependencies across multiple future time steps

        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values (optional, for validation)
            y_start_date: The start date timestamp for y_context and y_target in string form
            **kwargs: Additional keyword arguments


        Returns:
            self: The fitted model instance
        """

        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(y_context)

        # Build model if not already built
        if self.model is None:
            self._build_model(input_shape=(self.model_config["sequence_length"], 1))
        # TensorBoard logging is handled by the main benchmark runner
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
        y_context,
        timestamps_context,
        timestamps_target,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        Predicts len(y_target) steps ahead using non-overlapping windows.

        TECHNIQUE: Non-overlapping Multi-Step Windows
        - Starts with last sequence_length historical values
        - Predicts forecast_horizon steps at once (no autoregressive feedback)
        - Advances window by forecast_horizon steps for next prediction
        - Repeats until len(y_target) steps are predicted
        - Fair comparison with non-autoregressive models (ARIMA, Exponential Smoothing)
        - No data leakage from using own predictions as input

        Returns:
            np.ndarray: Model predictions with shape (1, forecast_steps)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")

        forecast_steps = len(timestamps_target)

        # Calculate how many prediction windows we need
        num_windows = math.ceil(
            (forecast_steps) // self.model_config["forecast_horizon"]
        )

        all_predictions = []
        current_sequence = y_context[-self.model_config["sequence_length"] :].reshape(
            1, self.model_config["sequence_length"], 1
        )

        for window in range(num_windows):
            # Predict forecast_horizon steps at once
            predictions = self.model.predict(current_sequence, verbose=0)
            all_predictions.extend(predictions[0])
            # Advance window by forecast_horizon steps and use predictions as input
            if window < num_windows - 1:  # Don't update on last iteration
                # Move window forward by forecast_horizon steps
                current_sequence = np.roll(
                    current_sequence, -self.model_config["forecast_horizon"], axis=1
                )
                # Only update as many as fit
                n = min(
                    self.model_config["forecast_horizon"],
                    self.model_config["sequence_length"],
                )
                current_sequence[0, -n:, 0] = predictions[0][:n]

        # Return only the requested number of predictions
        return np.array(all_predictions[:forecast_steps]).reshape(1, -1)
