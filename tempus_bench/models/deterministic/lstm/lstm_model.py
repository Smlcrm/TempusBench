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
import math

from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from ...base_model import BaseModel, validate_inputs

class LstmHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    units: int = Field(..., ge=1, description="Number of LSTM units")
    layers: int = Field(..., ge=1, description="Number of LSTM layers")
    dropout: float = Field(..., ge=0, le=1, description="Dropout rate")
    learning_rate: float = Field(..., gt=0, description="Learning rate for optimizer")
    ### Fixed Hyperparameters - Optional for User to override
    batch_size: Optional[int] = Field(default=32, ge=1, description="Batch size for training")
    epochs: Optional[int] = Field(default=50, ge=1, description="Number of training epochs")


class LSTMModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, LstmHyperparams)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "LSTMModel":
        """
        Train the Multivariate LSTM model on given data.

        Uses sliding window, multi-step learning for multiple targets.

        Args:
            y_context: Past target values, shape (num_steps, num_targets)
            y_target: Future target values (for supervised labels)
            timestamps_context: Timestamps for context (unused here)
            timestamps_target: Timestamps for target (unused here)
            **kwargs: Extra options (currently ignored)

        Returns:
            self: The fitted model instance
        """

        _, num_targets = y_context.shape

        # Combine context and target to make training sequences
        combined_data = np.concatenate([y_context, y_target], axis=0)
        X_seq, y_seq = self._prepare_sequences(combined_data)

        # Build model if not done yet
        if not "_model" in self.__dict__:
            self._build_model(
                input_shape=(self.model_config["context_length"], num_targets)
            )

        # Setup early stopping
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        validation_split = 0.1 if len(X_seq) > 10 else 0.0
        self.model.fit(
            X_seq,
            y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=[early_stopping] if validation_split > 0 else [],
            validation_split=validation_split
        )

        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Make predictions with the trained Multivariate LSTM model.

        Predicts the required number of steps ahead for all targets using non-overlapping multi-step windows.
        Does not use own predictions as further inputs (no autoregressive feedback).

        Args:
            y_context: Context/history values (n_steps, n_targets)
            timestamps_context: Timestamps for context (unused)
            timestamps_target: Timestamps for target (unused)
            freq: Frequency string (unused)

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, n_targets)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")

        forecast_length, num_targets = y_context.shape
        context_length = kwargs["context_window"]
        prediction_window = kwargs["forecast_horizon"]
        num_windows = math.ceil(forecast_length / prediction_window)

        all_predictions = []
        current_sequence = y_context[-context_length:].reshape(
            1, context_length, num_targets
        )

        for window in range(num_windows):
            preds = self._model.predict(current_sequence, verbose=0)
            preds_reshaped = preds[0].reshape(prediction_window, num_targets)
            all_predictions.extend(preds_reshaped)

            # Only update current_sequence if not last window
            if window < num_windows - 1:
                n = min(prediction_window, context_length)
                # Create the next sequence by rolling and appending new predictions
                current_sequence = np.roll(current_sequence, -prediction_window, axis=1)
                current_sequence[0, -n:, :] = preds_reshaped[:n, :]

        result = np.array(all_predictions[:forecast_length])

        if result.ndim == 1:
            result = np.expand_dims(result, axis=1)

        return result

    def _build_model(self,
        input_shape: Tuple[int, int],
        **kwargs: dict) -> None:
        """
        Build the Multivariate LSTM model architecture.

        Args:
            input_shape: Shape of input data (context_length, num_targets)
        """

        _, num_targets = input_shape
        self._model = Sequential()

        # Add LSTM layers
        for layer_idx in range(self.layers):
            return_sequences = layer_idx < self.layers - 1
            self._model.add(
                LSTM(
                    units=self.units,
                    return_sequences=return_sequences,
                    input_shape=input_shape if layer_idx == 0 else None,
                ),
            )
            if self.dropout > 0:
                self._model.add(Dropout(self.dropout))

        # Add output layer - predicts prediction_window * num_targets values (flattened)
        self._model.add(Dense(self.prediction_window * num_targets))

        # Compile model with optimized settings
        self._model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon
            ),
            loss=kwargs["tuning_loss"],
            metrics=kwargs["tuning_loss"]
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
            - self.context_length
            - self.prediction_window
            + 1
        ):
            curr_X = X[i : (i + self.context_length)]
            # curr_X = curr_X.flatten()

            X_seq.append(curr_X)
            # y_seq: flatten to 1D array of length prediction_window * num_targets
            future_values = X[
                i
                + self.context_length : i
                + self.context_length
                + self.prediction_window
            ]
            future_values = future_values.flatten()
            y_seq.append(future_values)
        return np.array(X_seq), np.array(y_seq)
