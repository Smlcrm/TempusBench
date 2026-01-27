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
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from tempus_bench.models.base_model import BaseModel, validate_inputs


class LstmHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    units: int = Field(default=32, ge=1, description="Number of LSTM units")
    layers: int = Field(default=2, ge=1, description="Number of LSTM layers")
    dropout: float = Field(default=0.3, ge=0, le=1, description="Dropout rate")
    learning_rate: float = Field(..., gt=0, description="Learning rate for optimizer")
    ### Fixed Hyperparameters - Optional for User to override
    batch_size: int = Field(default=32, ge=1, description="Batch size for training")
    epochs: int = Field(default=500, ge=1, description="Number of training epochs")
    context_length: int = Field(default=32, ge=1, description="Context length")
    prediction_window: int = Field(default=8, ge=1, description="Prediction window")


class LstmModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, LstmHyperparams)  # type: ignore

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "LstmModel":
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

        # Ensure input data is numeric (float32) - Keras requires numeric dtypes
        # Convert to new arrays to avoid issues with read-only arrays or object dtypes
        y_context = np.array(y_context, dtype=np.float32, copy=True)
        y_target = np.array(y_target, dtype=np.float32, copy=True)

        # Combine context and target to make training sequences
        combined_data = np.concatenate([y_context, y_target], axis=0)
        X_seq, y_seq = self._prepare_sequences(combined_data)

        # Build model if not done yet
        if "_model" not in self.__dict__:
            if "tuning_loss" not in kwargs:
                raise ValueError(
                    "tuning_loss must be provided in kwargs for LSTM model"
                )
            self._build_model(input_shape=(self.context_length, num_targets), tuning_loss=kwargs["tuning_loss"])  # type: ignore

        # Setup early stopping
        early_stopping = EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True, verbose=1
        )

        validation_split = 0.1 if len(X_seq) > 10 else 0.0

        # Ensure X_seq and y_seq are float32 and contiguous before passing to Keras
        # Force conversion to float32 and check for any string/object dtypes
        if X_seq.dtype.kind in ["O", "U", "S"]:  # Object, Unicode, or String types
            # Convert to numeric first, handling any string values
            X_seq = np.array(X_seq, dtype=np.float64, copy=True)
        X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)

        if y_seq.dtype.kind in ["O", "U", "S"]:  # Object, Unicode, or String types
            # Convert to numeric first, handling any string values
            y_seq = np.array(y_seq, dtype=np.float64, copy=True)
        y_seq = np.ascontiguousarray(y_seq, dtype=np.float32)

        # Final check - ensure arrays are numeric
        if not np.issubdtype(X_seq.dtype, np.number) or not np.issubdtype(
            y_seq.dtype, np.number
        ):
            raise ValueError(
                f"Data must be numeric. Got X_seq dtype: {X_seq.dtype}, y_seq dtype: {y_seq.dtype}"
            )

        # Final verification - ensure arrays are truly numeric float32
        assert X_seq.dtype == np.float32, f"X_seq must be float32, got {X_seq.dtype}"
        assert y_seq.dtype == np.float32, f"y_seq must be float32, got {y_seq.dtype}"
        assert X_seq.flags["C_CONTIGUOUS"], "X_seq must be contiguous"
        assert y_seq.flags["C_CONTIGUOUS"], "y_seq must be contiguous"

        self._model.fit(
            X_seq,
            y_seq,
            batch_size=int(self.batch_size),  # type: ignore - ensure int
            epochs=int(self.epochs),  # type: ignore - ensure int
            verbose=1,
            callbacks=[early_stopping] if validation_split > 0 else [],
            validation_split=float(validation_split) if validation_split > 0 else None,
        )

        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
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
        if not self.is_fitted or "_model" not in self.__dict__:
            raise ValueError("Model not fitted. Call train() first.")

        # Ensure input data is numeric (float32) - Keras requires numeric dtypes
        # Convert to new array to avoid issues with read-only arrays or object dtypes
        y_context = np.array(y_context, dtype=np.float32, copy=True)

        # Get forecast length from timestamps_target, not y_context shape
        forecast_length = len(timestamps_target)
        num_context_steps, num_targets = y_context.shape
        context_length = self.context_length  # type: ignore
        prediction_window = self.prediction_window  # type: ignore

        # Ensure we have enough context data
        if num_context_steps < context_length:
            raise ValueError(
                f"y_context must have at least {context_length} timesteps, "
                f"but got {num_context_steps}"
            )

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

        result = np.asarray(all_predictions[:forecast_length], dtype=np.float32)

        if result.ndim == 1:
            result = np.expand_dims(result, axis=1)

        return result

    def _build_model(self, input_shape: Tuple[int, int], tuning_loss: str) -> None:
        """
        Build the Multivariate LSTM model architecture.

        Args:
            input_shape: Shape of input data (context_length, num_targets)
        """

        _, num_targets = input_shape
        self._model = Sequential()

        # Add LSTM layers
        for layer_idx in range(self.layers):  # type: ignore
            return_sequences = layer_idx < self.layers - 1  # type: ignore
            self._model.add(
                LSTM(
                    units=self.units,  # type: ignore
                    return_sequences=return_sequences,
                    input_shape=input_shape if layer_idx == 0 else None,
                ),
            )
            if self.dropout > 0:  # type: ignore
                self._model.add(Dropout(self.dropout))  # type: ignore

        # Add output layer - predicts prediction_window * num_targets values (flattened)
        self._model.add(Dense(self.prediction_window * num_targets))  # type: ignore

        # Map tuning_loss to Keras loss function names
        loss_map = {
            "mae": "mean_absolute_error",
            "mase": "mean_absolute_error",  # MASE not directly available, use MAE
            "mape": "mean_absolute_percentage_error",
            "rmse": "root_mean_squared_error",
            "mse": "mean_squared_error",
        }

        keras_loss = loss_map.get(tuning_loss.lower(), tuning_loss.lower())

        # Compile model with optimized settings
        # Ensure optimizer parameters are floats
        self._model.compile(
            optimizer=Adam(
                learning_rate=float(self.learning_rate),  # type: ignore
                beta_1=float(self.beta_1),  # type: ignore
                beta_2=float(self.beta_2),  # type: ignore
                epsilon=float(self.epsilon),  # type: ignore
            ),
            loss=keras_loss,
            metrics=[keras_loss],  # Add metrics to track during training
        )

    def _prepare_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for Multivariate LSTM.

        Args:
            X: Input features (2D array with shape (num_timesteps, num_targets))

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared sequences and targets
        """
        # Ensure X is numeric (float32) - Keras requires numeric dtypes
        # Convert to new array to avoid issues with read-only arrays or object dtypes
        X = np.array(X, dtype=np.float32, copy=True)

        # Ensure X is 2D: (num_timesteps, num_targets) for multivariate
        if X.ndim == 1:
            # If univariate input, reshape to (num_timesteps, 1)
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            # If 3D or higher, flatten to 2D
            X = X.reshape(X.shape[0], -1)

        # Check if we have enough data
        min_length = self.context_length + self.prediction_window  # type: ignore
        if len(X) < min_length:
            context_length_val = self.context_length  # type: ignore
            prediction_window_val = self.prediction_window  # type: ignore
            raise ValueError(
                f"Input data must have at least {min_length} timesteps, "
                f"but got {len(X)}. Need context_length ({context_length_val}) + "
                f"prediction_window ({prediction_window_val})"
            )

        X_seq, y_seq = [], []
        for i in range(len(X) - self.context_length - self.prediction_window + 1):  # type: ignore
            curr_X = X[i : (i + self.context_length)]  # type: ignore
            # Ensure curr_X maintains 2D shape (context_length, num_targets)
            if curr_X.ndim == 1:
                curr_X = curr_X.reshape(-1, 1)

            # Ensure curr_X is float32 before appending
            curr_X = np.array(curr_X, dtype=np.float32, copy=True)

            X_seq.append(curr_X)
            # y_seq: flatten to 1D array of length prediction_window * num_targets
            future_values = X[
                i
                + self.context_length : i  # type: ignore
                + self.context_length  # type: ignore
                + self.prediction_window  # type: ignore
            ]
            future_values = future_values.flatten()
            # Ensure future_values is float32 before appending
            future_values = np.array(future_values, dtype=np.float32, copy=True)
            y_seq.append(future_values)

        # Convert to numpy arrays and ensure correct shapes and numeric dtype
        # Keras requires numeric dtypes (float32), not string or object
        # First, stack the arrays properly to avoid object dtype issues
        try:
            X_seq = np.stack(X_seq, axis=0).astype(np.float32)
        except (ValueError, TypeError) as e:
            # If stacking fails, try to convert each element individually
            X_seq_clean = [np.array(x, dtype=np.float32) for x in X_seq]
            X_seq = np.stack(X_seq_clean, axis=0)

        try:
            y_seq = np.stack(y_seq, axis=0).astype(np.float32)
        except (ValueError, TypeError) as e:
            # If stacking fails, try to convert each element individually
            y_seq_clean = [np.array(y, dtype=np.float32) for y in y_seq]
            y_seq = np.stack(y_seq_clean, axis=0)

        # Ensure arrays are contiguous and float32
        X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)
        y_seq = np.ascontiguousarray(y_seq, dtype=np.float32)

        # Ensure X_seq is 3D: (num_sequences, context_length, num_targets)
        if X_seq.ndim != 3:
            raise ValueError(
                f"X_seq should be 3D (num_sequences, context_length, num_targets), "
                f"but got shape {X_seq.shape}"
            )

        return X_seq, y_seq
