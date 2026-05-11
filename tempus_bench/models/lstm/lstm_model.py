"""
Multivariate LSTM model implementation.

This model extends the univariate LSTM to handle multiple target variables simultaneously.
Design choices (all Option A):
- Single output layer that predicts forecast_horizon * n_targets values (flattened)
- Concatenate all targets into a single input sequence (context_length, n_targets)
- Optional past and future covariates are concatenated on the feature axis; only targets are predicted.
- Sum/average losses across all targets
- Predict all targets simultaneously in one forward pass
- Keep same LSTM architecture, just change input/output dimensions

context_length and prediction_window are clamped each fit so that rolling-window training
works when task context_window + forecast_horizon is smaller than the hyperparameter defaults.
"""

import math

from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from pydantic import BaseModel as PydanticBaseModel, Field
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from tempus_bench.models.base_model import (
    BaseModel,
    validate_covariate_support,
    validate_inputs,
)


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


def _mape_denominator_floor_epsilon() -> float:
    """Floor on |y_true| inside MAPE so sparse / normalized series do not explode."""
    return 1e-6


class LstmModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, LstmHyperparams)  # type: ignore
        self._runtime_n_y: int = 0
        self._runtime_n_cov: int = 0
        self._runtime_context_length: int = 0
        self._runtime_prediction_window: int = 0

    @staticmethod
    def _resolve_runtime_lengths(
        *,
        n_ctx: int,
        n_tgt: int,
        requested_ctx: int,
        requested_pred: int,
    ) -> Tuple[int, int]:
        """
        Choose context and label horizon so at least one sliding window fits in n_ctx + n_tgt,
        while never using more past rows than n_ctx (matches inference: history is context slice only).
        """
        combined_len = int(n_ctx) + int(n_tgt)
        if combined_len < 2:
            raise ValueError(
                "LSTM requires at least 2 total timesteps in y_context ∪ y_target "
                f"(got n_ctx={n_ctx}, n_tgt={n_tgt})."
            )
        req_ctx = int(requested_ctx)
        req_pred = int(requested_pred)
        if req_ctx < 1 or req_pred < 1:
            raise ValueError(
                f"LSTM context_length and prediction_window must be >= 1 "
                f"(got {req_ctx}, {req_pred})."
            )

        # Past rows available in-rolling-window training are only the context slice (n_ctx);
        # labels may draw from the full concatenated series (context + target segment).
        ctx_eff = min(req_ctx, n_ctx, combined_len - 1)
        pred_eff = min(req_pred, combined_len - ctx_eff)

        if ctx_eff < 1 or pred_eff < 1:
            raise ValueError(
                "LSTM cannot form training sequences for this window: "
                f"n_ctx={n_ctx}, n_tgt={n_tgt}, requested context_length={req_ctx}, "
                f"prediction_window={req_pred}."
            )
        return ctx_eff, pred_eff

    def _resolve_compile_loss(self, tuning_loss: str):
        t = tuning_loss.lower()
        if t == "mape":

            def safe_mape(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                eps = _mape_denominator_floor_epsilon()
                denom = tf.maximum(tf.abs(y_true), eps)
                return tf.reduce_mean(tf.abs(y_true - y_pred) / denom)

            return safe_mape

        loss_map = {
            "mae": "mean_absolute_error",
            "mase": "mean_absolute_error",
            "mape": "mean_absolute_percentage_error",
            "rmse": "root_mean_squared_error",
            "mse": "mean_squared_error",
        }
        name = loss_map.get(t, t)
        return name

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
            x_context / x_target: Optional covariates aligned with y_context / y_target
            **kwargs: Must include tuning_loss

        Returns:
            self: The fitted model instance
        """
        has_xc = x_context is not None
        has_xt = x_target is not None
        if has_xc ^ has_xt:
            raise ValueError(
                "LstmModel requires both x_context and x_target when using covariates."
            )
        if has_xc and has_xt:
            validate_covariate_support(
                x_context,
                x_target,
                supports_past_only=False,
                supports_future_only=False,
                supports_both=True,
                model_name="LstmModel",
            )

        n_y = int(y_context.shape[1])

        # Ensure input data is numeric (float32) - Keras requires numeric dtypes
        y_context = np.asarray(y_context, dtype=np.float32, copy=True)
        y_target = np.asarray(y_target, dtype=np.float32, copy=True)

        if has_xc and has_xt:
            xc = np.asarray(x_context, dtype=np.float32, copy=True)
            xt = np.asarray(x_target, dtype=np.float32, copy=True)
            if xc.shape[0] != y_context.shape[0]:
                raise ValueError(
                    f"x_context rows ({xc.shape[0]}) must match y_context ({y_context.shape[0]})."
                )
            if xt.shape[0] != y_target.shape[0]:
                raise ValueError(
                    f"x_target rows ({xt.shape[0]}) must match y_target ({y_target.shape[0]})."
                )
            if xc.shape[1] != xt.shape[1]:
                raise ValueError(
                    f"x_context and x_target must have same num covariates "
                    f"({xc.shape[1]} vs {xt.shape[1]})."
                )
            z_context = np.concatenate([xc, y_context], axis=1)
            z_target = np.concatenate([xt, y_target], axis=1)
            n_cov = int(xc.shape[1])
        else:
            z_context = y_context
            z_target = y_target
            n_cov = 0

        n_ctx: int = int(y_context.shape[0])
        n_tgt: int = int(y_target.shape[0])
        ctx_eff, pred_eff = self._resolve_runtime_lengths(
            n_ctx=n_ctx,
            n_tgt=n_tgt,
            requested_ctx=int(self.context_length),  # type: ignore[arg-type]
            requested_pred=int(self.prediction_window),  # type: ignore[arg-type]
        )

        self._runtime_n_y = n_y
        self._runtime_n_cov = n_cov
        self._runtime_context_length = ctx_eff
        self._runtime_prediction_window = pred_eff

        combined_data = np.concatenate([z_context, z_target], axis=0)
        # Zero out the y columns in the target portion so the model cannot see future
        # target values in its input window during training.
        combined_data_X = combined_data.copy()
        combined_data_X[n_ctx:, n_cov:] = 0.0
        X_seq, y_seq = self._prepare_sequences(
            combined_data_X, combined_data, ctx_eff, pred_eff, n_y
        )

        if "tuning_loss" not in kwargs:
            raise ValueError(
                "tuning_loss must be provided in kwargs for LSTM model"
            )
        tuning_loss_kw = str(kwargs["tuning_loss"])
        compile_loss = self._resolve_compile_loss(tuning_loss_kw)

        if "_model" in self.__dict__:
            delattr(self, "_model")
        n_feats = int(combined_data.shape[1])
        self._build_model(
            input_shape=(ctx_eff, n_feats),
            prediction_window=pred_eff,
            n_y_targets=n_y,
            compile_loss=compile_loss,
        )

        early_stopping = EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True, verbose=1
        )
        validation_split = 0.1 if len(X_seq) > 10 else 0.0

        if X_seq.dtype.kind in ["O", "U", "S"]:
            X_seq = np.array(X_seq, dtype=np.float64, copy=True)
        X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)

        if y_seq.dtype.kind in ["O", "U", "S"]:
            y_seq = np.array(y_seq, dtype=np.float64, copy=True)
        y_seq = np.ascontiguousarray(y_seq, dtype=np.float32)

        if not np.issubdtype(X_seq.dtype, np.number) or not np.issubdtype(
            y_seq.dtype, np.number
        ):
            raise ValueError(
                f"Data must be numeric. Got X_seq dtype: {X_seq.dtype}, y_seq dtype: {y_seq.dtype}"
            )

        if X_seq.dtype != np.float32:
            raise ValueError(f"X_seq must be float32, got {X_seq.dtype}")
        if y_seq.dtype != np.float32:
            raise ValueError(f"y_seq must be float32, got {y_seq.dtype}")
        if not X_seq.flags["C_CONTIGUOUS"]:
            raise ValueError("X_seq must be contiguous")
        if not y_seq.flags["C_CONTIGUOUS"]:
            raise ValueError("y_seq must be contiguous")

        batch_fit = min(int(self.batch_size), max(1, len(X_seq)))  # type: ignore[arg-type]

        self._model.fit(  # type: ignore[union-attr]
            X_seq,
            y_seq,
            batch_size=batch_fit,
            epochs=int(self.epochs),  # type: ignore[arg-type]
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
        Does not use own predictions as further inputs for target channels (y); known future covariates are used when provided.

        Args:
            y_context: Context/history values (n_steps, n_targets)
            timestamps_context: Timestamps for context (unused)
            timestamps_target: Timestamps for target (unused)
            x_context: Covariates aligned with y_context (required if model was trained with covariates)
            x_target: Future covariates aligned with timestamps_target (required if trained with covariates)

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, n_targets)
        """
        if not self.is_fitted or "_model" not in self.__dict__:
            raise ValueError("Model not fitted. Call train() first.")

        n_y = self._runtime_n_y
        n_cov = self._runtime_n_cov
        context_length = self._runtime_context_length
        prediction_window = self._runtime_prediction_window
        if n_y < 1 or context_length < 1 or prediction_window < 1:
            raise ValueError(
                "LSTM runtime layout missing; train() must run before predict()."
            )

        y_context = np.asarray(y_context, dtype=np.float32, copy=True)
        forecast_length = len(timestamps_target)
        num_context_steps: int = int(y_context.shape[0])

        if n_cov > 0:
            if x_context is None or x_target is None:
                raise ValueError(
                    "Trained with covariates; predict requires x_context and x_target."
                )
            xc = np.asarray(x_context, dtype=np.float32, copy=True)
            xf = np.asarray(x_target, dtype=np.float32, copy=True)
            if xc.shape[0] != num_context_steps:
                raise ValueError(
                    f"x_context rows ({xc.shape[0]}) must match y_context ({num_context_steps})."
                )
            if xc.shape[1] != n_cov:
                raise ValueError(
                    f"x_context has {xc.shape[1]} columns but model expects {n_cov} covariates."
                )
            if xf.shape[0] != forecast_length:
                raise ValueError(
                    f"x_target rows ({xf.shape[0]}) must match forecast horizon ({forecast_length})."
                )
            if xf.shape[1] != n_cov:
                raise ValueError(
                    f"x_target has {xf.shape[1]} columns but model expects {n_cov} covariates."
                )
            z_context = np.concatenate([xc, y_context], axis=1)
        else:
            if x_context is not None or x_target is not None:
                raise ValueError(
                    "Model trained without covariates; omit x_context and x_target."
                )
            z_context = y_context

        if z_context.shape[1] != n_y + n_cov:
            raise ValueError(
                f"Feature width mismatch: z has {z_context.shape[1]} columns, "
                f"expected {n_y + n_cov} (n_y={n_y}, n_cov={n_cov})."
            )

        if num_context_steps < context_length:
            raise ValueError(
                f"y_context must have at least {context_length} timesteps, "
                f"but got {num_context_steps}"
            )

        num_windows = math.ceil(forecast_length / prediction_window)
        all_predictions: list[np.ndarray] = []
        current_sequence = z_context[-context_length:].reshape(
            1, context_length, n_y + n_cov
        )

        t_future = 0
        for window in range(num_windows):
            preds = self._model.predict(current_sequence, verbose=0)  # type: ignore[union-attr]
            preds_reshaped = preds[0].reshape(prediction_window, n_y)
            all_predictions.append(preds_reshaped)

            if window < num_windows - 1:
                n_roll = min(prediction_window, context_length)
                current_sequence = np.roll(current_sequence, -prediction_window, axis=1)
                for j in range(n_roll):
                    row_j = context_length - n_roll + j
                    current_sequence[0, row_j, n_cov:] = preds_reshaped[j, :]
                    if n_cov > 0:
                        current_sequence[0, row_j, :n_cov] = xf[t_future + j, :]
                t_future += n_roll

        stacked = np.concatenate(all_predictions, axis=0)
        result = np.asarray(stacked[:forecast_length], dtype=np.float32)

        if result.ndim == 1:
            result = np.expand_dims(result, axis=1)

        return result

    def _build_model(
        self,
        *,
        input_shape: Tuple[int, int],
        prediction_window: int,
        n_y_targets: int,
        compile_loss: Any,
    ) -> None:
        """
        Build the Multivariate LSTM model architecture.

        Args:
            input_shape: (context_length, num_input_features) where features are targets ∪ covariates
            prediction_window: Steps predicted per forward pass
            n_y_targets: Number of target variates (output channels only)
        """
        self._model = Sequential()

        for layer_idx in range(int(self.layers)):  # type: ignore[arg-type]
            return_sequences = layer_idx < int(self.layers) - 1  # type: ignore[arg-type]
            self._model.add(
                LSTM(
                    units=int(self.units),  # type: ignore[arg-type]
                    return_sequences=return_sequences,
                    input_shape=input_shape if layer_idx == 0 else None,
                ),
            )
            if float(self.dropout) > 0:  # type: ignore[arg-type]
                self._model.add(Dropout(float(self.dropout)))  # type: ignore[arg-type]

        self._model.add(
            Dense(int(prediction_window) * int(n_y_targets))
        )

        train_metrics: list[Any] = (
            [] if callable(compile_loss) else [compile_loss]
        )

        self._model.compile(
            optimizer=Adam(
                learning_rate=float(self.learning_rate),  # type: ignore[arg-type]
                beta_1=float(self.beta_1),  # type: ignore[arg-type]
                beta_2=float(self.beta_2),  # type: ignore[arg-type]
                epsilon=float(self.epsilon),  # type: ignore[arg-type]
            ),
            loss=compile_loss,
            metrics=train_metrics,
        )

    def _prepare_sequences(
        self,
        Z_X: np.ndarray,
        Z_y: np.ndarray,
        context_length: int,
        prediction_window: int,
        n_y_targets: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for Multivariate LSTM.

        Args:
            Z_X: Per-timestep features used for X inputs (future y zeroed out)
            Z_y: Per-timestep features used for y labels (actual values)
            context_length: LSTM time steps per input window
            prediction_window: Steps to predict ahead (labels = first n_y targets only)

        Returns:
            X_seq: (num_sequences, context_length, n_features)
            y_seq: (num_sequences, prediction_window * n_y_targets)
        """
        Z_X = np.asarray(Z_X, dtype=np.float32, copy=True)
        Z_y = np.asarray(Z_y, dtype=np.float32, copy=True)

        if Z_X.ndim == 1:
            Z_X = Z_X.reshape(-1, 1)
        elif Z_X.ndim > 2:
            Z_X = Z_X.reshape(Z_X.shape[0], -1)

        if Z_y.ndim == 1:
            Z_y = Z_y.reshape(-1, 1)
        elif Z_y.ndim > 2:
            Z_y = Z_y.reshape(Z_y.shape[0], -1)

        min_length = int(context_length) + int(prediction_window)
        if len(Z_X) < min_length:
            raise ValueError(
                f"Input data must have at least {min_length} timesteps, "
                f"but got {len(Z_X)}. Need context_length ({context_length}) + "
                f"prediction_window ({prediction_window})"
            )

        X_seq: list[np.ndarray] = []
        y_seq: list[np.ndarray] = []
        n_y = int(n_y_targets)
        for i in range(len(Z_X) - context_length - prediction_window + 1):
            curr_X = Z_X[i : (i + context_length)]
            if curr_X.ndim == 1:
                curr_X = curr_X.reshape(-1, 1)
            curr_X = np.asarray(curr_X, dtype=np.float32, copy=True)
            X_seq.append(curr_X)

            future_block = Z_y[
                i + context_length : i + context_length + prediction_window
            ]
            future_y = future_block[:, -n_y_targets:].reshape(-1)
            future_y = np.asarray(future_y, dtype=np.float32, copy=True)
            y_seq.append(future_y)

        try:
            X_arr = np.stack(X_seq, axis=0).astype(np.float32)
        except (ValueError, TypeError):
            X_clean = [np.array(x, dtype=np.float32) for x in X_seq]
            X_arr = np.stack(X_clean, axis=0)

        try:
            y_arr = np.stack(y_seq, axis=0).astype(np.float32)
        except (ValueError, TypeError):
            y_clean = [np.array(y, dtype=np.float32) for y in y_seq]
            y_arr = np.stack(y_clean, axis=0)

        X_arr = np.ascontiguousarray(X_arr, dtype=np.float32)
        y_arr = np.ascontiguousarray(y_arr, dtype=np.float32)

        if X_arr.ndim != 3:
            raise ValueError(
                f"X_seq should be 3D (num_sequences, context_length, num_features), "
                f"but got shape {X_arr.shape}"
            )

        return X_arr, y_arr