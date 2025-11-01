"""
DeepAR model implementation for stochastic time series forecasting.

This module provides a DeepAR model implementation that inherits from BaseModel
and returns probabilistic forecasts through sampling.
"""

import math
import os
import pickle
import time

from typing import Any, Dict, List, Literal, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_lightning.loggers import TensorBoardLogger

from tempus_bench.models.base_model import BaseModel, validate_inputs


class DeeparHyperparams(PydanticBaseModel):
    hidden_size: Optional[int] = Field(
        default=32, ge=1, description="Hidden size of RNN"
    )
    rnn_layers: Optional[int] = Field(
        default=2, ge=1, description="Number of RNN layers"
    )
    dropout: Optional[float] = Field(
        default=0.1, ge=0, le=1, description="Dropout rate"
    )
    learning_rate: Optional[float] = Field(
        default=0.001, gt=0, description="Learning rate"
    )
    batch_size: Optional[int] = Field(default=64, ge=1, description="Batch size")
    max_encoder_length: Optional[int] = Field(
        default=24, ge=1, description="Maximum encoder length"
    )
    max_prediction_length: Optional[int] = Field(
        default=6, ge=1, description="Maximum prediction length"
    )
    epochs: Optional[int] = Field(
        default=10, ge=1, description="Number of training epochs"
    )
    gradient_clip_val: Optional[float] = Field(
        default=0.01, gt=0, description="Gradient clipping value"
    )
    optimizer: Optional[Literal["adam", "adamw", "sgd"]] = Field(
        default="adam", description="Optimizer to use"
    )


class DeeparModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize DeepAR model with given configuration.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, DeeparHyperparams)

        self._model = None

    def _evenly_split_array(self, arr, chunk_size):
        """
        Split an array into chunks of approximately equal size.

        Args:
            arr: Array-like object to split
            chunk_size: Target size for each chunk

        Returns:
            List of arrays, each of size approximately chunk_size
        """
        arr_len = len(arr)
        if arr_len == 0:
            return []

        # Calculate number of chunks
        num_chunks = max(1, math.ceil(arr_len / chunk_size))

        # Calculate actual chunk size (might be slightly smaller for the last chunk)
        actual_chunk_size = math.ceil(arr_len / num_chunks)

        chunks = []
        for i in range(0, arr_len, actual_chunk_size):
            chunk = arr[i : i + actual_chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)

        return chunks

    def _series_to_TimeSeriesDataset(self, series, train=True):

        values = None
        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series

        # Convert to numpy array if needed
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # Handle different input shapes
        if values.ndim == 1:
            # Univariate: reshape to (num_steps, 1) for consistency
            values = values[:, np.newaxis]
        elif values.ndim > 2:
            raise ValueError(f"Series must be 1D or 2D, got {values.ndim}D")

        num_steps, num_features = values.shape

        # Reshape multivariate data to long format for pytorch-forecasting
        # Each feature becomes a separate time series identified by group_id
        data_rows = []

        if train:
            # For training, split data into chunks for faster training
            # Process each feature separately and create chunks
            for feature_idx in range(num_features):
                feature_values = values[:, feature_idx]
                list_of_sub_chunks = self._evenly_split_array(
                    feature_values, self.batch_size
                )

                sub_chunk_idx = 0
                for sub_chunk in list_of_sub_chunks:
                    for time_idx, value in enumerate(sub_chunk):
                        data_rows.append(
                            {
                                "value": value,
                                "time_idx": time_idx,
                                "group_id": f"feature_{feature_idx}_chunk_{sub_chunk_idx}",
                            }
                        )
                    sub_chunk_idx += 1
        else:
            # For prediction, don't split into chunks - use single group per feature
            for feature_idx in range(num_features):
                feature_values = values[:, feature_idx]
                for time_idx, value in enumerate(feature_values):
                    data_rows.append(
                        {
                            "value": value,
                            "time_idx": time_idx,
                            "group_id": f"feature_{feature_idx}",
                        }
                    )

        dataset_altered_form = pd.DataFrame(data_rows)

        dataset = TimeSeriesDataSet(
            dataset_altered_form,
            time_idx="time_idx",
            target="value",
            group_ids=["group_id"],
            time_varying_unknown_reals=["value"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["group_id"],
        )

        return dataset

    def _build_model(self, training_dataset):
        self._model = DeepAR.from_dataset(
            training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
        )

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "DeepARModel":
        """
        Train the DeepAR model on given data.

        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values - used for training
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]

        # Reference params, settings, device, python_version
        hidden_size = self.hidden_size
        rnn_layers = self.rnn_layers
        dropout = self.dropout
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        max_encoder_length = self.max_encoder_length
        max_prediction_length = self.max_prediction_length
        num_workers = self.num_workers

        training_dataset = self._series_to_TimeSeriesDataset(y_context)
        # Skip validation_dataset creation as it's not used and y_target might be too short
        # validation_dataset = self._series_to_TimeSeriesDataset(y_target)

        if self._model is None:
            self._build_model(training_dataset)

        train_dataloader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            batch_sampler="synchronized",
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        # validation_dataloader = validation_dataset.to_dataloader(
        #    train=False, batch_size=self.batch_size, batch_sampler="synchronized",
        #    num_workers=self.num_workers, persistent_workers=True
        # )
        # TensorBoard logging is handled by the main benchmark runner
        # Create the PyTorch Lightning trainer without separate logger
        trainer = pl.Trainer(
            logger=False,
            accelerator="auto",
            gradient_clip_val=self.gradient_clip_val,
            max_epochs=self.epochs,
        )
        # trainer.fit(self._model,train_dataloader,validation_dataloader)
        trainer.fit(self._model, train_dataloader)
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make autoregressive predictions using the trained model.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (num_samples, forecast_horizon, num_targets)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        num_samples = kwargs["num_samples"]

        # Reference params, settings, device, python_version
        hidden_size = self.hidden_size
        rnn_layers = self.rnn_layers
        dropout = self.dropout
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        max_encoder_length = self.max_encoder_length
        max_prediction_length = self.max_prediction_length
        num_workers = self.get("num_workers", 0)

        # Use y_target to determine forecast length if provided
        y_target = kwargs.get("y_target", None)
        forecast_horizon = kwargs.get("forecast_horizon", None)

        if self._model is None:
            raise ValueError("Model not initialized. Call train first.")
        # Fix this so we

        # train_dataset = self._series_to_TimeSeriesDataset(y_context, train=False)
        # train_dataloader = train_dataset.to_dataloader(
        #    train=False, batch_size=1, batch_sampler="synchronized",
        #    num_workers=self.num_workers, persistent_workers=True
        # )

        # Fix this code so we do sliding window inference on previously made predictions.
        all_predictions = []

        values = None
        if isinstance(y_context, pd.Series):
            values = y_context.values
        else:
            values = y_context

        # We need at least self.max_encoder_length+self.max_prediction_length values to get enough data to predict
        # So we get that amount of values by sampling the end of y_context
        all_predictions.extend(
            values[-(self.max_encoder_length + self.max_prediction_length) :]
        )

        val_length = len(y_target)

        num_windows = math.ceil((val_length) / self.max_prediction_length)
        for window in range(num_windows):

            # Get enough input to formulate next prediction

            current_encoder_sequence = all_predictions[
                -(self.max_encoder_length + self.max_prediction_length) :
            ]

            # Convert to form compatible with data loader
            current_encoder_sequence_TimeSeriesDataset = (
                self._series_to_TimeSeriesDataset(
                    np.array(current_encoder_sequence), train=False
                )
            )

            # Create dataloader - dataloaders are needed to predict with Pytorch Lightning models
            current_encoder_sequence_dataloader = (
                current_encoder_sequence_TimeSeriesDataset.to_dataloader(
                    train=False,
                    batch_size=1,
                    batch_sampler="synchronized",
                    num_workers=self.num_workers,
                    persistent_workers=True if self.num_workers > 0 else False,
                )
            )

            # Get the prediction for the current encoder sequence input
            current_predictions = (
                self._model.predict(current_encoder_sequence_dataloader).cpu().numpy()
            )
            # Append model predictions all_predictions, to prep for future forecasting
            all_predictions.extend(current_predictions[0])

        return np.array(
            all_predictions[
                self.max_prediction_length : self.max_prediction_length + val_length
            ]
        )

    @validate_inputs
    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make autoregressive predictions using the trained model.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            freq: Frequency string (e.g., 'H', 'D', 'M') - MUST be provided from CSV data

        Returns:
            np.ndarray: Model prediction samples with shape (num_samples, forecast_horizon, num_targets)
        """
        if self._model is None:
            raise ValueError("Model not initialized. Call train first.")

        # Extract kwargs (NO defaults, use kwargs["var_name"])
        num_samples = kwargs["num_samples"]

        forecast_horizon = timestamps_target.shape[0]

        # Ensure y_context is 2D (num_steps, num_features)
        if y_context.ndim == 1:
            y_context = y_context[:, np.newaxis]
        elif y_context.ndim > 2:
            raise ValueError(f"y_context must be 1D or 2D, got {y_context.ndim}D")

        num_steps, num_targets = y_context.shape

        # For multivariate: predict each feature separately and aggregate
        # Store predictions for each feature
        all_feature_samples = []

        for feature_idx in range(num_targets):
            feature_context = y_context[:, feature_idx]  # Shape: (num_steps,)

            # Get deterministic predictions for this feature
            all_predictions = []
            # We need at least max_encoder_length+max_prediction_length values
            min_required = self.max_encoder_length + self.max_prediction_length
            if len(feature_context) < min_required:
                # Pad with first value if needed
                padding = np.full(
                    min_required - len(feature_context), feature_context[0]
                )
                feature_context = np.concatenate([padding, feature_context])

            all_predictions.extend(
                feature_context[
                    -(self.max_encoder_length + self.max_prediction_length) :
                ]
            )

            val_length = forecast_horizon
            num_windows = math.ceil(val_length / self.max_prediction_length)

            deterministic_preds = []
            for window in range(num_windows):
                current_encoder_sequence = all_predictions[
                    -(self.max_encoder_length + self.max_prediction_length) :
                ]

                # Convert to numpy array
                current_encoder_sequence = np.array(current_encoder_sequence)

                # Convert to form compatible with data loader
                # Reshape to (num_steps, 1) for multivariate handling
                current_encoder_sequence_reshaped = current_encoder_sequence[
                    :, np.newaxis
                ]

                current_encoder_sequence_TimeSeriesDataset = (
                    self._series_to_TimeSeriesDataset(
                        current_encoder_sequence_reshaped, train=False
                    )
                )

                # Create dataloader
                current_encoder_sequence_dataloader = (
                    current_encoder_sequence_TimeSeriesDataset.to_dataloader(
                        train=False,
                        batch_size=1,
                        batch_sampler="synchronized",
                        num_workers=self.num_workers,
                        persistent_workers=True if self.num_workers > 0 else False,
                    )
                )

                # Get the prediction for the current encoder sequence input
                # The model predicts for all groups, we need to extract predictions for our feature
                current_predictions = (
                    self._model.predict(current_encoder_sequence_dataloader)
                    .cpu()
                    .numpy()
                )

                # Extract predictions for this feature (first group)
                # The predictions come back as an array, we take the first element
                if current_predictions.ndim > 1:
                    pred_values = current_predictions[0]  # Take first sample/batch
                    if pred_values.ndim > 1:
                        pred_values = pred_values.flatten()
                    deterministic_preds.extend(pred_values)
                    all_predictions.extend(pred_values)
                else:
                    deterministic_preds.extend([current_predictions])
                    all_predictions.extend([current_predictions])

            # Extract the relevant portion
            deterministic_preds_array = np.array(deterministic_preds)
            if deterministic_preds_array.ndim > 1:
                deterministic_preds_array = deterministic_preds_array.flatten()

            # Trim to forecast_horizon
            if len(deterministic_preds_array) > forecast_horizon:
                deterministic_preds_array = deterministic_preds_array[:forecast_horizon]
            elif len(deterministic_preds_array) < forecast_horizon:
                # Pad with last value if needed
                padding = np.full(
                    forecast_horizon - len(deterministic_preds_array),
                    (
                        deterministic_preds_array[-1]
                        if len(deterministic_preds_array) > 0
                        else 0.0
                    ),
                )
                deterministic_preds_array = np.concatenate(
                    [deterministic_preds_array, padding]
                )

            # Generate samples by adding noise to deterministic predictions
            feature_samples = []
            for _ in range(num_samples):
                # Add Gaussian noise to deterministic predictions
                noise_std = (
                    np.std(deterministic_preds_array) * 0.1
                )  # 10% of std as noise
                sample = deterministic_preds_array + np.random.normal(
                    0, noise_std, deterministic_preds_array.shape
                )
                feature_samples.append(sample)

            feature_samples = np.array(
                feature_samples
            )  # Shape: (num_samples, forecast_horizon)
            all_feature_samples.append(feature_samples)

        # Stack all features: (num_samples, forecast_horizon, num_targets)
        samples = np.stack(
            all_feature_samples, axis=2
        )  # Shape: (num_samples, forecast_horizon, num_targets)

        return samples
