"""
DeepAR model implementation for stochastic time series forecasting.

This module provides a DeepAR model implementation that inherits from BaseModel
and returns probabilistic forecasts through sampling.
"""

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from typing import Dict, Any, Union, Tuple, Optional, List
import pickle
import os
import time
import math
from pytorch_lightning.loggers import TensorBoardLogger
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Literal
from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel


class DeeparHyperparams(PydanticBaseModel):
    hidden_size: Optional[int] = Field(default=32, ge=1, description="Hidden size of RNN")
    rnn_layers: Optional[int] = Field(default=2, ge=1, description="Number of RNN layers")
    dropout: Optional[float] = Field(default=0.1, ge=0, le=1, description="Dropout rate")
    learning_rate: Optional[float] = Field(default=0.001, gt=0, description="Learning rate")
    batch_size: Optional[int] = Field(default=64, ge=1, description="Batch size")
    max_encoder_length: Optional[int] = Field(default=24, ge=1, description="Maximum encoder length")
    max_prediction_length: Optional[int] = Field(default=6, ge=1, description="Maximum prediction length")
    epochs: Optional[int] = Field(default=10, ge=1, description="Number of training epochs")
    gradient_clip_val: Optional[float] = Field(default=0.01, gt=0, description="Gradient clipping value")
    optimizer: Optional[Literal["adam", "adamw", "sgd"]] = Field(default="adam", description="Optimizer to use")

# Using this link to assist in building this model file implementation:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html

# Useful for understanding how to work with TimeSeriesDataset
# https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html#passing-data
# time_idx is the index of the particular element in the time series
# group_ids or group is what time series the element belongs to.
# If you only have one time series, you can set group to be 0.

class DeepARModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize DeepAR model with given configuration.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, DeeparHyperparams)

        self._model = None

    def _series_to_TimeSeriesDataset(self, series, train=True):

        values = None
        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series


        if train:
            # Increase speed of training
            list_of_sub_chunks = self._evenly_split_array(values, self.batch_size)
            #for sub_chunk in list_of_sub_chunks:


            # Each array, besides the last one, has to have the same number of elements
            list_of_ids = []
            sub_chunk_idx = 0
            for sub_chunk in list_of_sub_chunks:
                current_number_of_ids = len(sub_chunk)
                current_id_list = [str(sub_chunk_idx)] * current_number_of_ids
                list_of_ids.extend(current_id_list)
                sub_chunk_idx += 1


            dataset_altered_form = pd.DataFrame({
                "value": values,
                "time_idx": np.concatenate([np.arange(len(sub_chunk)) for sub_chunk in list_of_sub_chunks]),
                "group_id": list_of_ids
            })
        else:
            dataset_altered_form = pd.DataFrame({
                "value": values,
                "time_idx": list(range(len(series))),
                "group_id": ["0"] * len(series)
            })


        dataset = TimeSeriesDataSet(
            dataset_altered_form,
            time_idx="time_idx",
            target="value",
            group_ids=["group_id"],
            time_varying_unknown_reals=["value"],
            max_encoder_length = self.max_encoder_length,
            max_prediction_length = self.max_prediction_length,
            static_categoricals=["group_id"]
        )

        return dataset

    def _build_model(self, training_dataset):
        self._model = DeepAR.from_dataset(training_dataset,
                                           learning_rate=self.learning_rate,
                                           hidden_size=self.hidden_size,
                                           rnn_layers=self.rnn_layers,
                                           dropout=self.dropout)

    def train(self,
              y_context: np.ndarray,
              y_target: np.ndarray,
              timestamps_context: np.ndarray,
              timestamps_target: np.ndarray,
              **kwargs,
    ) -> 'DeepARModel':
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
        num_workers = self.get("num_workers", 0)

        training_dataset = self._series_to_TimeSeriesDataset(y_context)
        validation_dataset = self._series_to_TimeSeriesDataset(y_target)

        if self._model is None:
            self._build_model(training_dataset)
        self.logger.info("DeepARModel.train", "Model built!")

        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, batch_sampler="synchronized",
            num_workers=self.num_workers, persistent_workers=True
        )
        self.logger.info("DeepARModel.train", "Train Dataloader Finished")

        #validation_dataloader = validation_dataset.to_dataloader(
        #    train=False, batch_size=self.batch_size, batch_sampler="synchronized",
        #    num_workers=self.num_workers, persistent_workers=True
        #)
        # TensorBoard logging is handled by the main benchmark runner
        # Create the PyTorch Lightning trainer without separate logger
        trainer = pl.Trainer(logger=False, accelerator="auto", gradient_clip_val=self.gradient_clip_val, max_epochs=self.epochs)
        self.logger.info("DeepARModel.train", "Trainer initialized")
        #trainer.fit(self._model,train_dataloader,validation_dataloader)
        trainer.fit(self._model,train_dataloader)
        self.logger.info("DeepARModel.train", "Trainer fitted")
        return self


    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        **kwargs
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

        #train_dataset = self._series_to_TimeSeriesDataset(y_context, train=False)
        #train_dataloader = train_dataset.to_dataloader(
        #    train=False, batch_size=1, batch_sampler="synchronized",
        #    num_workers=self.num_workers, persistent_workers=True
        #)

        # Fix this code so we do sliding window inference on previously made predictions.
        all_predictions = []

        values = None
        if isinstance(y_context, pd.Series):
            values = y_context.values
        else:
            values = y_context

        # We need at least self.max_encoder_length+self.max_prediction_length values to get enough data to predict
        # So we get that amount of values by sampling the end of y_context
        all_predictions.extend(values[-(self.max_encoder_length+self.max_prediction_length):])

        val_length = len(y_target)

        num_windows = math.ceil((val_length) / self.max_prediction_length)
        for window in range(num_windows):

            # Get enough input to formulate next prediction

            current_encoder_sequence = all_predictions[-(self.max_encoder_length+self.max_prediction_length):]
            self.logger.debug("DeepARModel.predict", "DEBUG: Current Encoder Sequence")

            # Convert to form compatible with data loader
            current_encoder_sequence_TimeSeriesDataset = self._series_to_TimeSeriesDataset(np.array(current_encoder_sequence), train=False)
            self.logger.debug("DeepARModel.predict", "Time Series Dataset Conversion Finished")

            # Create dataloader - dataloaders are needed to predict with Pytorch Lightning models
            current_encoder_sequence_dataloader = current_encoder_sequence_TimeSeriesDataset.to_dataloader(
                train=False, batch_size=1, batch_sampler="synchronized",
                num_workers=self.num_workers, persistent_workers=True
            )
            self.logger.debug("DeepARModel.predict", "Time Series Dataset Dataloader Finished")

            # Get the prediction for the current encoder sequence input
            current_predictions = self._model.predict(current_encoder_sequence_dataloader).cpu().numpy()
            self.logger.debug("DeepARModel.predict", f"Current predictions: {current_predictions}")
            self.logger.debug("DeepARModel.predict", f"Window {window} out of {num_windows}")
            # Append model predictions all_predictions, to prep for future forecasting
            all_predictions.extend(current_predictions[0])

        return np.array(all_predictions[self.max_prediction_length:self.max_prediction_length+val_length])

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
        **kwargs
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

        forecast_horizon = timestamps_target.shape[0]

        # For DeepAR, we need to generate samples through multiple forward passes
        # This is a simplified implementation - in practice, you'd want to use the model's
        # built-in sampling capabilities

        # Get deterministic predictions first
        all_predictions = []
        values = y_context.flatten() if y_context.ndim > 1 else y_context

        # We need at least max_encoder_length+max_prediction_length values
        all_predictions.extend(values[-(self._model_config["max_encoder_length"]+self._model_config["max_prediction_length"]):])

        val_length = forecast_horizon
        num_windows = math.ceil(val_length / self._model_config["max_prediction_length"])

        deterministic_preds = []
        for window in range(num_windows):
            current_encoder_sequence = all_predictions[-(self._model_config["max_encoder_length"]+self._model_config["max_prediction_length"]):]

            # Convert to form compatible with data loader
            current_encoder_sequence_TimeSeriesDataset = self._series_to_TimeSeriesDataset(np.array(current_encoder_sequence), train=False)

            # Create dataloader
            current_encoder_sequence_dataloader = current_encoder_sequence_TimeSeriesDataset.to_dataloader(
                train=False, batch_size=1, batch_sampler="synchronized",
                num_workers=self._model_config["num_workers"], persistent_workers=True
            )

            # Get the prediction for the current encoder sequence input
            current_predictions = self._model.predict(current_encoder_sequence_dataloader).cpu().numpy()
            deterministic_preds.extend(current_predictions[0])
            all_predictions.extend(current_predictions[0])

        deterministic_preds = np.array(deterministic_preds[self._model_config["max_prediction_length"]:self._model_config["max_prediction_length"]+val_length])

        # Generate samples by adding noise to deterministic predictions
        # This is a simplified approach - real DeepAR would use the model's probabilistic output
        num_samples = self.num_samples
        samples = []

        for _ in range(num_samples):
            # Add Gaussian noise to deterministic predictions
            noise_std = np.std(deterministic_preds) * 0.1  # 10% of std as noise
            sample = deterministic_preds + np.random.normal(0, noise_std, deterministic_preds.shape)
            samples.append(sample)

        samples = np.array(samples)  # Shape: (num_samples, forecast_horizon)

        # Ensure correct shape: (num_samples, forecast_horizon, num_targets)
        if samples.ndim == 2:
            samples = samples[:, :, np.newaxis]

        return samples