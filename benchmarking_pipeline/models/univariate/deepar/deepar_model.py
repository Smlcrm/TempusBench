"""
DeepAR model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import lightning.pytorch as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from typing import Dict, Any, Union, Tuple, Optional, List
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger
import time
import math

# Using this link to assist in building this model file implementation:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html

# Useful for understanding how to work with TimeSeriesDataset
# https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html#passing-data
# time_idx is the index of the particular element in the time series
# group_ids or group is what time series the element belongs to. 
# If you only have one time series, you can set group to be 0.

class DeepARModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize LSTM model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - hidden_size: int, number of DeepAR units per recurrent layer
                - rnn_layers: int, number of DeepAR RNN layers
                - dropout: float, dropout rate in DeepAR RNN layers
                - learning_rate: float, learning rate for the DeepAR model
                - batch_size: int, batch size for training
                - feature_cols: list of str, names of feature columns
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
                - max_encoder_length: int, number of steps to use as input data during autoregressive training
                - max_prediction_length: int, number of steps to use as output data during autoregressive training
                - epochs: int, number of training epochs
                - gradient_clip_val: float, threshold to clip gradient to during training
                - num_workers: int, number of workers used for the dataloaders
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        if 'hidden_size' not in self.config:
            raise ValueError("hidden_size must be specified in config")
        if 'rnn_layers' not in self.config:
            raise ValueError("rnn_layers must be specified in config")
        if 'dropout' not in self.config:
            raise ValueError("dropout must be specified in config")
        if 'learning_rate' not in self.config:
            raise ValueError("learning_rate must be specified in config")
        if 'batch_size' not in self.config:
            raise ValueError("batch_size must be specified in config")
        if 'forecast_horizon' not in self.config:
            raise ValueError("forecast_horizon must be specified in config")
        if 'max_encoder_length' not in self.config:
            raise ValueError("max_encoder_length must be specified in config")
        
        self.hidden_size = self.config['hidden_size']
        self.rnn_layers = self.config['rnn_layers']
        self.dropout = self.config['dropout']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        if 'feature_cols' not in self.config:
            raise ValueError("feature_cols must be specified in config")
        self.feature_cols = self.config['feature_cols']
        self.forecast_horizon = self.config['forecast_horizon']
        self.max_encoder_length = self.config['max_encoder_length']
        
        if 'max_prediction_length' not in self.config:
            raise ValueError("max_prediction_length must be specified in config")
        if 'epochs' not in self.config:
            raise ValueError("epochs must be specified in config")
        if 'gradient_clip_val' not in self.config:
            raise ValueError("gradient_clip_val must be specified in config")
        if 'num_workers' not in self.config:
            raise ValueError("num_workers must be specified in config")
        
        self.max_prediction_length = self.config['max_prediction_length']
        self.epochs = self.config['epochs']
        self.gradient_clip_val = self.config['gradient_clip_val']
        self.num_workers = self.config['num_workers']
        self.model = None
    
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
        self.model = DeepAR.from_dataset(training_dataset,
                                           learning_rate=self.learning_rate,
                                           hidden_size=self.hidden_size,
                                           rnn_layers=self.rnn_layers,
                                           dropout=self.dropout)
    
    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              y_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: Optional[str] = None, 
              **kwargs
    ) -> 'DeepARModel':
        training_dataset = self._series_to_TimeSeriesDataset(y_context)
        validation_dataset = self._series_to_TimeSeriesDataset(y_target)

        if self.model is None:
            self._build_model(training_dataset)
        print("Model built!")

        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, batch_sampler="synchronized",
            num_workers=self.num_workers, persistent_workers=True
        )
        print("Train Dataloader Finished")

        #validation_dataloader = validation_dataset.to_dataloader(
        #    train=False, batch_size=self.batch_size, batch_sampler="synchronized",
        #    num_workers=self.num_workers, persistent_workers=True
        #)
        # TensorBoard logging is handled by the main benchmark runner
        # Create the PyTorch Lightning trainer without separate logger
        trainer = pl.Trainer(logger=False, accelerator="auto", gradient_clip_val=self.gradient_clip_val, max_epochs=self.epochs)
        print("Trainer initialized")
        #trainer.fit(self.model,train_dataloader,validation_dataloader)
        trainer.fit(self.model,train_dataloader)
        print("Trainer fitted")
        return self
        
        
    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        forecast_horizon: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Make autoregressive predictions using the trained model.
        
        Args:
            y_context: Recent/past target values 
            y_target: Future target values (used to determine forecast length)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon,)
        """
        if self.model is None:
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
            print("DEBUG: Current Encoder Sequence", current_encoder_sequence)

            # Convert to form compatible with data loader
            current_encoder_sequence_TimeSeriesDataset = self._series_to_TimeSeriesDataset(np.array(current_encoder_sequence), train=False)
            print("Time Series Dataset Conversion Finished")

            # Create dataloader - dataloaders are needed to predict with Pytorch Lightning models
            current_encoder_sequence_dataloader = current_encoder_sequence_TimeSeriesDataset.to_dataloader(
                train=False, batch_size=1, batch_sampler="synchronized",
                num_workers=self.num_workers, persistent_workers=True
            )
            print("Time Series Dataset Dataloader Finished")

            # Get the prediction for the current encoder sequence input
            current_predictions = self.model.predict(current_encoder_sequence_dataloader).cpu().numpy()
            print(f"Current predictions: {current_predictions}")
            print(f"Window {window} out of {num_windows}")
            # Append model predictions all_predictions, to prep for future forecasting
            all_predictions.extend(current_predictions[0])
        
        return np.array(all_predictions[self.max_prediction_length:self.max_prediction_length+val_length])
        
        
      
    def set_params(self, **params: Dict[str, Any]) -> 'BaseModel':
        """
        Set model parameters. This will rebuild the sktime model instance.
        """
        model_params_changed = False
        for key, value in params.items():
            if hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['learning_rate', 'hidden_size', 'rnn_layers', 'dropout'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed:
            self.model = None
            
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")
        return({
        "hidden_size": self.hidden_size, 
        "rnn_layers" : self.rnn_layers,
        "dropout" : self.dropout,
        "learning_rate" : self.learning_rate,
        "batch_size" : self.batch_size,
        "feature_cols" : self.feature_cols,
        "forecast_horizon" : self.forecast_horizon,
        "max_encoder_length" : self.max_encoder_length,
        "max_prediction_length" : self.max_prediction_length,
        "epochs" : self.epochs,
        "gradient_clip_val" : self.gradient_clip_val,
        "num_workers" : self.num_workers
        })
    

    def _evenly_split_array(self, array: np.ndarray, batch_size: int) -> List[np.ndarray]:
        assert len(array) >= batch_size
        return np.array_split(array, self.batch_size)