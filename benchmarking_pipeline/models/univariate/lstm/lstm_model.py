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
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
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
        super().__init__(config, config_file)
        if 'units' not in self.config:
            raise ValueError("units must be specified in config")
        if 'layers' not in self.config:
            raise ValueError("layers must be specified in config")
        if 'dropout' not in self.config:
            raise ValueError("dropout must be specified in config")
        if 'learning_rate' not in self.config:
            raise ValueError("learning_rate must be specified in config")
        if 'batch_size' not in self.config:
            raise ValueError("batch_size must be specified in config")
        if 'epochs' not in self.config:
            raise ValueError("epochs must be specified in config")
        if 'sequence_length' not in self.config:
            raise ValueError("sequence_length must be specified in config")
        if 'training_loss' not in self.config:
            raise ValueError("training_loss must be specified in config")
        
        self.units = self.config['units']
        self.layers = self.config['layers']
        self.dropout = self.config['dropout']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.sequence_length = self.config['sequence_length']
        if 'feature_cols' not in self.config:
            raise ValueError("feature_cols must be specified in config")
        self.feature_cols = self.config['feature_cols']
        self.training_loss = self.config['training_loss']
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        self.model = Sequential()
        
        # Add LSTM layers
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            self.model.add(LSTM(
                units=self.units,
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            ))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))
                
        # Add output layer
        self.model.add(Dense(self.forecast_horizon))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.training_loss
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
        for i in range(len(X) - self.sequence_length - self.forecast_horizon + 1):
            X_seq.append(X[i:(i + self.sequence_length)])
            # y_seq: flatten to 1D if forecast_horizon == 1, else keep as 1D array
            y_seq.append(X[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon, 0])
        return np.array(X_seq), np.array(y_seq)
        
    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, y_start_date: pd.Timestamp = None, **kwargs) -> 'LstmModel':
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
        # Convert input to numpy array
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            target = y_context.values
        else:
            target = y_context
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(target)
        
        # Build model if not already built
        if self.model is None:
            self._build_model(input_shape=(self.sequence_length, 1))
        # TensorBoard logging is handled by the main benchmark runner
        # Train model
        self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0
        )
        
        self.is_fitted = True
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> np.ndarray:
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
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")

        # Convert input to numpy array
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            input_data = y_context.values
        else:
            input_data = y_context
        forecast_steps = len(y_target)

        # Ensure input_data is 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        # Calculate how many prediction windows we need
        num_windows = math.ceil((forecast_steps) // self.forecast_horizon)
        
        all_predictions = []
        current_sequence = input_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        for window in range(num_windows):
            # Predict forecast_horizon steps at once
            predictions = self.model.predict(current_sequence, verbose=0)
            all_predictions.extend(predictions[0])
            # Advance window by forecast_horizon steps and use predictions as input
            if window < num_windows - 1:  # Don't update on last iteration
                # Move window forward by forecast_horizon steps
                current_sequence = np.roll(current_sequence, -self.forecast_horizon, axis=1)
                # Only update as many as fit
                n = min(self.forecast_horizon, self.sequence_length)
                current_sequence[0, -n:, 0] = predictions[0][:n]
        
        # Return only the requested number of predictions
        return np.array(all_predictions[:forecast_steps]).reshape(1, -1)
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'sequence_length': self.sequence_length,
            'feature_cols': self.feature_cols,
            'loss_function': self.training_loss,
            'forecast_horizon': self.forecast_horizon
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'LSTMModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if key == 'dataset':
                # Skip dataset parameter - it's not an LSTM model parameter
                continue
            elif hasattr(self, key):
                setattr(self, key, value)
        return self
        
    def save(self, path: str) -> None:
        """
        Save the LSTM model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params()
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
        # Save TensorFlow model separately
        model_path = path.replace('.pkl', '_tf')
        self.model.save(model_path)
        
    def load(self, path: str) -> None:
        """
        Load the LSTM model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        # Load model state from file
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.set_params(**model_state['params'])
        
        # Load TensorFlow model
        model_path = path.replace('.pkl', '_tf')
        self.model = tf.keras.models.load_model(model_path) 