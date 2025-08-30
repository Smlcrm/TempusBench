"""
Multivariate LSTM model for time series forecasting.

Features:
- Single output layer that predicts forecast_horizon * num_targets values (flattened)
- Concatenate all targets into a single input sequence (sequence_length, num_targets)
- Handles both univariate and multivariate data automatically
- Supports variable sequence lengths and forecast horizons
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
    def __init__(self, config: Dict[str, Any):
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
                - sequence_length: int, length of input sequences
                - feature_cols: list of str, names of feature columns
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config)
        if 'units' not in self.model_config:
            raise ValueError("units must be specified in config")
        if 'layers' not in self.model_config:
            raise ValueError("layers must be specified in config")
        if 'dropout' not in self.model_config:
            raise ValueError("dropout must be specified in config")
        if 'learning_rate' not in self.model_config:
            raise ValueError("learning_rate must be specified in config")
        if 'batch_size' not in self.model_config:
            raise ValueError("batch_size must be specified in config")
        if 'epochs' not in self.model_config:
            raise ValueError("epochs must be specified in config")
        if 'sequence_length' not in self.model_config:
            raise ValueError("sequence_length must be specified in config")
        if 'forecast_horizon' not in self.model_config:
            raise ValueError("forecast_horizon must be specified in config")
        
        
        self.model_config['layers'] = self.config['layers']
        self.dropout = self.config['dropout']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.sequence_length = self.config['sequence_length']
        if 'feature_cols' not in self.config:
            raise ValueError("feature_cols must be specified in config")
        self.feature_cols = self.config['feature_cols']
        self.forecast_horizon = self.config['forecast_horizon']
        self.model = None
        # num_targets will be calculated from data during training
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the Multivariate LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_targets)
        """
        
        self.model = Sequential()
        
        # Add LSTM layers
        for i in range(self.model_config['layers'):]
            return_sequences = i < self.model_config['layers'] - 1
            self.model.add(LSTM(
                units=self.model_config['units'],
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            ))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))
                
        # Add output layer - predicts forecast_horizon * num_targets values (flattened)
        self.model.add(Dense(self.forecast_horizon * self.num_targets))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.training_loss
        )
        
    def _prepare_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for Multivariate LSTM.
        
        Args:
            X: Input features (2D array with shape (num_timesteps, num_targets))
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared sequences and targets
        """
        # Ensure X is 2D: (num_timesteps, num_targets) for multivariate
        if X.ndim == 1:
            # If univariate input, reshape to (num_timesteps, 1)
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            # If 3D or higher, flatten to 2D
            X = X.reshape(X.shape[0], -1)
            
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length - self.forecast_horizon + 1):
            X_seq.append(X[i:(i + self.sequence_length)])
            # y_seq: flatten to 1D array of length forecast_horizon * num_targets
            future_values = X[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
            y_seq.append(future_values.flatten())
        return np.array(X_seq), np.array(y_seq)
        
    def train(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame], y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_start_date: pd.Timestamp = None, **kwargs) -> 'MultivariateLSTMModel':
        """
        Train the Multivariate LSTM model on given data.
        
        TECHNIQUE: Sliding Window Multi-Step Learning for Multiple Targets
        - Creates overlapping sequences of length sequence_length from historical data
        - Each sequence predicts forecast_horizon future values for all targets simultaneously
        - Training pairs: Input [t, t+1, ..., t+sequence_length-1] → Target [t+sequence_length, ..., t+sequence_length+forecast_horizon-1]
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
        if isinstance(y_context, pd.DataFrame):
            # Multivariate case - use all columns
            y_data = y_context.values
            # Calculate num_targets from data shape
            self.num_targets = y_data.shape[1]
        elif isinstance(y_context, pd.Series):
            # Univariate case - reshape to 2D
            y_data = y_context.values.reshape(-1, 1)
            self.num_targets = 1
        else:
            # Numpy array case
            y_data = y_context
            if y_data.ndim == 1:
                y_data = y_data.reshape(-1, 1)
            self.num_targets = y_data.shape[1]

        # Store the raw target data for later use
        self.y_context = y_data
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(y_data)
        
        # Build model if not already built
        if self.model is None:
            self._build_model(input_shape=(self.sequence_length, self.num_targets))
            
        # TensorBoard logging is handled by the main benchmark runner
        # No need for separate logging here
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0
        )
        
        self.is_fitted = True
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained Multivariate LSTM model.
        Predicts len(y_target) steps ahead for all targets using non-overlapping windows.
        
        TECHNIQUE: Non-overlapping Multi-Step Windows for Multiple Targets
        - Starts with last sequence_length historical values for all targets
        - Predicts forecast_horizon steps at once for all targets (no autoregressive feedback)
        - Advances window by forecast_horizon steps for next prediction
        - Repeats until len(y_target) steps are predicted for all targets
        - Fair comparison with non-autoregressive models (ARIMA, Exponential Smoothing)
        - No data leakage from using own predictions as input
        
        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")

        # Convert input to numpy array
        if isinstance(y_context, pd.DataFrame):
            input_data = y_context.values
        elif isinstance(y_context, pd.Series):
            input_data = y_context.values.reshape(-1, 1)
        else:
            input_data = y_context
            if input_data.ndim == 1:
                input_data = input_data.reshape(-1, 1)
                
        forecast_steps = len(y_target)

        # Calculate how many prediction windows we need
        num_windows = math.ceil(forecast_steps / self.forecast_horizon)
        
        all_predictions = []
        current_sequence = input_data[-self.sequence_length:].reshape(1, self.sequence_length, self.num_targets)
        
        for window in range(num_windows):
            # Predict forecast_horizon steps at once for all targets
            predictions = self.model.predict(current_sequence, verbose=0)
            # Reshape predictions from (1, forecast_horizon * num_targets) to (forecast_horizon, num_targets)
            predictions_reshaped = predictions[0].reshape(self.forecast_horizon, self.num_targets)
            all_predictions.extend(predictions_reshaped)
            
            # Advance window by forecast_horizon steps and use predictions as input
            if window < num_windows - 1:  # Don't update on last iteration
                # Move window forward by forecast_horizon steps
                current_sequence = np.roll(current_sequence, -self.forecast_horizon, axis=1)
                # Only update as many as fit
                n = min(self.forecast_horizon, self.sequence_length)
                current_sequence[0, -n:, :] = predictions_reshaped[:n, :]
        
        # Return only the requested number of predictions
        result = np.array(all_predictions[:forecast_steps])
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'units': self.model_config['units'],
            'layers': self.model_config['layers',]
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'sequence_length': self.sequence_length,
            'num_targets': self.num_targets,
            'feature_cols': self.feature_cols,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateLSTMModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
        
    def save(self, path: str) -> None:
        """
        Save the Multivariate LSTM model to disk.
        
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
        Load the Multivariate LSTM model from disk.
        
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
        
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Name of the loss function to use (defaults to training_loss)
            
        Returns:
            Dict[str, float]: Dictionary of computed loss metrics
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        elif isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Store for TensorBoard logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # For multivariate data, compute loss for each target and average
        if y_true.ndim == 2 and y_pred.ndim == 2:
            # Multivariate case - compute loss for each target and average
            all_metrics = {}
            for i in range(y_true.shape[1]):
                target_true = y_true[:, i]
                target_pred = y_pred[:, i]
                
                # Ensure both arrays have the same length
                min_length = min(len(target_true), len(target_pred))
                target_true = target_true[:min_length]
                target_pred = target_pred[:min_length]
                
                # Compute metrics for this target
                target_metrics = self.evaluator.evaluate(target_pred, target_true)
                
                # Store metrics for this target
                for metric_name, metric_value in target_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Average metrics across all targets
            averaged_metrics = {}
            for metric_name, metric_values in all_metrics.items():
                averaged_metrics[metric_name] = np.mean(metric_values)
            
            return averaged_metrics
        else:
            # Univariate case - use original logic
            # Handle shape mismatches
            if y_pred.ndim == 2 and y_true.ndim == 1:
                # If predictions are 2D and true values are 1D, flatten predictions
                if y_pred.shape[0] == 1:
                    # Single prediction row, flatten it
                    y_pred = y_pred.flatten()
                elif y_pred.shape[1] == 1:
                    # Single prediction column, flatten it
                    y_pred = y_pred.flatten()
                else:
                    # Multiple predictions, take the first row
                    y_pred = y_pred[0]
            
            # Ensure both arrays have the same length
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            
            # Use evaluator to compute all metrics
            return self.evaluator.evaluate(y_pred, y_true) 