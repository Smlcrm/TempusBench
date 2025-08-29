import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.foundation_model import FoundationModel
from momentfm import MOMENTPipeline
from tqdm import tqdm


class MomentDataset(Dataset):
    """Dataset class for MOMENT model training and inference."""
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int, scaler: Optional[StandardScaler] = None):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_series, self.n_timesteps = data.shape
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(data.T)
        else:
            self.scaler = scaler
        self.scaled_data = self.scaler.transform(data.T).T
    def __len__(self):
        samples_per_series = max(0, self.n_timesteps - self.context_length - self.prediction_length + 1)
        return self.n_series * samples_per_series
    def __getitem__(self, idx):
        samples_per_series = max(0, self.n_timesteps - self.context_length - self.prediction_length + 1)
        series_idx = idx // samples_per_series
        time_idx = idx % samples_per_series
        start_idx = time_idx
        context_end = start_idx + self.context_length
        target_end = context_end + self.prediction_length
        context = self.scaled_data[series_idx, start_idx:context_end]
        target = self.scaled_data[series_idx, context_end:target_end]
        context = context.reshape(1, -1)
        target = target.reshape(1, -1)
        input_mask = np.ones(self.context_length)
        return torch.FloatTensor(context), torch.FloatTensor(target), torch.FloatTensor(input_mask)


class MomentModel(FoundationModel):
    """MOMENT model wrapper for time series forecasting, extending FoundationModel."""
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        super().__init__(config, config_file)
        if 'model_path' not in self.config:
            raise ValueError("model_path must be specified in config")
        if 'context_length' not in self.config:
            raise ValueError("context_length must be specified in config")
        if 'fine_tune_epochs' not in self.config:
            raise ValueError("fine_tune_epochs must be specified in config")
        if 'batch_size' not in self.config:
            raise ValueError("batch_size must be specified in config")
        if 'learning_rate' not in self.config:
            raise ValueError("learning_rate must be specified in config")
        
        self.model_path = self.config['model_path']
        self.context_length = self.config['context_length']
        self.fine_tune_epochs = self.config['fine_tune_epochs']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        # forecast_horizon is inherited from parent class (FoundationModel)
        self.is_fitted = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.pdt is also a 
        print(f"MOMENT Model initialized - Device: {self.device}")
        print(f"Context length: {self.context_length}, Fine-tune epochs: {self.fine_tune_epochs}")

    def get_params(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'context_length': self.context_length,
            'fine_tune_epochs': self.fine_tune_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'forecast_horizon': self.forecast_horizon,
        }

    def set_params(self, **params: Dict[str, Any]) -> 'MomentModel':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _load_model(self, forecast_horizon: int):
        print(f"Loading MOMENT model for forecast horizon: {forecast_horizon}")
        self.model = MOMENTPipeline.from_pretrained(
            self.model_path,
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': forecast_horizon,
                'head_dropout': 0.1,
                'freeze_encoder': True,
                'freeze_embedder': True,
                'freeze_head': False,
            },
        )
        self.model.init()
        self.model = self.model.to(self.device)
        print("MOMENT model loaded successfully!")

    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray]], 
              y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_start_date: Optional[str] = None
    ) -> 'MomentModel':
        """
        Train/fine-tune the MOMENT model on given data.
        
        Args:
            y_context: Past target values
            y_target: Future target values (not used for MOMENT)
            y_start_date: Start date timestamp (not used for MOMENT)
            
        Returns:
            self: The fitted model instance
        """
        if y_context is None:
            raise ValueError("y_context is required for MOMENT")
        
        print(f"DEBUG: y_context type: {type(y_context)}")
        print(f"DEBUG: y_context shape: {y_context.shape if hasattr(y_context, 'shape') else 'no shape'}")
        print(f"DEBUG: y_context: {y_context}")
        
        # Convert to DataFrame format
        if isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame({'series': y_context})
            else:
                df = pd.DataFrame(y_context.T)
        else:
            df = y_context
        
        print(f"DEBUG: df shape: {df.shape}")
        print(f"DEBUG: df head:\n{df.head()}")
        
        # Use the existing fit method
        return self.fit(df, self.forecast_horizon, verbose=True)

    def fit(self, df: pd.DataFrame, forecast_horizon: int, verbose: bool = True):
        self._load_model(forecast_horizon)
        print(f"DEBUG: df in fit: {df.shape}")
        print(f"DEBUG: df head in fit:\n{df.head()}")
        
        # The data comes in as (n_targets, time_steps) from DataFrame
        # We need to convert it to (n_series, time_steps) format for MomentDataset
        # where each row is a different time series
        data = df.values  # Shape: [n_targets, time_steps]
        print(f"DEBUG: data before processing: {data.shape}")
        print(f"DEBUG: data sample: {data[:, :5] if data.shape[1] >= 5 else data}")
        
        min_required = self.context_length + forecast_horizon
        if data.shape[1] < min_required:
            raise ValueError(
                f"Time series too short! Need at least {min_required} timesteps, "
                f"got {data.shape[1]}. Consider reducing context_length or forecast_horizon."
            )
        dataset = MomentDataset(data, self.context_length, forecast_horizon)
        self.scaler = dataset.scaler  # Store scaler for inference
        if len(dataset) == 0:
            raise ValueError("No valid training samples created. Check your data length and parameters.")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.learning_rate))
        self.model.train()
        if verbose:
            print(f"Fine-tuning MOMENT on {len(dataset)} samples...")
        for epoch in range(self.fine_tune_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.fine_tune_epochs}") if verbose else dataloader
            for context, target, input_mask in pbar:
                context = context.to(self.device)
                target = target.to(self.device)
                input_mask = input_mask.to(self.device)
                optimizer.zero_grad()
                output = self.model(x_enc=context, input_mask=input_mask)
                loss = criterion(output.forecast, target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                if verbose and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            avg_loss = np.mean(epoch_losses)
            if verbose:
                print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
        self.is_fitted = True
        #print("MOMENT fine-tuning completed!")

    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        y_context_timestamps=None,
        y_target_timestamps=None,
        forecast_horizon: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions using the trained MOMENT model.
        
        Args:
            y_context: Recent/past target values
            y_target: Future target values (not used for MOMENT)
            y_context_timestamps: Timestamps for context data (not used for MOMENT)
            y_target_timestamps: Timestamps for target data (not used for MOMENT)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon,)
        """
        if y_context is None:
            raise ValueError("y_context is required for prediction")
        
        horizon = forecast_horizon or self.forecast_horizon
        
        # Convert to DataFrame format
        if isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame({'series': y_context})
            else:
                df = pd.DataFrame(y_context.T)
        else:
            df = y_context
        
        # Ensure we have a fitted model
        if not self.is_fitted:
            self.train(y_context)
        
        # Use the internal prediction method
        results = self._sub_predict(df, horizon)
        
        # Return as numpy array for single series
        if 'series' in results:
            return np.array(results['series'])
        elif len(results) == 1:
            # Return first series as numpy array
            first_series = list(results.values())[0]
            return np.array(first_series)
        else:
            # Fallback: return zeros
            return np.zeros(horizon)

    def _sub_predict(self, df: pd.DataFrame, forecast_horizon: int) -> Dict[str, List[float]]:
        if not self.is_fitted:
            self.fit(df, forecast_horizon, verbose=True)
        
        self.model.eval()
        results = {}
        with torch.no_grad():
            for col in df.columns:
                series_data = df[col].values
                if len(series_data) >= self.context_length:
                    context = series_data[-self.context_length:]
                else:
                    padding = np.zeros(self.context_length - len(series_data))
                    context = np.concatenate([padding, series_data])
                    warnings.warn(
                        f"Series '{col}' is shorter than context_length {self.context_length}. "
                        "Padded with zeros.",
                        UserWarning
                    )
                
                # Scale the context data using the fitted scaler
                context_scaled = self.scaler.transform(context.reshape(-1, 1)).flatten()
                
                # Create proper 3D tensor: [batch_size=1, sequence_length=1, features=1]
                context_tensor = torch.FloatTensor(context_scaled).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Create input mask
                input_mask = torch.ones(1, self.context_length).to(self.device)
                
                # Get prediction
                output = self.model(x_enc=context_tensor, input_mask=input_mask)
                
                # Inverse scale the forecast
                forecast_scaled = output.forecast.cpu().numpy().flatten()
                forecast = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                
                results[col] = forecast.tolist()
        
        return results

    def predict_zero_shot(self, df: pd.DataFrame, forecast_horizon: int) -> Dict[str, List[float]]:
        self._load_model(forecast_horizon)
        # For zero-shot, we don't fine-tune, just use the pre-trained model
        return self._sub_predict(df, forecast_horizon)

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'MOMENT-1-large',
            'context_length': self.context_length,
            'is_fitted': self.is_fitted,
            'device': self.device,
            'fine_tune_epochs': self.fine_tune_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'forecast_horizon': self.forecast_horizon,
        }


# testing
"""
if __name__ == "__main__":
    # Create sample data similar to your TimesFM example
            # Use frequency from CSV data - this should be passed in as a parameter
        raise ValueError("Frequency must be provided from CSV data. Cannot use hardcoded defaults.")
    np.random.seed(42)
    # Generate synthetic time series
    series1 = 100 + np.arange(1000) * 0.1 + 10 * np.sin(2 * np.pi * np.arange(1000) / 30) + np.random.normal(0, 2, 1000)
    series2 = 200 + np.arange(1000) * 0.05 + 15 * np.cos(2 * np.pi * np.arange(1000) / 45) + np.random.normal(0, 3, 1000)
    df = pd.DataFrame({
        'sales': series1,
        'revenue': series2
    }, index=dates)
    print("Testing MOMENT Model...")
    # Initialize model
    config = {
        'context_length': 256,
        'fine_tune_epochs': 2,
        'batch_size': 4
    }
    model = MomentModel(config=config)
    try:
        forecasts = model._sub_predict(df, prediction_length=24)
        print("\nForecasts generated successfully!")
        for series_name, forecast in forecasts.items():
            print(f"{series_name}: {forecast[:5]}... (showing first 5 values)")
        print(f"\nModel info: {model.get_model_info()}")
    except Exception as e:
        print(f"Error: {e}")
"""

if __name__ == "__main__":
    # Test the model
    from benchmarking_pipeline.pipeline.data_loader import DataLoader
    
    # Create a simple test dataset
    np.random.seed(42)
    n_series = 3
    n_timesteps = 100
    test_data = np.random.randn(n_series, n_timesteps)
    test_df = pd.DataFrame(test_data.T, columns=[f'series_{i}' for i in range(n_series)])
    
    # Initialize and test the model
    config = {'dataset': {'forecast_horizon': 24}}
    model = MomentModel(config)
    
    print("Testing MOMENT model...")
    forecasts = model.predict_zero_shot(test_df, forecast_horizon=24)
    print(f"Generated forecasts for {len(forecasts)} series")
    for series_name, forecast in forecasts.items():
        print(f"{series_name}: {len(forecast)} forecast steps")