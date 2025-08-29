import pandas as pd
import numpy as np
import torch
import timesfm
from typing import Dict, Any, Optional, List, Union
from benchmarking_pipeline.models.foundation_model import FoundationModel

class TimesFMModel(FoundationModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        super().__init__(config, config_file)
        if 'per_core_batch_size' not in self.config:
            raise ValueError("per_core_batch_size must be specified in config")
        if 'num_layers' not in self.config:
            raise ValueError("num_layers must be specified in config")
        if 'context_len' not in self.config:
            raise ValueError("context_len must be specified in config")
        if 'use_positional_embedding' not in self.config:
            raise ValueError("use_positional_embedding must be specified in config")
        
        self.per_core_batch_size = self.config["per_core_batch_size"]
        self.num_layers = self.config["num_layers"]
        self.context_len = self.config["context_len"]
        self.use_positional_embedding = self.config["use_positional_embedding"]
        print(f"Loading TimesFM model: {self.model_path} to device '{self.device}'")
        self.is_fitted = True

    def set_params(self, **params: Dict[str, Any]) -> 'TimesFMModel':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            self.config[key] = value
        return self

    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_context_timestamps=None,
        y_target_timestamps=None,
        **kwargs
    ):
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if self.device == "cuda" else "cpu",
                per_core_batch_size=self.per_core_batch_size,
                horizon_len=self.forecast_horizon,
                num_layers=self.num_layers,
                use_positional_embedding=self.use_positional_embedding,
                context_len=self.context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_path
            ),
        )
        
        # Convert y_context to DataFrame
        if y_context is None:
            raise ValueError("y_context must be provided")
        if isinstance(y_context, pd.Series):
            df = y_context.to_frame()
        elif isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame(y_context, columns=["series"])
            else:
                df = pd.DataFrame(y_context.T)
        elif isinstance(y_context, pd.DataFrame):
            df = y_context.copy()
        else:
            raise ValueError("Unsupported y_context type")

        # Prepare context windows
        contexts = []
        for series_name in df.columns:
            series_data = df[series_name].values
            context = series_data[-self.context_len:]
            if len(context) < self.context_len:
                padding = np.zeros(self.context_len - len(context))
                context = np.concatenate([padding, context])
            contexts.append(context)

        input_data = torch.tensor(np.stack(contexts), dtype=torch.float32).to(self.device)

        # Generate forecasts
        point_forecasts = model.forecast(input_data)
        results = {}
        for i, series_name in enumerate(df.columns):
            results[series_name] = point_forecasts[i].tolist()
        if len(results) == 1:
            return np.array(list(results.values())[0])
        else:
            return np.array(list(results.values()))

    def train(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Optional[Union[pd.Series, np.ndarray]] = None,
        y_start_date: Optional[str] = None,
        **kwargs,
    ) -> 'TimesFMModel':
        """
        Foundation model: no training needed. Mark as fitted and return self.
        """
        self.is_fitted = True
        return self

    def get_params(self) -> Dict[str, Any]:
        """
        Return model configuration parameters used by this instance.
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'per_core_batch_size': self.per_core_batch_size,
            'forecast_horizon': self.forecast_horizon,
            'num_layers': self.num_layers,
            'context_len': self.context_len,
            'use_positional_embedding': self.use_positional_embedding,
        }