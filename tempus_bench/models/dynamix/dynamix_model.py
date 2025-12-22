import numpy as np
import torch
from tempus_bench.models.base_model import BaseModel, validate_inputs
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.dynamix.dynamix.model.forecaster import DynaMixForecaster
from tempus_bench.models.dynamix.dynamix.utilities.utilities import load_hf_model


class DynamixHyperparameters(PydanticBaseModel):
  pass

class DynamixModel(BaseModel):

  def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
    """
        Initialize Dynamix model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
    super().__init__(params, settings, DynamixHyperparameters)

    self.model = None
  
  #@validate_inputs
  def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "DynamixModel" :
    if self.model is None:
      self.model = load_hf_model("dynamix-3d-alrnn-v1.0")
    return self 
  
  #@validate_inputs
  def predict(self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,) -> np.ndarray:
    
    print("before tensor step",y_context.shape)

    self.model.eval()

    forecaster = DynaMixForecaster(self.model)


    torch_y_context = torch.tensor(y_context[-self.lookback_length:].T,  dtype=torch.float32)
    
    print("after tensor step",torch_y_context.shape)
    #print("timesteps", self.forecast_length)
    #print("samples", kwargs["num_samples"])
    

    forecast_length = int(timestamps_target.shape[0])
    print("forecast length", forecast_length)

    with torch.no_grad():  # No gradient tracking needed for inference
      predictions = forecaster.forecast(
          context=torch_y_context,
          horizon=forecast_length,
          preprocessing_method="pos_embedding",
          standardize=True,
          fit_nonstationary=False,
          initial_x=None
      )

    #predictions = self.model.generate(torch_y_context, max_new_tokens=forecast_length, \
    #                                  num_samples=kwargs["num_samples"])
    #predictions = np.asarray(predictions)

    print("predictions shape", predictions.shape)
    
    #predictions_transposed = np.transpose(predictions, axes=(1,2,0))
    #print("Final tranpose shape", predictions_transposed.shape)
    #print("Timestamps target shape", timestamps_target.shape)


    return predictions