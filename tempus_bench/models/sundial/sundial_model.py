import numpy as np
import torch
from tempus_bench.models.base_model import BaseModel, validate_inputs
from transformers import AutoModelForCausalLM
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel


class SundialHyperparameters(PydanticBaseModel):
  pass

class SundialModel(BaseModel):

  def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
    """
        Initialize Moirai model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
    super().__init__(params, settings, SundialHyperparameters)

    self.model = None
  
  #@validate_inputs
  def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "SundialModel" :
    if self.model is None:
      self.model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', \
                                                        trust_remote_code=True)
    return self 
  
  #@validate_inputs
  def predict(self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,) -> np.ndarray:
    
    print("before tensor step",y_context.shape)




    torch_y_context = torch.tensor(y_context[-self.lookback_length:].T)

    torch_y_context = torch_y_context.float()
    print("after tensor step",torch_y_context.shape)
    print("timesteps", self.forecast_length)
    print("samples", kwargs["num_samples"])
    print(torch_y_context.shape)

    forecast_length = int(timestamps_target.shape[0])
    predictions = self.model.generate(torch_y_context, max_new_tokens=forecast_length, \
                                      num_samples=kwargs["num_samples"])
    predictions = np.asarray(predictions)

    print("predictions shape", predictions.shape)
    print("predictions shape squeeze", predictions.squeeze().shape)
    
    predictions_transposed = np.transpose(predictions, axes=(1,2,0))
    print("Final tranpose shape", predictions_transposed.shape)
    print("Timestamps target shape", timestamps_target.shape)


    return predictions_transposed
    