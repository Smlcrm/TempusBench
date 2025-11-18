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
    return self.model 
  
  #@validate_inputs
  def predict(self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,) -> np.ndarray:
    
    if len(y_context.shape) == 1:
      torch_y_context = torch.tensor(y_context[-self.lookback_length:])
    else:
      torch_y_context = torch.tensor(y_context[:, -self.lookback_length:])

    torch_y_context = torch_y_context.unsqueeze(0).float()
    predictions = self.model.generate(torch_y_context, max_new_tokens=self.forecast_length, \
                                      num_samples=self.num_samples)
    
    return np.transpose(predictions.squeeze())
    