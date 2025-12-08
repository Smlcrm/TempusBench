import torch
from tempus_bench.models.base_model import BaseModel, validate_inputs
from transformers import AutoModelForCausalLM
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel
import numpy as np


class YingHyperparameters(PydanticBaseModel):
  pass

class YingModel(BaseModel):

  def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
    """
        Initialize Ying model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
    super().__init__(params, settings, YingHyperparameters)

    self.model = None
  
  @validate_inputs
  def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "YingModel" :
    if self.model is None:
      self.model = AutoModelForCausalLM.from_pretrained('qcw2333/YingLong_6m', \
                                                        trust_remote_code=True,\
                                                          torch_dtype=torch.bfloat16)
    return self
  
  @validate_inputs
  def predict(self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,) -> np.ndarray:
    
    if len(y_context.shape) == 1:
      torch_y_context = torch.tensor(y_context[-self.lookback_length:])
    else:
      torch_y_context = torch.tensor(y_context[:, -self.lookback_length:])

    torch_y_context = torch_y_context.unsqueeze(0).bfloat16().float()


    predictions = self.model.generate(torch_y_context, future_token=self.prediction_length)
    
    return predictions.float().cpu().detach().numpy()
    