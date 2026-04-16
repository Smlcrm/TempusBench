"""PyTorch Lightning trainer options for NeuralForecast models (TFT, TimesNet, etc.)."""

from typing import Dict, Optional, Union


def resolve_neuralforecast_trainer_kwargs(requested_device: Optional[str]) -> Dict[str, Union[str, int]]:
    """Map settings `device` to Lightning Trainer kwargs for NeuralForecast."""
    import torch

    requested = str(requested_device or "cpu").strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Model settings request device=cuda but torch.cuda.is_available() is false. "
                "Use the CUDA worker image and a Google Batch GPU machine type, or set device: cpu "
                "in settings.yaml."
            )
        return {"accelerator": "gpu", "devices": 1}
    if requested == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    raise ValueError(
        f"Unsupported NeuralForecast device {requested!r}; expected 'cpu' or 'cuda'."
    )
