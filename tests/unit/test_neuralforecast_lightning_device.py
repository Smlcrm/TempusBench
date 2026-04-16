"""NeuralForecast Lightning trainer device resolution (CPU vs GPU)."""

from unittest.mock import patch

import pytest

from tempus_bench.models.neuralforecast_lightning_device import resolve_neuralforecast_trainer_kwargs


class TestResolveNeuralforecastTrainerKwargs:
    def test_cpu_explicit(self) -> None:
        assert resolve_neuralforecast_trainer_kwargs("cpu") == {"accelerator": "cpu", "devices": 1}

    def test_default_when_none(self) -> None:
        assert resolve_neuralforecast_trainer_kwargs(None) == {"accelerator": "cpu", "devices": 1}

    def test_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_neuralforecast_trainer_kwargs("cuda") == {"accelerator": "gpu", "devices": 1}

    def test_cuda_raises_when_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="device=cuda"):
                resolve_neuralforecast_trainer_kwargs("cuda")

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError, match="Unsupported NeuralForecast device"):
            resolve_neuralforecast_trainer_kwargs("mps")
