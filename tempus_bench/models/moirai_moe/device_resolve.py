"""Torch device selection for Moirai-MoE (no uni2ts import — safe for lightweight unit tests)."""

from __future__ import annotations

import torch


def resolve_moirai_moe_torch_device(device_setting: str) -> torch.device:
    """Map ``settings.yaml`` ``device`` string to a :class:`torch.device`.

    - ``auto`` / ``cuda_if_available``: CUDA when :func:`torch.cuda.is_available` else CPU.
    - ``cuda``: CUDA only; raises if CUDA is not available.
    - ``cuda:N``: specific GPU index when CUDA is available.
    - ``cpu``: CPU.
    """
    raw = (device_setting or "cpu").strip().lower()
    if raw in ("auto", "cuda_if_available"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "moirai_moe: settings.device is 'cuda' but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    if raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"moirai_moe: settings.device is {device_setting!r} but "
                "torch.cuda.is_available() is False"
            )
        return torch.device(raw)
    return torch.device("cpu")
