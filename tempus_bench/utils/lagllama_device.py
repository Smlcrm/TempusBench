"""
Lag-Llama device selection.

``models/lagllama/settings.yaml`` uses ``device: cpu`` for CPU-only workers and local
runs. Explicit ``cuda`` / ``cuda:0`` values are preserved. Requested ``cpu`` is never
upgraded based on job tier so multivariate / covariate fan-out stays on the configured
device (notably CPU-only Batch jobs).
"""

from __future__ import annotations


def resolve_lagllama_device_string(
    requested: str,
    *,
    worker_compute_tier: str,
    cuda_available: bool,
) -> str:
    """
    Return the device string passed to ``torch.device`` for Lag-Llama.

    Args:
        requested: Value from model settings (e.g. ``\"cpu\"``, ``\"cuda\"``).
        worker_compute_tier: Raw ``WORKER_COMPUTE_TIER`` env (e.g. ``\"gpu\"``). Ignored
            for device selection; kept for API stability with existing call sites.
        cuda_available: Whether CUDA is available at runtime. Ignored unless ``requested``
            is a CUDA device string (callers may use it in the future for validation).
    """
    _ = (worker_compute_tier, cuda_available)
    raw = (requested or "cpu").strip()
    if not raw:
        return "cpu"
    low = raw.lower()
    if low.startswith("cuda"):
        return raw
    return raw
