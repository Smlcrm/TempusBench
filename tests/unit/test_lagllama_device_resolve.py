"""Pure tests for Lag-Llama device string resolution (no torch / gluonts import chain)."""

from __future__ import annotations

from tempus_bench.utils.lagllama_device import resolve_lagllama_device_string


def test_cpu_tier_keeps_cpu() -> None:
    assert (
        resolve_lagllama_device_string(
            "cpu",
            worker_compute_tier="",
            cuda_available=True,
        )
        == "cpu"
    )


def test_gpu_tier_does_not_override_explicit_cpu() -> None:
    assert (
        resolve_lagllama_device_string(
            "cpu",
            worker_compute_tier="gpu",
            cuda_available=True,
        )
        == "cpu"
    )


def test_gpu_tier_stays_cpu_when_cuda_unavailable() -> None:
    assert (
        resolve_lagllama_device_string(
            "cpu",
            worker_compute_tier="gpu",
            cuda_available=False,
        )
        == "cpu"
    )


def test_explicit_cuda_preserved_even_on_cpu_tier() -> None:
    assert (
        resolve_lagllama_device_string(
            "cuda:0",
            worker_compute_tier="cpu",
            cuda_available=False,
        )
        == "cuda:0"
    )
