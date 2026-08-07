"""Data generators for the TempusBench generator taskbed.

One module per generator; ``metadata.json`` is the registry and doubles as the
generator-to-category mapping. Generator modules are imported lazily so a run
only pays for the generators it actually uses.

Typical use::

    from tempus_bench import generators

    seed = generators.resolve_seed(base_seed=0, name="random_walk")
    series = generators.generate("random_walk", T=2048, seed=seed)
"""

from __future__ import annotations

import functools
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_PATH = Path(__file__).parent / "metadata.json"

SEED_STRIDE = 10_000
"""Multiplier applied to the base seed before adding the generator ID.

Keeps per-generator seed ranges disjoint across base seeds, so no two
(base seed, generator) pairs ever resolve to the same effective seed. A plain
additive offset would collide: base 5 with ID 3 and base 6 with ID 2 both give 8.

Changing this value changes every derived seed and invalidates published results.
"""


@functools.lru_cache(maxsize=1)
def load_metadata() -> dict[str, dict[str, Any]]:
    """Return the generator registry, keyed by generator name."""
    with _METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _entry(name: str) -> dict[str, Any]:
    metadata = load_metadata()
    if name not in metadata:
        raise KeyError(f"Unknown generator {name!r}; not present in metadata.json")
    return metadata[name]


def _resolve_callable(name: str):
    """Import the module for one generator and return its function."""
    entry = _entry(name)
    module = importlib.import_module(f"{__name__}.{name}")
    return getattr(module, entry["function"])


def resolve_seed(base_seed: int, name: str) -> int:
    """Effective seed for one generator under a given base seed.

    Args:
        base_seed: The base seed from ``benchmark.yaml``.
        name: Generator name, a key of ``metadata.json``.

    Returns:
        ``base_seed * SEED_STRIDE + generator_id``.

    Raises:
        KeyError: If the generator is not in the registry.
    """
    return base_seed * SEED_STRIDE + _entry(name)["generator_id"]


def generate(
    name: str,
    T: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate the series for one generator.

    Args:
        name: Generator name, a key of ``metadata.json``.
        T: Series length. Defaults to the generator's ``default_series_length``.
        seed: Effective seed; pass the value returned by :func:`resolve_seed`.
        **kwargs: Forwarded to the generator function.

    Returns:
        ``(T,)`` for univariate generators, ``(T, m)`` otherwise.

    Raises:
        KeyError: If the generator is not in the registry.
    """
    entry = _entry(name)
    if T is None:
        T = entry["default_series_length"]
    return _resolve_callable(name)(T=T, seed=seed, **kwargs)


__all__ = ["SEED_STRIDE", "generate", "load_metadata", "resolve_seed"]
