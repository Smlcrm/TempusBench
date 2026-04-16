"""
Per-model ``models/<name>/settings.yaml``: capabilities parsing, validation, and
covariate-mode helpers used by the pipeline and UI export.

Each settings.yaml must include a top-level `capabilities` block:

  capabilities:
    covariates: past_future | past_only | future_only | none
    univariate: true | false
    multivariate: true | false

Optional (used by the cloud Submit UI; see the private cloud repo’s ``deployment/deploy_submit_defaults.py``):

  foundation: true | false
  default_hyperparameter_grid:
    <param_name>: [candidate values, ...]   # empty ``{}`` for models tuned via empty grid

For **non-foundation** models, the benchmark YAML ``model.<name>`` block is merged with
``default_hyperparameter_grid``: any hyperparameter **omitted** from the benchmark config
is filled with the **full candidate list** from the grid (e.g. ``stochastic_samples: [50, 100, 200]``
in settings becomes that same list in ``ModelConfig``). Scalar grid values are wrapped as
a one-element list. Foundation models are unchanged (empty config stays empty).

See ``docs/models_capabilities.md`` in this package's documentation tree.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .paths import get_models_dir


CovariateMode = Literal["past_future", "past_only", "future_only", "none"]
TaskFamily = Literal["univariate", "multivariate", "covariate"]


class ModelCapabilities(BaseModel):
    """Mandatory capability flags for a registered model (see settings.yaml)."""

    model_config = ConfigDict(extra="forbid")

    covariates: CovariateMode = Field(
        description="How external covariate channels are wired into the model."
    )
    univariate: bool = Field(
        description="Safe to run on tasks under tempus_bench/tasks/univariate/",
    )
    multivariate: bool = Field(
        description="Safe to run on tasks under tempus_bench/tasks/multivariate/",
    )


def parse_capabilities_from_settings(
    settings: dict[str, Any],
    *,
    model_name: str,
) -> ModelCapabilities:
    raw = settings.get("capabilities")
    if raw is None:
        raise ValueError(
            f"Model {model_name!r}: settings.yaml must define a top-level "
            "'capabilities' block (covariates, univariate, multivariate). "
            "See docs/models_capabilities.md in the package documentation."
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"Model {model_name!r}: 'capabilities' must be a mapping, got {type(raw).__name__}."
        )
    try:
        return ModelCapabilities.model_validate(raw)
    except Exception as e:
        raise ValueError(
            f"Model {model_name!r}: invalid capabilities: {e}"
        ) from e


@lru_cache(maxsize=256)
def load_model_settings_yaml(model_name: str) -> dict[str, Any]:
    """Load raw ``models/<name>/settings.yaml`` as a mapping (cached)."""
    path = get_models_dir() / model_name / "settings.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No settings.yaml for model {model_name!r} at {path}")
    with open(path, encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    if not isinstance(settings, dict):
        raise ValueError(f"Model {model_name!r}: settings.yaml must be a mapping")
    return settings


def load_capabilities_for_model(model_name: str) -> ModelCapabilities:
    """Load and validate capabilities from models/<name>/settings.yaml."""
    settings = load_model_settings_yaml(model_name)
    return parse_capabilities_from_settings(settings, model_name=model_name)


def _grid_values_as_list(raw: Any, *, model_name: str, param_name: str) -> list[Any]:
    if isinstance(raw, list):
        values = raw
    else:
        values = [raw]
    if len(values) == 0:
        raise ValueError(
            f"Model {model_name!r}: default_hyperparameter_grid[{param_name!r}] "
            "cannot be an empty list."
        )
    return values


def merge_benchmark_params_with_default_grid(
    model_name: str,
    benchmark_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge benchmark ``model.<name>`` hyperparameters with ``default_hyperparameter_grid``.

    Foundation models: returns a shallow copy of ``benchmark_params`` (no grid merge).

    Non-foundation: for each key in ``default_hyperparameter_grid``, if the key is absent
    from ``benchmark_params``, sets it to a copy of that grid entry as a list (full search
    space for :class:`~tempus_bench.utils.configs.ModelConfig`). Scalars become ``[scalar]``.
    """
    settings = load_model_settings_yaml(model_name)
    foundation = settings.get("foundation")
    if foundation is None:
        raise ValueError(
            f"Model {model_name!r}: settings.yaml must define top-level boolean 'foundation'."
        )
    if not isinstance(foundation, bool):
        raise ValueError(
            f"Model {model_name!r}: 'foundation' must be a boolean, got {type(foundation).__name__}."
        )
    if foundation:
        return copy.copy(benchmark_params)

    raw_grid = settings.get("default_hyperparameter_grid")
    if raw_grid is None:
        grid: dict[str, Any] = {}
    elif isinstance(raw_grid, dict):
        grid = raw_grid
    else:
        raise ValueError(
            f"Model {model_name!r}: 'default_hyperparameter_grid' must be a mapping or omitted, "
            f"got {type(raw_grid).__name__}."
        )

    merged = copy.copy(benchmark_params)
    for param_name, raw_values in grid.items():
        if not isinstance(param_name, str) or not param_name.strip():
            raise ValueError(
                f"Model {model_name!r}: default_hyperparameter_grid keys must be non-empty strings."
            )
        if param_name in merged:
            continue
        values = _grid_values_as_list(raw_values, model_name=model_name, param_name=param_name)
        merged[param_name] = list(values)
    return merged


def task_path_to_family(task_path: str) -> TaskFamily:
    """Infer task family from dataset path (folder under tempus_bench/tasks/)."""
    p = Path(task_path).resolve()
    parts = p.parts
    try:
        i = parts.index("tasks")
        kind = parts[i + 1].lower()
    except (ValueError, IndexError):
        raise ValueError(f"Cannot infer task family from path: {task_path!r}")
    if kind == "univariate":
        return "univariate"
    if kind == "multivariate":
        return "multivariate"
    if kind == "covariate":
        return "covariate"
    raise ValueError(
        f"Unknown task family {kind!r} under tasks/ in path: {task_path!r}"
    )


def assert_model_supports_task_family(
    cap: ModelCapabilities,
    *,
    model_name: str,
    family: TaskFamily,
) -> None:
    if family == "univariate" and not cap.univariate:
        raise ValueError(
            f"Model {model_name!r} is not declared for univariate tasks "
            f"(capabilities.univariate: false)."
        )
    if family == "multivariate" and not cap.multivariate:
        raise ValueError(
            f"Model {model_name!r} is not declared for multivariate tasks "
            f"(capabilities.multivariate: false)."
        )
    if family == "covariate" and cap.covariates == "none":
        raise ValueError(
            f"Model {model_name!r} is not declared for covariate tasks "
            f"(capabilities.covariates: none)."
        )


def iter_capabilities_for_all_registered_models() -> list[tuple[str, ModelCapabilities]]:
    """Scan models directory for valid settings (same discovery as get_available_models)."""
    from .paths import get_available_models

    out: list[tuple[str, ModelCapabilities]] = []
    for name in sorted(get_available_models()):
        out.append((name, load_capabilities_for_model(name)))
    return out


def _cap(model_name: str) -> ModelCapabilities:
    return load_capabilities_for_model(model_name)


def get_covariate_support(model_name: str) -> str:
    """Return covariate mode: past_future | past_only | future_only | none."""
    return _cap(model_name).covariates


def is_past_only_covariates(model_name: str) -> bool:
    """True if the model receives only past covariates (x_context); x_target=None."""
    return get_covariate_support(model_name) == "past_only"


def is_no_covariates(model_name: str) -> bool:
    """True if the model supports no covariates; x_context=None, x_target=None."""
    return get_covariate_support(model_name) == "none"


def has_covariate_support(model_name: str) -> bool:
    """True if the model supports any covariate mode other than none."""
    return get_covariate_support(model_name) != "none"


@lru_cache(maxsize=1)
def get_past_only_covariate_models() -> frozenset[str]:
    from .paths import get_available_models

    return frozenset(
        m
        for m in get_available_models()
        if get_covariate_support(m) == "past_only"
    )


@lru_cache(maxsize=1)
def get_no_covariate_models() -> frozenset[str]:
    from .paths import get_available_models

    return frozenset(
        m for m in get_available_models() if get_covariate_support(m) == "none"
    )


def clear_model_settings_cache() -> None:
    """Clear caches (e.g. in tests after editing settings.yaml)."""
    get_past_only_covariate_models.cache_clear()
    get_no_covariate_models.cache_clear()
    load_model_settings_yaml.cache_clear()
