"""
Task and model catalog helpers (for UIs and tooling that need task lists and model flags).
"""
from __future__ import annotations

from tempus_bench.utils.model_settings import load_capabilities_for_model


def derive_job_type(tasks: list[str]) -> str:
    """Derive job_type from task paths (e.g. univariate/solar, multivariate/..., covariate/...)."""
    if not tasks:
        return "univariate"
    for t in tasks:
        if t.startswith("covariate/"):
            return "covariate"
    for t in tasks:
        if t.startswith("multivariate/"):
            return "multivariate"
    return "univariate"


def extract_task_paths_from_config(config_yaml: str) -> list[str]:
    """Extract task_path or task_paths from evaluation config YAML. Returns [] if not found."""
    import yaml

    try:
        data = yaml.safe_load(config_yaml) or {}
        ev = data.get("evaluation") or {}
        task_path = ev.get("task_path")
        task_paths = ev.get("task_paths")
        if task_paths and isinstance(task_paths, list):
            return [str(p) for p in task_paths]
        if task_path:
            return [str(task_path)]
    except Exception:
        pass
    return []


def get_tasks() -> dict[str, list[str]]:
    """Return tasks grouped by evaluation mode for tasks/."""
    from tempus_bench.utils.paths import get_tasks_dir
    from tempus_bench.utils.task_yaml_loader import COVARIATE_TASK_SUFFIX

    tasks_dir = get_tasks_dir()
    result: dict[str, list[str]] = {"univariate": [], "multivariate": [], "covariate": []}

    uni = tasks_dir / "univariate"
    if uni.is_dir():
        for task_path in uni.iterdir():
            if task_path.is_dir():
                result["univariate"].append(f"univariate/{task_path.name}")

    multi = tasks_dir / "multivariate"
    if multi.is_dir():
        for task_path in multi.iterdir():
            if task_path.is_dir():
                rel = f"multivariate/{task_path.name}"
                result["multivariate"].append(rel)
                result["covariate"].append(f"{rel}{COVARIATE_TASK_SUFFIX}")

    for key in result:
        result[key] = sorted(result[key])
    return result


def get_models() -> list[dict]:
    """Return capability flags for each registered model (from settings.yaml)."""
    from tempus_bench.utils.paths import get_available_models

    models = sorted(get_available_models())
    out: list[dict] = []
    for m in models:
        cap = load_capabilities_for_model(m)
        out.append(
            {
                "id": m,
                "covariate_support": cap.covariates != "none",
                "supports_univariate": cap.univariate,
                "supports_multivariate": cap.multivariate,
            }
        )
    return out
