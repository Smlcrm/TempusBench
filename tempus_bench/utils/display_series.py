"""
Build display-series JSON for the web UI: same missing-value handling as training,
targets without StandardScaler (raw units for plots vs denormalized predictions).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import DatasetConfig, EvaluationConfig, TaskConfig
from tempus_bench.utils.log_manager import LogManager

# Keep in sync with the Next.js route app/api/tasks/data/route.ts in inference-tempusbench-cloud (plot payload cap).
_PLOT_BENCHMARK_MAX_WINDOW_INDEX: int = 1023


def _plot_display_prefix_len(
    n_steps: int,
    context_window: int,
    forecast_horizon: int,
    *,
    max_window_idx: int = _PLOT_BENCHMARK_MAX_WINDOW_INDEX,
) -> int:
    """
    Prefix length that covers benchmark windows 0..max_window_idx with stride = forecast_horizon.

    Minimum is context_window + 2 * forecast_horizon (one context+train+validate span at series start).
    """
    if n_steps <= 0:
        return 0
    fh = max(1, int(forecast_horizon))
    cw = max(0, int(context_window))
    cap = cw + fh * (2 + max(0, int(max_window_idx)))
    return min(n_steps, cap)


def _ensure_log_manager_for_export() -> None:
    if LogManager.log_manager is not None:
        return
    td = tempfile.mkdtemp(prefix="tempusbench-display-")
    LogManager(
        logs_path=td,
        console_logging=False,
        file_logging=False,
        tensorboard_logging=False,
    )


def load_task_config_from_task_dir(task_dir: Path) -> TaskConfig:
    """Load ``TaskConfig`` from ``task_dir/task.yaml`` (``task_path`` is the on-disk task directory)."""
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing task.yaml: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        documents = list(yaml.safe_load_all(f))

    task_name = task_dir.name
    resolved_dir = str(task_dir.resolve())
    for task_data in documents:
        if not task_data or "task" not in task_data:
            continue
        task_data["task"].pop("task_name", None)
        dataset = DatasetConfig(**task_data["task"].pop("dataset"))
        return TaskConfig(
            task_name=task_name,
            task_path=resolved_dir,
            **task_data["task"],
            dataset=dataset,
        )

    raise ValueError(f"No valid task document in {yaml_path}")


def default_evaluation_config_for_display() -> EvaluationConfig:
    """Minimal eval config for loading full series (matches integration-style usage)."""
    return EvaluationConfig(
        task_path="*",
        tuning_loss="mae",
        max_windows=4,
        max_num_variates=None,
        num_samples=1,
        num_quantiles=1,
        point_forecast_statistic="mean",
    )


def build_display_series_document(
    task_config: TaskConfig,
    catalog_task_id: str,
    evaluation_config: EvaluationConfig | None = None,
) -> Dict[str, Any]:
    """
    Returns a dict aligned with the Next.js task plot API (timestamps, variates, etc.).

    ``catalog_task_id`` is the key in ``task_artifacts`` (e.g. ``univariate/foo``).
    """
    _ensure_log_manager_for_export()
    eval_cfg = evaluation_config or default_evaluation_config_for_display()
    loader = DataLoader(
        task_config, eval_cfg, force_no_normalize=True
    )
    ds = loader.dataset
    meta = ds.metadata or {}

    ts_raw = ds.timestamps
    idx = pd.DatetimeIndex(pd.to_datetime(ts_raw, utc=True))
    unix_ts = (idx.astype(np.int64) // 10**9).astype(np.int64).tolist()

    arr = np.asarray(ds.target, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected target (n_steps, n_variates), got {arr.shape}")
    n_steps, n_variates = arr.shape
    n_steps_full = int(n_steps)
    variates = [arr[:, j].tolist() for j in range(n_variates)]

    cap = _plot_display_prefix_len(
        n_steps_full,
        task_config.context_window,
        task_config.forecast_horizon,
    )
    if cap < n_steps_full:
        unix_ts = unix_ts[:cap]
        variates = [row[:cap] for row in variates]
        n_steps = cap

    csv_path = Path(task_config.task_path) / task_config.dataset.file_name
    df_meta = pd.read_csv(
        csv_path,
        usecols=["variable_name", "variable_type"],
    )
    vt = df_meta["variable_type"].astype(str).str.lower()
    target_variable_names = df_meta.loc[vt == "target", "variable_name"].astype(str).tolist()
    if len(target_variable_names) != n_variates:
        target_variable_names = [f"target_{j}" for j in range(n_variates)]

    covariate_variates: List[List[float]] | None = None
    covariate_variable_names: List[str] | None = None
    if ds.covariate is not None:
        cov_arr = np.asarray(ds.covariate, dtype=float)
        if cov_arr.ndim != 2:
            raise ValueError(
                f"Expected covariate (n_steps, n_covariates), got {cov_arr.shape}"
            )
        covariate_variates = [cov_arr[:, j].tolist() for j in range(cov_arr.shape[1])]
        if cap < n_steps_full:
            covariate_variates = [row[:cap] for row in covariate_variates]
        names_series = df_meta.loc[vt == "covariate", "variable_name"].astype(str)
        covariate_variable_names = names_series.tolist()
        if len(covariate_variable_names) != len(covariate_variates):
            covariate_variable_names = [
                f"covariate_{j}" for j in range(len(covariate_variates))
            ]

    freq = str(meta.get("time_freq", "unknown"))
    note = (
        "Series from Python DataLoader + Preprocessor with target normalization disabled; "
        "missing-value handling matches training. Compare to denormalized BQ predictions in original units."
    )
    if cap < n_steps_full:
        note += (
            f" Plot JSON keeps the first {cap} of {n_steps_full} steps "
            f"(context_window + forecast_horizon span through benchmark window index "
            f"{_PLOT_BENCHMARK_MAX_WINDOW_INDEX}); "
            "use n_steps_full with run plot alignment."
        )

    out: Dict[str, Any] = {
        "task_id": catalog_task_id,
        "timestamps": unix_ts,
        "variates": variates,
        "context_window": int(task_config.context_window),
        "forecast_horizon": int(task_config.forecast_horizon),
        "freq": freq,
        "n_variates": int(n_variates),
        "n_steps": int(n_steps),
        "n_steps_full": n_steps_full,
        "dataset_normalize": bool(task_config.dataset.normalize),
        "dataset_handle_missing": str(task_config.dataset.handle_missing),
        "plot_preprocessing_note": note,
        "target_variable_names": target_variable_names,
    }
    if covariate_variates is not None and len(covariate_variates) > 0:
        out["covariate_variates"] = covariate_variates
        out["n_covariates"] = len(covariate_variates)
        if covariate_variable_names is not None:
            out["covariate_variable_names"] = covariate_variable_names
    return out
