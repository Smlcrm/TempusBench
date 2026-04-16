"""
Hyperparameter tuning for time series forecasting models.

This module provides the HyperparameterTuner class which performs rolling-window
hyperparameter optimization using a grid search approach across multiple validation
windows. It evaluates each hyperparameter combination and selects the best performing
parameters based on the specified tuning loss metric.
"""

import copy
import csv
import importlib.util
import math
import numbers
import os
import sys

from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from tempus_bench.utils.utils import compute_point_forecast

from ..utils.configs import JobConfig
from ..utils.log_manager import LogManager
from ..utils.visualizer import Visualizer
from ..utils.paths import get_task_path
from .data_loader import DataLoader
from .model_executor import ModelExecutor
from .metric_registry import MetricRegistry


def _finite_scalar_metric_for_bq(v: object) -> float | None:
    """Pick scalar finite floats for BigQuery ``error_metrics`` (NumPy / 0-d tensor-safe)."""
    if v is None or isinstance(v, (str, bytes, dict, list, tuple, set)):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, np.bool_):
        return None
    try:
        if isinstance(v, np.ndarray):
            if v.size != 1:
                return None
            x = float(v.reshape(-1)[0])
        elif isinstance(v, np.generic):
            x = float(v)
        elif isinstance(v, numbers.Real):
            x = float(v)
        else:
            # PyTorch / JAX-style 0-d scalars often expose .item(); recurse to native float.
            item_fn = getattr(v, "item", None)
            if callable(item_fn):
                try:
                    extracted = item_fn()
                except Exception:
                    return None
                if extracted is v:
                    return None
                return _finite_scalar_metric_for_bq(extracted)
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(x):
        return x
    return None


def _metrics_dict_for_bq(eval_outputs: dict) -> dict[str, float]:
    """Convert ``eval_outputs`` metric values to finite floats for BigQuery ``error_metrics``.

    ``compute_metrics`` almost always returns NumPy scalars (and sometimes 0-d torch tensors).
    The old ``isinstance(v, (int, float))`` filter dropped those, leaving an empty map so
    ``append_tempus_window`` often had nothing to buffer (successful runs with no
    ``tempus_benchmarks`` rows). Applies to stochastic and deterministic foundation models alike.
    """
    out: dict[str, float] = {}
    for k, v in eval_outputs.items():
        f = _finite_scalar_metric_for_bq(v)
        if f is not None:
            out[str(k)] = f
    return out


def _task_id_for_results_sink(task_config) -> str:
    """
    Task key for external sinks (e.g. BigQuery ``task_id`` / Firestore execution_plan).

    Uses the path segment after a ``tasks`` directory (e.g. ``univariate/foo``) so it
    matches ``execution_plan[].tasks[]`` strings; falls back to folder name.
    """
    parts = Path(task_config.task_path).parts
    for i, seg in enumerate(parts):
        if seg.lower() == "tasks" and i + 1 < len(parts):
            return "/".join(parts[i + 1 :])
    return task_config.task_name


class HyperparameterTuner:
    """
    Performs rolling-window hyperparameter sweeps for a task-model pair.

    The tuner evaluates hyperparameter combinations across rolling validation windows,
    selects optimal parameters based on tuning loss, generates visualizations, and
    aggregates cross-window evaluation metrics.

    Attributes:
        job_config (JobConfig): Complete job configuration including task and model settings.
        evaluation_config (EvaluationConfig): Evaluation-specific configuration.
        evaluation_setting (EvaluationSetting): System-wide evaluation settings.
        model_config (ModelConfig): Model-specific configuration and hyperparameter grid.
        model_setting (dict): Model execution settings (device, Python version, etc.).
        task_config (TaskConfig): Task-specific configuration including dataset metadata.
        model_name (str): Name of the model being tuned.
        tuning_loss (str): Loss metric used for hyperparameter selection.
        dataset_path (Path): Path to the dataset directory.
        dataset_file_path (Path): Path to the dataset CSV file.
        logger (LoggerManager): Logger instance for logging operations.
        data_loader (DataLoader): Data loader instance for accessing dataset windows.
    """

    def __init__(
        self,
        job_config: JobConfig,
        job_id: str | None = None,
        results_callback: Callable[..., None] | None = None,
    ):
        """
        Initialize tuner with job configuration.

        Args:
            job_config: Fully validated `JobConfig` produced by `ConfigManager.generate_run_configs`.
                Provides benchmark settings, task metadata, and model hyperparameter grid.
            job_id: Optional run identifier for result sinks (passed to ``results_callback``).
            results_callback: Optional callback(job_id, model, task, window_idx, metrics, forecast_data?)
                for per-window metrics/forecasts (optional persistence layer).
        """
        self.job_config = job_config
        self.job_id = job_id
        self.results_callback = results_callback
        self.evaluation_config = job_config.evaluation_config
        self.evaluation_setting = job_config.evaluation_setting
        self.model_config = job_config.model_config
        self.model_setting = job_config.model_setting
        self.task_config = job_config.task_config

        self.model_name = job_config.model_config.model_name
        self.tuning_loss = self.evaluation_config.tuning_loss
        self.dataset_path = Path(self.task_config.task_path)
        self.dataset_file_path = self.dataset_path / self.task_config.dataset.file_name
        self.visualizer = Visualizer()

    def _generate_hyperparameter_grid(self) -> List[dict]:
        """
        Generate the Cartesian product of the configured hyperparameter search space.

        This method creates all possible combinations of hyperparameter values from
        the model configuration, where each hyperparameter can have multiple candidate
        values defined as a list.

        Returns:
            List[dict]: List of dictionaries, where each dictionary maps hyperparameter
                names to concrete values for the single model defined in the active job.
                Each dictionary represents one hyperparameter combination to evaluate.

        Raises:
            ValueError: If more than one model is defined for the job (the tuner only
                supports a single model).
        """

        model_name = self.model_name

        # Build grid from config directly without constructing the model
        params_space = {
            k: v for k, v in self.model_config.__dict__.items() if k != "model_name"
        }

        keys = list(params_space.keys())
        values_lists = [params_space[k] for k in keys]

        grid: list[dict] = []

        for combo in product(*values_lists):

            yield dict(zip(keys, combo))

        return grid

    def optimize_hyperparameters(
        self, context_steps: int, train_steps: int, validate_steps: int
    ) -> Tuple[dict, dict]:
        """
        Evaluate every hyperparameter combination on rolling windows and select the best.

        The tuner iterates over the dataset using context/train/validate windows, executes
        the model for each hyperparameter configuration, logs metrics, generates visualizations,
        and aggregates cross-window statistics. The best hyperparameters are selected based on
        the tuning loss metric averaged across validation windows.

        Args:
            context_steps (int): Number of historical context points supplied to the model.
            train_steps (int): Number of points used for the training/fit segment within
                each window.
            validate_steps (int): Number of points reserved for evaluation within each window.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - First dict: Nested dictionary keyed by model name then dataset path,
                  containing averaged evaluation metrics across windows.
                - Second dict: Nested dictionary keyed by model name then dataset path,
                  containing ordered lists of best hyperparameter assignments for each window.
        """
        print("Tuner start")
        all_evals = {}
        best_hyperparameters = {}

        # When the cloud worker sets job_id + results_callback, every evaluated window must
        # reach the sink; clearing the callback mid-loop used to complete "successfully" with
        # partial BQ data.
        sink_enforced = bool(self.job_id and self.results_callback)

        # Initialize model executor
        model_executor = ModelExecutor(job_config=self.job_config.to_dict())

        LogManager.get_logger().info(
            "HyperparameterTuner",
            f"=================== Starting hyperparameter optimization for {self.model_name} ===================",
        )

        # Store results for each window
        window_results = []
        optimal_hyperparameters = []
        evaluations = []

        tuning_losses = {}
        eval_metrics = {}
        evaluation_metrics = None

        y_true = []
        y_pred = []
        timestamps_pred = []
        # Try each hyperparameter combination
        _grid_list = list(self._generate_hyperparameter_grid())
        is_foundation_single_run = len(_grid_list) == 1 and _grid_list[0] == {}
        if is_foundation_single_run:
            LogManager.get_logger().info(
                "HyperparameterTuner",
                f"Running {self.model_name} (foundation model, no tuning). "
                "First run may take several minutes to load/download model weights.",
            )
        # tqdm: (1) std `tqdm` only — `tqdm.auto` can pick the notebook backend, which may
        # ignore `file=` and still write to stderr. (2) On Batch/COS, stdout is not a TTY;
        # disable the bar entirely so no bytes hit stderr (Batch task logs label stderr as ERROR).
        # Set TEMPUSBENCH_FORCE_TQDM=1 to show the bar in non-interactive runs.
        _force_tqdm = os.environ.get("TEMPUSBENCH_FORCE_TQDM", "").strip() == "1"
        _show_hp_bar = _force_tqdm or sys.stdout.isatty()
        last_trial_error: Optional[BaseException] = None
        for params in tqdm(
            _grid_list,
            desc="Hyperparameter Combinations",
            file=sys.stdout,
            mininterval=1.0,
            disable=not _show_hp_bar,
        ):
            try:
                # Execute model with these hyperparameters
                windows_eval_outputs = model_executor.execute_model(
                    hyperparameters=params,
                    context_steps=context_steps,
                    train_steps=train_steps,
                    validate_steps=validate_steps,
                )

            except Exception as e:
                last_trial_error = e
                LogManager.get_logger().error(
                    "HyperparameterTuner",
                    f"Error executing model {self.model_name} with params {params}: {e}",
                )
                continue
            for window_idx, eval_outputs in enumerate(windows_eval_outputs):
                if (
                    self.job_id
                    and sink_enforced
                    and self.results_callback is None
                ):
                    raise RuntimeError(
                        "results_callback mismatch: job_id is set but results_callback was "
                        "cleared or unset before every rolling window was exported to the sink"
                    )
                #print("eval outpus", eval_outputs)
                immutable_params = tuple(sorted(params.items()))
                # Set evaluation metrics list on first successful eval

                tuning_losses[immutable_params] = eval_outputs[self.tuning_loss]
                y_true.append(eval_outputs.pop("y_true"))
                y_pred.append(eval_outputs.pop("y_pred"))
                timestamps_pred.append(eval_outputs.pop("timestamps_pred"))
                context_values = eval_outputs.pop("context_values", None)
                context_timestamps = eval_outputs.pop("context_timestamps", None)
                eval_outputs.pop("forecast_validation", None)  # exclude from metric aggregation

                # Optional per-window results sink (metrics + forecast series)
                if self.results_callback and self.job_id:
                    metrics_for_bq = _metrics_dict_for_bq(eval_outputs)
                    y_pred_arr = np.array(y_pred[window_idx])
                    point_pred = compute_point_forecast(
                        y_pred_arr,
                        self.job_config.evaluation_config.point_forecast_statistic,
                    )
                    # BigQuery stores denormalized predictions/quantiles only; actuals come from task CSV in the app.
                    forecast_data = {
                        "y_pred": point_pred,
                        "timestamps_pred": timestamps_pred[window_idx],
                    }
                    if context_values is not None and context_timestamps is not None:
                        forecast_data["context_values"] = context_values
                        forecast_data["context_timestamps"] = context_timestamps
                    # p10, p50, p90 for stochastic models (y_pred is 3D: samples x steps x targets)
                    if self.model_setting.get("model_type") == "stochastic" and y_pred_arr.ndim == 3:
                        forecast_data["p10"] = np.percentile(y_pred_arr, 10, axis=0)
                        forecast_data["p50"] = np.percentile(y_pred_arr, 50, axis=0)
                        forecast_data["p90"] = np.percentile(y_pred_arr, 90, axis=0)
                    self.results_callback(
                        self.job_id,
                        self.model_name,
                        _task_id_for_results_sink(self.task_config),
                        window_idx,
                        metrics_for_bq,
                        forecast_data=forecast_data,
                    )

                if evaluation_metrics is None:
                    evaluation_metrics = list(eval_outputs.keys())

                eval_metrics[immutable_params] = eval_outputs

                # Log hyperparameters and metrics to TensorBoard (HParams: one sub-run per trial)
                LogManager.get_logger().log_hparams(
                    params,
                    eval_outputs,
                    model_name=self.model_name,
                    task_name=self.task_config.task_name,
                    window_idx=window_idx,
                )

                # Find the hyperparams with lowest tuning_loss for this window
                best_params = min(tuning_losses, key=lambda k: tuning_losses[k])
                optimal_hyperparameters.append(best_params)
                # Snapshot per window (eval_metrics is reused; shallow copy would alias inner dicts)
                evaluations.append(copy.deepcopy(eval_metrics))

                self.visualizer.plot_forecast_window(
                    y_pred=compute_point_forecast(
                        np.array(y_pred[window_idx]),
                        self.job_config.evaluation_config.point_forecast_statistic,
                    ),
                    y_true=np.array(y_true[window_idx]),
                    timestamps_pred=np.array(timestamps_pred[window_idx]),
                    model_name=self.model_name,
                    task_name=self.task_config.task_name,
                    hyperparameters=dict(params),
                    window_idx=window_idx,
                )

        num_windows = len(optimal_hyperparameters)
        if evaluation_metrics is None or num_windows == 0:
            parts = [
                f"No successful evaluation windows for model {self.model_name} on task "
                f"{self.task_config.task_name!r} (task_path={self.task_config.task_path!r}).",
                "Hint: ModelExecutor may have returned an empty window list, the series may be "
                "shorter than context+train+validate, max_windows may be too small for the "
                "series length, or every hyperparameter trial failed.",
            ]
            if last_trial_error is not None:
                parts.append(f"Last trial error: {last_trial_error}")
            LogManager.get_logger().error("HyperparameterTuner", " ".join(parts))
            return {}, {}

        # Aggregate test loss over windows, for each metric.
        # With 2+ windows: use hyperparams chosen on window j to score metrics on window j+1.
        # With 1 window: there is no next-window holdout — report that window's metrics directly.
        test_loss = {metric: [] for metric in evaluation_metrics}  # type: ignore
        if num_windows >= 2:
            for window_j in range(num_windows - 1):
                best_params_prev = optimal_hyperparameters[window_j]
                for metric in evaluation_metrics:  # type: ignore
                    if best_params_prev in evaluations[window_j + 1]:
                        test_loss[metric].append(
                            evaluations[window_j + 1][best_params_prev][metric]
                        )
        elif num_windows == 1:
            bp = optimal_hyperparameters[0]
            ev0 = evaluations[0]
            if bp in ev0:
                for metric in evaluation_metrics:  # type: ignore
                    test_loss[metric].append(float(ev0[bp][metric]))

        avg_test_loss = {
            metric: (
                float(np.mean(test_loss[metric])) if test_loss[metric] else float("nan")
            )
            for metric in evaluation_metrics  # type: ignore
        }

        # Write to evaluations CSV in parent directory
        csv_filename = f"evaluations.csv"
        evals_dir = Path(self.job_config.run_path) / "evals"
        evals_dir.mkdir(exist_ok=True)
        csv_outpath = evals_dir / csv_filename
        file_exists = csv_outpath.exists()

        # Obtain all possible metrics from MetricRegistry, if possible
        metrics_registry = MetricRegistry()
        all_metric_names = sorted(list(metrics_registry.metric_registry.keys()))
        deterministic_metric_names = sorted(
            list(metrics_registry.deterministic_metrics)
        )
        stochastic_metric_names = sorted(list(metrics_registry.stochastic_metrics))

        # Determine model type (fallback to 'deterministic' if not present)
        model_type = self.model_setting["model_type"]

        # Prepare the row to write: 'NA' for metrics not available for this model type
        formatted_row = [self.model_name, self.task_config.task_name]

        for metric in all_metric_names:
            # Determine if this metric applies to this model type
            metric_applies = False
            if model_type == "deterministic":
                # For deterministic models, only include deterministic metrics
                metric_applies = metric in deterministic_metric_names
            else:
                # For non-deterministic models, include both stochastic and deterministic metrics
                metric_applies = metric in (
                    stochastic_metric_names + deterministic_metric_names
                )

            if metric_applies:
                # Get metric value if available, otherwise use 'NA'
                value = avg_test_loss.get(metric, "NA")
                # Convert nan/inf to "NA" for consistent CSV formatting
                # Handle both Python float and numpy float types
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        value = "NA"
                formatted_row.append(str(value))
            else:
                # Metric doesn't apply to this model type
                formatted_row.append("NA")

        formatted_row.append(str(optimal_hyperparameters))

        # Append a new line to evaluations.csv if it already exists
        with open(csv_outpath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:  # write header on the first line
                writer.writerow(
                    ["model_name", "task_name"]
                    + [f"avg_test_{metric}" for metric in all_metric_names]
                    + ["best_params"]
                )
            writer.writerow(formatted_row)

        # Store results for this dataset
        model_evals = {}
        model_best_params = {}
        model_evals[self.dataset_path] = avg_test_loss
        model_best_params[self.dataset_path] = optimal_hyperparameters

        # Store results for this model
        all_evals[self.model_name] = model_evals
        best_hyperparameters[self.model_name] = model_best_params

        LogManager.get_logger().success(
            "HyperparameterTuner", "Hyperparameter optimization completed"
        )

        return all_evals, best_hyperparameters
