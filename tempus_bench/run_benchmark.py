"""
Entry point for running the benchmarking pipeline.

This module provides the BenchmarkRunner class which orchestrates the end-to-end
benchmarking process including configuration loading, hyperparameter tuning, and
model evaluation across multiple task-model combinations.
"""

import sys

def _import_stage(label: str):
    """Telemetry: print before each import stage to find where mutex crash occurs."""
    print(f"[run_benchmark] import stage: {label}", flush=True)


_import_stage("stdlib")
import argparse
import datetime
import os
import pickle
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

# Telemetry: log uncaught Python exceptions to /tmp for the cloud worker (job_execution.run) to capture
def _telemetry_excepthook(exc_type, exc_value, exc_tb):
    try:
        log_path = Path("/tmp") / "tempusbench_run_benchmark_crash.log"
        with open(log_path, "a") as f:
            f.write(f"\n--- {datetime.datetime.utcnow().isoformat()}Z ---\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        print(f"[run_benchmark] uncaught exception logged to {log_path}", flush=True)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


if getattr(sys, "excepthook", None) == sys.__excepthook__:
    sys.excepthook = _telemetry_excepthook

_import_stage("ConfigManager")
from tempus_bench.utils.config_manager import ConfigManager
_import_stage("HyperparameterTuner")
from tempus_bench.pipeline.hyperparameter_tuner import HyperparameterTuner
_import_stage("DataLoader")
from tempus_bench.pipeline.data_loader import DataLoader
_import_stage("imports_done")


def _job_id_from_env_strict() -> Optional[str]:
    """Trimmed ``JOB_ID`` only — no ``RUN_ID`` or other aliases."""
    raw = (os.environ.get("JOB_ID") or "").strip()
    return raw if raw else None


def _gcp_project_from_env_strict() -> Optional[str]:
    """Trimmed ``GCP_PROJECT`` only — no ``GOOGLE_CLOUD_PROJECT`` or other aliases."""
    raw = (os.environ.get("GCP_PROJECT") or "").strip()
    return raw if raw else None


def _resolve_job_id_and_results_callback(
    results_callback: Optional[Callable[..., Any]],
) -> Tuple[Optional[str], Optional[Callable[..., Any]]]:
    """Resolve ``JOB_ID`` and whether per-window export may run (strict env names only).

    Uses ``JOB_ID`` and ``GCP_PROJECT`` only — no ``RUN_ID`` / ``GOOGLE_CLOUD_PROJECT``
    aliases. If any required variable is missing while a callback was supplied, logs and
    returns ``(job_id, None)`` so the benchmark run still completes without crashing.
    """
    job_id = _job_id_from_env_strict()
    if results_callback is None:
        return job_id, None
    if not job_id:
        print(
            "[BenchmarkRunner] results_callback ignored: JOB_ID not set",
            flush=True,
        )
        return job_id, None
    if not _gcp_project_from_env_strict():
        print(
            "[BenchmarkRunner] results_callback ignored: GCP_PROJECT not set",
            flush=True,
        )
        return job_id, None
    if os.environ.get("BQ_BUFFER_RESULTS") != "1":
        print(
            "[BenchmarkRunner] BQ_BUFFER_RESULTS is not '1'; per-window BigQuery export disabled",
            flush=True,
        )
        return job_id, None
    return job_id, results_callback


class BenchmarkRunner:
    """
    Orchestrates the end-to-end benchmarking pipeline execution.

    The BenchmarkRunner coordinates the execution of multiple benchmarking jobs,
    where each job represents a combination of a task (dataset) and model. It
    handles configuration loading, hyperparameter tuning, and result aggregation.

    Attributes:
        config_path (str): Path to the configuration YAML file.
        config_name (str): Name of the configuration file (without extension).
        manager (ConfigManager): Configuration manager instance.
        logger (LoggerManager): Logger instance for logging operations.
    """

    def __init__(
        self,
        config_path: str,
        *,
        results_callback: Callable[..., Any] | None = None,
        task_status_callback: Callable[..., Any] | None = None,
    ):
        """
        Initialize benchmark runner with configuration.

        Args:
            config_path (str): Path to the configuration YAML file used for
                this benchmark run.
            results_callback: Optional per-window hook
                ``(job_id, model, task, window_idx, metrics, forecast_data?)``.
                Injected by downstream runners that persist per-window metrics. Omit for plain local runs.
            task_status_callback: Optional ``(task_name, model_name, status, error=None)``
                hook called when each task starts / completes / fails. Downstream runners use
                this for per-task status tracking and buffer management.
        """
        self.config_path = config_path
        self.results_callback = results_callback
        self.task_status_callback = task_status_callback
        # Initialize the ConfigManager (which creates its own unified Logger with TensorBoard support)
        self.manager = ConfigManager(
            config_path=self.config_path,
        )

        # Get reference to logger from manager
        self.logger = self.manager.logger

        self.logger.debug(
            "BenchmarkRunner",
            f"Config path resolved to: {self.config_path}",
        )

    def __enter__(self):

        self._task_datasets_tmp = tempfile.TemporaryDirectory(
            prefix="tempusbench_task_datasets_"
        )
        self.temp_task_datasets_dir = Path(self._task_datasets_tmp.name)

        for task_name, task_config in self.manager.task_configs.items():

            task_path = self.temp_task_datasets_dir / f"{task_name}.pkl"
            data_loader = DataLoader(task_config, self.manager.evaluation_config)

            dataset = data_loader.dataset

            with open(task_path, "wb") as f:
                pickle.dump(dataset, f)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._task_datasets_tmp.cleanup()
        self.logger.close()

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""

        requested_callback = self.results_callback
        job_id, results_callback = _resolve_job_id_and_results_callback(requested_callback)
        if requested_callback is None:
            print(
                "[BenchmarkRunner] No results_callback; per-window export disabled.",
                flush=True,
            )
        elif results_callback is None:
            print(
                "[BenchmarkRunner] Per-window export not active (see messages above if any).",
                flush=True,
            )

        run_configs = list(self.manager.generate_run_configs())
        task_ds = str(self.temp_task_datasets_dir)
        task_cb = self.task_status_callback
        failed_tasks: list[str] = []

        for job_idx, job_config in enumerate(run_configs):
            task_name = job_config.task_config.task_name
            model_name = job_config.model_config.model_name
            try:
                if task_cb:
                    task_cb(task_name, model_name, "running")

                job_config.task_datasets_dir = task_ds
                hyperparameter_tuner = HyperparameterTuner(
                    job_config=job_config,
                    job_id=job_id,
                    results_callback=results_callback,
                )

                self.logger.info(
                    "BenchmarkRunner",
                    f"Hyperparameter Tuning Starts for job: {job_idx}, task: {task_name}",
                )

                task_config = job_config.task_config
                evals, hyperparameters = hyperparameter_tuner.optimize_hyperparameters(
                    context_steps=task_config.context_window,
                    train_steps=task_config.forecast_horizon,
                    validate_steps=task_config.forecast_horizon,
                )

                self.logger.success(
                    "BenchmarkRunner",
                    f"Hyperparameters Optimized for task: {task_name}",
                )
                self.logger.success(
                    "BenchmarkRunner",
                    f"Final Model Evaluation Executed for task: {task_name}",
                )
                if task_cb:
                    task_cb(task_name, model_name, "completed")

            except Exception as exc:
                failed_tasks.append(task_name)
                self.logger.error(
                    "BenchmarkRunner",
                    f"Task {task_name} failed: {exc}",
                )
                if task_cb:
                    task_cb(task_name, model_name, "failed", str(exc))

        # Logger lifecycle: close only in __exit__ (avoid double-close / TensorBoard errors).
        return failed_tasks


if __name__ == "__main__":
    _import_stage("__main__")
    parser = argparse.ArgumentParser(
        description="Run benchmarking pipeline with specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "benchmark.yaml"),
        help="Path to the config YAML file. If not specified, uses the default config in tempus_bench/configs/all_models.yaml",
    )
    args = parser.parse_args()

    config_path = args.config
    _import_stage("creating BenchmarkRunner")

    with BenchmarkRunner(config_path=config_path) as runner:
        runner.run()
