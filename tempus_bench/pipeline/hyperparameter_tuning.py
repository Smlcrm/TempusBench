"""
Hyperparameter tuning for time series forecasting models.

This module provides the HyperparameterTuner class which performs rolling-window
hyperparameter optimization using a grid search approach across multiple validation
windows. It evaluates each hyperparameter combination and selects the best performing
parameters based on the specified tuning loss metric.
"""

import csv
import importlib.util

from itertools import product
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..utils.configs import JobConfig
from ..utils.log_manager import LogManager
from ..utils.visualizer import Visualizer
from ..utils.paths import get_task_path
from .data_loader import DataLoader
from .model_executor import ModelExecutor


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

    def __init__(self, job_config: JobConfig):
        """
        Initialize tuner with job configuration.

        Args:
            job_config: Fully validated `JobConfig` produced by `ConfigManager.generate_run_configs`.
                Provides benchmark settings, task metadata, and model hyperparameter grid.
        """
        self.job_config = job_config
        self.evaluation_config = job_config.evaluation_config
        self.evaluation_setting = job_config.evaluation_setting
        self.model_config = job_config.model_config
        self.model_setting = job_config.model_setting
        self.task_config = job_config.task_config

        self.model_name = job_config.model_config.model_name
        self.tuning_loss = self.evaluation_config.tuning_loss
        self.dataset_path = Path(self.task_config.task_path)
        self.dataset_file_path = self.dataset_path / self.task_config.dataset.file_name
        self.data_loader = DataLoader(self.task_config, self.evaluation_config)
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
            grid.append(dict(zip(keys, combo)))

        LogManager.get_logger().info(
            "HyperparameterTuner",
            f"Generated hyperparameter grid for {model_name}: number of combinations = {len(grid)}",
        )

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
        all_evals = {}
        best_hyperparameters = {}

        # Initialize model executor
        model_executor = ModelExecutor(self.job_config)
        # Generate windows for this dataset

        # Store results for each window
        window_results = []
        optimal_hyperparameters = []
        evaluations = []

        tuning_losses = {}
        eval_metrics = {}
        evaluation_metrics = None

        # Try each hyperparameter combination
        for params in self._generate_hyperparameter_grid():

            try:
                # Execute model with these hyperparameters
                windows_eval_losses = model_executor.execute_model(
                    hyperparameters=params,
                    context_steps=context_steps,
                    train_steps=train_steps,
                    validate_steps=validate_steps,
                )

            except Exception as e:
                LogManager.get_logger().error(
                    "HyperparameterTuner",
                    f"Error executing model {self.model_name} with params {params}: {e}",
                )
                continue

            for window_idx, eval_losses in enumerate(windows_eval_losses):
                if evaluation_metrics is None:
                    evaluation_metrics = list(eval_losses.keys())

                immutable_params = tuple(sorted(params.items()))
                # Set evaluation metrics list on first successful eval

                tuning_losses[immutable_params] = eval_losses[self.tuning_loss]
                eval_metrics[immutable_params] = eval_losses

                LogManager.get_logger().debug(
                    "HyperparameterTuner",
                    f"Evaluated model {self.model_name} with params {params}: {eval_losses}",
                )
                # Log hyperparameters and metrics to TensorBoard
                LogManager.get_logger().log_hparams(params, eval_losses)

                # Find the hyperparams with lowest tuning_loss for this window
                best_params = min(tuning_losses, key=lambda k: tuning_losses[k])
                optimal_hyperparameters.append(best_params)
                evaluations.append(eval_metrics)

                # Generate forecast plot for best hyperparameters
                self.visualizer.plot_forecast_window(
                    y_pred=eval_losses["y_pred"],
                    y_true=eval_losses["y_true"],
                    timestamps_pred=eval_losses["timestamps_pred"],
                    model_name=self.model_name,
                    hyperparameters=dict(best_params),
                    window_idx=window_idx
                )

        num_windows = len(optimal_hyperparameters)
        # Aggregate test loss over all windows, for each metric
        test_loss = {metric: [] for metric in evaluation_metrics}  # type: ignore
        for window_j in range(num_windows - 1):
            best_params_prev = optimal_hyperparameters[window_j]
            for metric in evaluation_metrics:  # type: ignore
                if best_params_prev in evaluations[window_j + 1]:
                    test_loss[metric].append(
                        evaluations[window_j + 1][best_params_prev][metric]
                    )

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
        row = (
            [self.model_name, self.dataset_path]
            + [avg_test_loss[metric] for metric in evaluation_metrics]  # type: ignore
            + [str(optimal_hyperparameters)]
        )
        # Append a new line to evaluations.csv if it already exists
        with open(csv_outpath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:  # write header on the first line
                writer.writerow(
                    ["model_name", "dataset_path"]
                    + [f"avg_test_{metric}" for metric in evaluation_metrics]  # type: ignore
                    + ["best_params"]
                )
            writer.writerow(row)

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

