import csv
import importlib.util

from itertools import product
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config.configs import JobConfig
from ..models.model_router import ModelRouter
from ..utils.paths import get_task_path
from ..utils.tf_logger import get_tf_logger
from .data_loader import DataLoader
from .model_executor import ModelExecutor


class HyperparameterTuner:
    """Run rolling-window hyperparameter sweeps for the active task/model pair."""

    def __init__(self, job_config: JobConfig):
        """
        Build a tuner that is pre-configured for a single job.

        Args:
            job_config: Fully validated `JobConfig` produced by `ConfigManager.generate_run_configs`.
                Provides benchmark settings, task metadata, and model hyperparameter grid.
        """
        self.job_config = job_config
        self.config = job_config.benchmark_config
        self.benchmark_settings = job_config.benchmark_settings
        self.model_settings = job_config.model_settings
        self.task_config = job_config.task_config
        self.eval_config = self.config.evaluation
        self.model_name, self.model_config = next(
            iter(self.config.model.model_dump(exclude_none=True).items())
        )
        self.tuning_loss = self.eval_config.tuning_loss
        self.dataset_path = job_config.task_paths[self.task_config.name]
        self.dataset_file_path = self.dataset_path + self.task_config.dataset.file_name
        self.logger = job_config.logger
        self.tf_logger = get_tf_logger()
        self.data_loader = DataLoader(job_config)

    def _generate_hyperparameter_grid(self) -> List[dict]:
        """
        Materialize the Cartesian product of the configured hyperparameter search space.

        Returns:
            List[dict]: Each element is a dict mapping hyperparameter names to concrete values
                for the single model defined in the active job.

        Raises:
            ValueError: If more than one model is defined for the job (the tuner only supports a single model).
        """
        model_map = self.config.model.model_dump(exclude_none=True)
        if len(model_map) > 1:
            raise ValueError(
                "Hyperparameter tuning is not supported for multiple models"
            )

        model_name = list(model_map.keys())[0]

        # Route resolves model location, but we don't need to import the class to build the grid
        router = ModelRouter(logger=self.logger)
        _ = router.get_model_path_by_model_type(model_name, "deterministic")
        # Build grid from config directly without constructing the model
        params_space = model_map[model_name]
        keys = list(params_space.keys())
        values_lists = [params_space[k] for k in keys]
        grid: list[dict] = []
        for combo in product(*values_lists):
            grid.append(dict(zip(keys, combo)))
        self.logger.info(
            "HyperparameterTuner",
            f"Generated hyperparameter grid for {model_name}: number of combinations = {len(grid)}",
        )
        return grid

    def optimize_hyperparameters(
        self, context_steps: int, train_steps: int, validate_steps: int
    ) -> Tuple[dict, dict]:
        """
        Evaluate every hyperparameter combination on rolling windows and pick the best performer.

        The tuner iterates over the dataset using context/train/validate windows, executes the model
        for each hyperparameter configuration, logs metrics, generates visualizations, and aggregates
        cross-window statistics.

        Args:
            context_steps: Number of historical context points supplied to the model.
            train_steps: Number of points used for the training/fit segment within each window.
            validate_steps: Number of points reserved for evaluation within each window.

        Returns:
            Tuple[dict, dict]: Two nested dictionaries keyed by model name then dataset path.
            The first contains averaged evaluation metrics, the second stores the ordered list
            of window-level best hyperparameter assignments.
        """
        all_evals = {}
        best_hyperparameters = {}

        # Initialize model executor
        model_executor = ModelExecutor(
            model_settings=self.model_settings,
            logger=self.logger,
        )
        # Generate windows for this dataset
        steps = [
            ("context", context_steps),
            ("train", train_steps),
            ("validate", validate_steps),
        ]
        window_generator = self.data_loader.generate_dataset_split(
            dataset_path=self.dataset_path, steps=steps, stride=validate_steps
        )

        # Store results for each window
        window_results = []
        optimal_hyperparameters = []
        evaluations = []
        num_windows = 0

        # For each rolling window
        for window_idx, dataset in window_generator:
            self.logger.debug(
                "HyperparameterTuner",
                f"Processing window {window_idx} for dataset {self.task_config.dataset.file_name}",
            )

            tuning_losses = {}
            eval_metrics = {}
            evaluation_metrics = None

            # Try each hyperparameter combination
            for params in self._generate_hyperparameter_grid():
                try:
                    # Execute model with these hyperparameters
                    eval_losses = model_executor.execute_model(
                        model_name=self.model_name,
                        hyperparameters=params,
                        context_steps=context_steps,
                        train_steps=train_steps,
                        validate_steps=validate_steps,
                        task_path=self.dataset_path,
                        window_idx=window_idx,
                    )

                    immutable_params = tuple(sorted(params.items()))
                    # Set evaluation metrics list on first successful eval
                    if evaluation_metrics is None:
                        evaluation_metrics = list(eval_losses.keys())
                    tuning_losses[immutable_params] = eval_losses[self.tuning_loss]
                    eval_metrics[immutable_params] = eval_losses

                    self.logger.debug(
                        "HyperparameterTuner",
                        f"Evaluated model {self.model_name} with params {params}: {eval_losses}",
                    )
                    # Log hyperparameters and metrics to TensorBoard
                    self.tf_logger.log_hparams(params, eval_losses)

                except Exception as e:
                    if self.benchmark_settings.console_logging:
                        self.logger.error(
                            "HyperparameterTuner",
                            f"Error executing model {self.model_name} with params {params}: {e}",
                        )
                    continue

            # Find the hyperparams with lowest tuning_loss for this window
            if tuning_losses:
                best_params = min(tuning_losses, key=lambda k: tuning_losses[k])
                optimal_hyperparameters.append(best_params)
                evaluations.append(eval_metrics)
                num_windows += 1

                # Generate forecast plot for best hyperparameters
                self._generate_forecast_plot(
                    model_name=self.model_name,
                    hyperparameters=dict(best_params),
                    context_steps=context_steps,
                    train_steps=train_steps,
                    validate_steps=validate_steps,
                    dataset_path=self.dataset_path,
                    window_idx=window_idx,
                )

        if num_windows == 0:
            if self.benchmark_settings.console_logging:
                self.logger.warning(
                    "HyperparameterTuner",
                    f"No valid windows for dataset {self.dataset_path}",
                )
            return {}, {}

        # Aggregate test loss over all windows, for each metric
        test_loss = {metric: [] for metric in evaluation_metrics}
        for window_j in range(num_windows - 1):
            best_params_prev = optimal_hyperparameters[window_j]
            for metric in evaluation_metrics:
                if best_params_prev in evaluations[window_j + 1]:
                    test_loss[metric].append(
                        evaluations[window_j + 1][best_params_prev][metric]
                    )

        avg_test_loss = {
            metric: (
                float(np.mean(test_loss[metric])) if test_loss[metric] else float("nan")
            )
            for metric in evaluation_metrics
        }

        # Write to evaluations CSV in parent directory
        csv_filename = f"evaluations.csv"
        evals_dir = Path(self.job_config.run_path) / "evals"
        evals_dir.mkdir(exist_ok=True)
        csv_outpath = evals_dir / csv_filename
        file_exists = csv_outpath.exists()
        row = (
            [self.model_name, self.dataset_path]
            + [avg_test_loss[metric] for metric in evaluation_metrics]
            + [str(optimal_hyperparameters)]
        )
        # Append a new line to evaluations.csv if it already exists
        with open(csv_outpath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:  # write header on the first line
                writer.writerow(
                    ["model_name", "dataset_path"]
                    + [f"avg_test_{metric}" for metric in evaluation_metrics]
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

        if self.benchmark_settings.console_logging:
            self.logger.success(
                "HyperparameterTuner", "Hyperparameter optimization completed"
            )

        return all_evals, best_hyperparameters

    def _generate_forecast_plot(
        self,
        model_name,
        hyperparameters,
        context_steps,
        train_steps,
        validate_steps,
        dataset_path,
        window_idx,
    ):
        """
        Produce and log a time-series plot contrasting predictions with the underlying window data.

        The plot contains the concatenated context/train/validation segments alongside the model
        forecasts, and is written to disk and TensorBoard for later inspection.
        """
        try:
            import matplotlib.pyplot as plt

            # Create data loader to get the window data
            data_loader = DataLoader(self.job_config)

            # Get the specific window data
            steps = [
                ("context", context_steps),
                ("train", train_steps),
                ("validate", validate_steps),
            ]
            window_iter = data_loader.generate_dataset_split(
                dataset_path, steps, stride=1
            )

            # Find the specific window
            window_data = None
            for idx, window in window_iter:
                if idx == window_idx:
                    window_data = window
                    break

            if window_data is None:
                self.logger.warning(
                    "HyperparameterTuner", f"Window {window_idx} not found for plotting"
                )
                return

            # Create model executor to get predictions
            model_executor = ModelExecutor(
                model_settings=self.model_settings,
                logger=self.logger,
            )

            # Execute model to get predictions
            eval_results = model_executor.execute_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                context_steps=context_steps,
                train_steps=train_steps,
                validate_steps=validate_steps,
                task_path=dataset_path,
                window_idx=window_idx,
            )

            # Create plots directory
            plots_dir = (
                Path(self.job_config.run_path) / "tensorboard" / "plots" / model_name
            )
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Get data from window using indices
            context_data = window_data.target[
                window_data.context.start : window_data.context.end
            ]
            train_data = window_data.target[
                window_data.train.start : window_data.train.end
            ]
            validate_data = window_data.target[
                window_data.validate.start : window_data.validate.end
            ]

            # Extract actual predictions from results
            predictions = np.array(eval_results.get("predictions"))
            y_true_validate = np.array(eval_results.get("y_true"))

            # Create subplots for each target
            num_targets = context_data.shape[1] if context_data.ndim > 1 else 1
            fig, axes = plt.subplots(num_targets, 1, figsize=(15, 4 * num_targets))
            if num_targets == 1:
                axes = [axes]

            # Create continuous time indices for smooth plotting
            context_len = len(context_data)
            train_len = len(train_data)
            validate_len = len(validate_data)

            # Create continuous time series by concatenating all segments
            if context_data.ndim == 1:
                full_data = np.concatenate([context_data, train_data, validate_data])
            else:
                full_data = np.concatenate(
                    [context_data, train_data, validate_data], axis=0
                )

            # Create continuous time indices
            full_time = np.arange(len(full_data))
            context_time = full_time[:context_len]
            train_time = full_time[context_len : context_len + train_len]
            validate_time = full_time[context_len + train_len :]

            # For each target, create a subplot
            for target_idx in range(num_targets):
                ax = axes[target_idx]

                # Plot context data
                if context_data.ndim == 1:
                    ax.plot(
                        context_time,
                        context_data,
                        "b-",
                        label="Context",
                        linewidth=2,
                        alpha=0.8,
                    )
                else:
                    ax.plot(
                        context_time,
                        context_data[:, target_idx],
                        "b-",
                        label="Context",
                        linewidth=2,
                        alpha=0.8,
                    )

                # Plot training data
                if train_data.ndim == 1:
                    ax.plot(
                        train_time,
                        train_data,
                        "g-",
                        label="Train",
                        linewidth=2,
                        alpha=0.8,
                    )
                else:
                    ax.plot(
                        train_time,
                        train_data[:, target_idx],
                        "g-",
                        label="Train",
                        linewidth=2,
                        alpha=0.8,
                    )

                # Plot validation data (true values)
                if validate_data.ndim == 1:
                    ax.plot(
                        validate_time,
                        validate_data,
                        "r-",
                        label="True Values",
                        linewidth=2,
                        alpha=0.8,
                    )
                else:
                    ax.plot(
                        validate_time,
                        validate_data[:, target_idx],
                        "r-",
                        label="True Values",
                        linewidth=2,
                        alpha=0.8,
                    )

                # Plot actual model predictions
                if len(predictions) > 0:
                    if predictions.ndim == 1:
                        ax.plot(
                            validate_time,
                            predictions,
                            "orange",
                            linestyle="--",
                            label="Model Predictions",
                            linewidth=2,
                            alpha=0.8,
                        )
                    else:
                        ax.plot(
                            validate_time,
                            predictions[:, target_idx],
                            "orange",
                            linestyle="--",
                            label="Model Predictions",
                            linewidth=2,
                            alpha=0.8,
                        )
                else:
                    # Fallback to simple prediction if no actual predictions available
                    if validate_data.ndim == 1:
                        last_train_val = (
                            train_data[-1] if len(train_data) > 0 else context_data[-1]
                        )
                        fallback_predictions = np.full_like(
                            validate_data, last_train_val
                        )
                        ax.plot(
                            validate_time,
                            fallback_predictions,
                            "orange",
                            linestyle="--",
                            label="Predictions (Fallback)",
                            linewidth=2,
                            alpha=0.8,
                        )
                    else:
                        last_train_vals = (
                            train_data[-1, :]
                            if len(train_data) > 0
                            else context_data[-1, :]
                        )
                        fallback_predictions = np.full_like(
                            validate_data, last_train_vals
                        )
                        ax.plot(
                            validate_time,
                            fallback_predictions[:, target_idx],
                            "orange",
                            linestyle="--",
                            label="Predictions (Fallback)",
                            linewidth=2,
                            alpha=0.8,
                        )

                # Customize subplot
                ax.set_title(
                    f"{model_name} - Target {target_idx + 1} (Window {window_idx})"
                )
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Add hyperparameters info to the figure
            fig.suptitle(
                f"Best Hyperparameters: {hyperparameters}", fontsize=12, y=0.98
            )

            # Save plot
            plot_path = plots_dir / f"window_{window_idx}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Log to TensorBoard
            self.tf_logger.log_image_file(
                image_path=str(plot_path), tag=f"{model_name}/forecast", step=window_idx
            )

            self.logger.info(
                "HyperparameterTuner",
                f"Generated time series plot for {model_name} window {window_idx}",
            )

        except Exception as e:
            self.logger.error(
                "HyperparameterTuner",
                f"Error generating time series plot for {model_name}: {e}",
            )
