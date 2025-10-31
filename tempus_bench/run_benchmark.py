"""
Entry point for running the benchmarking pipeline.

This module provides the BenchmarkRunner class which orchestrates the end-to-end
benchmarking process including configuration loading, hyperparameter tuning, and
model evaluation across multiple task-model combinations.
"""

import argparse
import datetime
import os

from pathlib import Path

from tempus_bench.utils.manager import Manager
from tempus_bench.pipeline.hyperparameter_tuning import HyperparameterTuner
from tempus_bench.utils.paths import get_project_root


class BenchmarkRunner:
    """
    Orchestrates the end-to-end benchmarking pipeline execution.

    The BenchmarkRunner coordinates the execution of multiple benchmarking jobs,
    where each job represents a combination of a task (dataset) and model. It
    handles configuration loading, hyperparameter tuning, and result aggregation.

    Attributes:
        config_path (str): Path to the configuration YAML file.
        config_name (str): Name of the configuration file (without extension).
        manager (Manager): Configuration manager instance.
        logger (LoggerManager): Logger instance for logging operations.
    """

    def __init__(self, config_path: str):
        """
        Initialize benchmark runner with configuration.

        Args:
            config_path (str): Path to the configuration YAML file used for
                this benchmark run.
        """
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(self.config_path))[0]

    def _initialize_run(self):
        """
        Initialize the run by creating the manager and logger.

        This method initializes the Manager which handles configuration loading
        and creates a unified logger with TensorBoard support. The logger is then
        stored as an instance attribute for use throughout the benchmark execution.
        """
        # Initialize the Manager (which creates its own unified Logger with TensorBoard support)
        self.manager = Manager(
            config_path=self.config_path,
        )

        # Get reference to logger from manager
        self.logger = self.manager.logger

        # Emit early log lines so failures during Manager initialization are still captured

        self.logger.debug(
            "BenchmarkRunner",
            f"Config path resolved to: {self.config_path}",
        )

    def run(self):
        """
        Execute the end-to-end benchmarking pipeline.

        This method iterates through all job configurations (task-model combinations),
        performs hyperparameter tuning for each combination, and aggregates results.
        Each job configuration is processed sequentially, with hyperparameter optimization
        using rolling window validation.
        """
        self._initialize_run()

        # We execute multiple jobs per run, each with a different configuration (JobConfig).
        for job_idx, job_config in enumerate(self.manager.generate_run_configs()):
            hyperparameter_tuner = HyperparameterTuner(
                job_config=job_config, logger=self.logger
            )

            # Hyper-parameter Tuning
            task_name = job_config.task_config.name
            self.logger.info(
                "BenchmarkRunner",
                f"Hyperparameter Tuning Starts for job: {job_idx}, task: {task_name}",
            )

            # Hyperparameter Tuning - Context + Train + Validate Losses (Rolling Window with strides of validate_steps)
            task_config = job_config.task_config
            evals, hyperparameters = hyperparameter_tuner.optimize_hyperparameters(
                context_steps=task_config.context_window,
                train_steps=task_config.forecast_horizon,
                validate_steps=task_config.forecast_horizon,
            )

            self.logger.success(
                "BenchmarkRunner",
                f"Hyperparameters Optimized for task: {job_config.task_config.name}",
            )
            self.logger.success(
                "BenchmarkRunner",
                f"Final Model Evaluation Executed for task: {job_config.task_config.name}",
            )

        # Close logger
        self.logger.close()


if __name__ == "__main__":
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

    try:
        # Run the Benchmarks
        runner = BenchmarkRunner(config_path=config_path)
        runner.run()
    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close logger
        runner.logger.close()
