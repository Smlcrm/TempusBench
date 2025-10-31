import argparse
import datetime
import os

from pathlib import Path

from tempus_bench.config import Manager
from tempus_bench.pipeline.hyperparameter_tuning import HyperparameterTuner
from tempus_bench.utils.paths import get_project_root


class BenchmarkRunner:
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner with configuration.
        Args:
            config_path: Path to the config file used
        """
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(self.config_path))[0]

    def _initialize_run(self):
        """Initialize and update all path-related attributes."""

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
        """Execute the end-to-end benchmarking pipeline."""
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

        # Cleanup TensorBoard writer
        self.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarking pipeline with specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config YAML file. If not specified, uses the default config in tempus_bench/configs/all_models.yaml",
    )
    args = parser.parse_args()

    config_path = args.config

    # Run the Benchmarks
    runner = BenchmarkRunner(config_path=config_path)
    runner.run()
