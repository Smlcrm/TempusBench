import argparse
import datetime
import os
import tempfile
import pickle

from pathlib import Path

from tempus_bench.utils.manager import Manager
from tempus_bench.pipeline.hyperparameter_tuning import HyperparameterTuner
from tempus_bench.utils.paths import get_project_root
from tempus_bench.pipeline.data_loader import DataLoader


class BenchmarkRunner:
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner with configuration.
        Args:
            config_path: Path to the config file used
        """
        self.config_path = config_path
        # Initialize the Manager (which creates its own unified Logger with TensorBoard support)
        self.manager = Manager(
            config_path=self.config_path,
        )

        # Get reference to logger from manager
        self.logger = self.manager.logger

        self.logger.debug(
            "BenchmarkRunner",
            f"Config path resolved to: {self.config_path}",
        )

    def __enter__(self):

        # Create the temporary directory at a specific path (e.g., under project root "temp")
        temp_root = Path(get_project_root()) / "temp_task_datasets"
        temp_root.mkdir(parents=True, exist_ok=True)

        self.temp_task_datasets_dir = tempfile.TemporaryDirectory(dir=str(temp_root))

        for task_name, task_config in self.manager.task_configs.items():

            task_path = Path(self.temp_task_datasets_dir.name) / task_name
            data_loader = DataLoader(task_config, self.manager.evaluation_config)

            dataset = data_loader.dataset

            with open(task_path, "wb") as f:
                pickle.dump(dataset, f)


        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.temp_task_datasets_dir.cleanup()
        self.logger.close()

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""

        # We execute multiple jobs per run, each with a different configuration (JobConfig).
        for job_idx, job_config in enumerate(self.manager.generate_run_configs()):
            hyperparameter_tuner = HyperparameterTuner(job_config=job_config)

            # Hyper-parameter Tuning
            task_name = job_config.task_config.task_name
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
                f"Hyperparameters Optimized for task: {job_config.task_config.task_name}",
            )
            self.logger.success(
                "BenchmarkRunner",
                f"Final Model Evaluation Executed for task: {job_config.task_config.task_name}",
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
        default="tempus_bench/config/benchmark.yaml",
        help="Path to the config YAML file. If not specified, uses the default config in tempus_bench/configs/all_models.yaml",
    )
    args = parser.parse_args()

    config_path = args.config

    with BenchmarkRunner(config_path=config_path) as runner:
        runner.run()
