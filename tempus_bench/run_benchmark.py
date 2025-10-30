import argparse
import datetime
import os

from pathlib import Path

from tempus_bench.config import Manager
from tempus_bench.pipeline.hyperparameter_tuning import HyperparameterTuner
from tempus_bench.utils.logger import Logger
from tempus_bench.utils.paths import get_configs_dir, get_project_root
from tempus_bench.utils.tf_logger import get_tf_logger


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
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.runs_path = get_project_root() / "runs"
        self.run_path = self.runs_path / f"run_{self.config_name}_{self.run_timestamp}"
        self.logs_path = self.run_path / "logs"

        # Create logger first (needed by Manager)
        self.logger = Logger(
            logs_path=str(self.logs_path),
            console_logging=True,  # Default values, will be updated after config is loaded
            file_logging=True,
            console_log_level="INFO",
            file_log_level="DEBUG",
        )

        # Emit early log lines so failures during Manager initialization are still captured
        self.logger.info(
            "BenchmarkRunner",
            f"Initializing run at {self.run_timestamp}; logs at: {self.logs_path}",
        )
        self.logger.debug(
            "BenchmarkRunner",
            f"Config path resolved to: {self.config_path}",
        )

        # Initialize the Manager directly (no longer using singleton pattern)
        self.manager = Manager(
            config_path=self.config_path,
            run_path=str(self.run_path),
            logger=self.logger,
        )
        self.initialize_tf_logger()

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        self._initialize_run()
        self.logger.info("BenchmarkRunner", f"Run starts - {self.run_timestamp}")
        self.logger.info("BenchmarkRunner", f"Results stored at: {self.run_path}")

        # We execute multiple jobs per run, each with a different configuration (JobConfig).
        for job_idx, (job_config, task_idx) in enumerate(
            self.manager.generate_run_configs()
        ):
            hyperparameter_tuner = HyperparameterTuner(job_config=job_config)

            # Hyper-parameter Tuning
            task_name = job_config.task_config.name
            self.logger.info(
                "BenchmarkRunner",
                f"Hyperparameter Tuning Starts for job: {job_idx}, task: {task_name}, task config idx: {task_idx}",
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

        self.clean_tf_logger()

    def clean_tf_logger(self):
        """Cleanup TensorBoard writer and ensure all logs are flushed."""
        if self.tf_logger:
            try:
                self.tf_logger.close()
                self.logger.info(
                    "BenchmarkRunner",
                    f"Benchmark runner TensorBoard writer closed, TF logs saved to: {self.tensorboard_dir}",
                )
            except Exception as e:
                self.logger.warning(
                    "BenchmarkRunner",
                    f"Failed to close benchmark TensorBoard writer: {e}",
                )

    def initialize_tf_logger(self):
        """Setup logging configuration and TensorBoard."""
        # Logger already created in _initialize_run with the settings from config
        tensorboard_logging = self.manager.evaluation_settings.tensorboard_logging
        self.logger.info("BenchmarkRunner", f"Python logs saved at: {self.logs_path}")
        self.logger.debug("BenchmarkRunner", "Debug logging is working!")

        # Setup TensorBoard logger - always instantiate
        try:
            self.tensorboard_dir = str(Path(self.run_path) / "tensorboard")
            self.tf_logger = get_tf_logger(
                self.tensorboard_dir, tensorboard_logging=tensorboard_logging
            )
            if tensorboard_logging:
                self.logger.info(
                    "BenchmarkRunner",
                    f"TensorBoard logging enabled at: {self.tensorboard_dir}",
                )
        except Exception as e:
            raise RuntimeError(f"Failed to setup benchmark TensorBoard logging: {e}")


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

    if args.config is not None:
        config_path = args.config
    else:
        # Use absolute path for the default config
        default_config_path = get_configs_dir() / "benchmark.yaml"
        config_path = str(default_config_path)

    # Run the Benchmarks
    runner = BenchmarkRunner(config_path=config_path)
    runner.run()
