import os
import sys
import datetime
import argparse
from pathlib import Path

from tempus_bench.utils.logger import get_logger
from tempus_bench.utils.tf_logger import get_tf_logger
from tempus_bench.utils.paths import get_project_root, get_configs_dir
from tempus_bench.config import load_config, get_config_manager
from tempus_bench.pipeline.hyperparameter_tuning import HyperparameterTuner

class BenchmarkRunner:
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner with configuration.
        Args:
            config_path: Path to the config file used
        """
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(self.config_path))[0]

    def setup_logging(self):
        """Setup logging for benchmark runner execution."""
        console_logging = self.config_manager.settings.console_logging
        tensorboard_logging = self.config_manager.settings.tensorboard_logging

        if console_logging:
            # Setup Python logger for orchestration
            self.logger = get_logger(self.logs_path)
            self.logger.info("BenchmarkRunner", f"Python logs saved at: {self.logs_path}")
        else:
            self.logger = None

        if tensorboard_logging:
            try:
                self.tensorboard_dir = str(Path(self.run_path) / 'tensorboard')
                self.tf_logger = get_tf_logger(self.tensorboard_dir)
                if self.logger:
                    self.logger.info("BenchmarkRunner", f"TensorBoard logging enabled at: {self.tensorboard_dir}")
            except ImportError:
                if self.logger:
                    self.logger.warning("BenchmarkRunner", "TensorBoard not available, benchmark logging disabled")
            except Exception as e:
                if self.logger:
                    self.logger.warning("BenchmarkRunner", f"Failed to setup benchmark TensorBoard logging: {e}")
        else:
            self.tf_logger = None

    def _initialize_run(self):
        """Initialize and update all path-related attributes."""
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Initialize runs_path from project root and settings
        self.runs_path = str(Path(get_project_root()) / "runs")  # Will be updated after ConfigManager loads settings

        # Directory where the run logs and evaluations are stored
        self.run_path = str(Path(self.runs_path) / f"run_{self.config_name}_{self.run_timestamp}")
        self.logs_path = str(Path(self.run_path) / 'logs')

        # Load and validate configuration using ConfigManager
        self.config = load_config(self.config_path, self.logs_path)
        self.config_manager = get_config_manager()

        # Update runs_path from validated settings
        self.runs_path = str(Path(get_project_root()) / self.config_manager.settings.runs_dir)
        self.run_path = str(Path(self.runs_path) / f"run_{self.config_name}_{self.run_timestamp}")
        self.logs_path = str(Path(self.run_path) / 'logs')
        self.setup_logging()

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        self._initialize_run()
        if self.logger: self.logger.info("BenchmarkRunner", f"Run starts - {self.run_timestamp}")
        if self.logger: self.logger.info("BenchmarkRunner", f"Results stored at: {self.run_path}")

        # Configs - access from validated ConfigManager
        if self.logger: self.logger.info("BenchmarkRunner", "Extracting Configs")
        # Iterate across all tasks and their configs
        for task_name, task_configs in self.config_manager.task.items():
            for idx, task_config in enumerate(task_configs):
                if self.logger:
                    self.logger.debug("BenchmarkRunner", f"Task config idx={idx} data={task_config.model_dump()}")
                context_window = task_config.context_window
                forecast_horizon = task_config.forecast_horizon

                # Hyper-parameter Tuning
                if self.logger: self.logger.info("BenchmarkRunner", f"Hyperparameter Tuning Starts for task: {task_name} (config idx={idx})")
                hyperparameter_tuner = HyperparameterTuner(
                    config_path=self.config_path,
                    run_path=self.run_path
                )

                # Hyperparameter Tuning - Context + Train + Validate Losses (Rolling Window with strides of validate_steps)
                evals, hyperparameters = hyperparameter_tuner.optimize_hyperparameters(
                    context_steps=context_window,
                    train_steps=forecast_horizon,
                    validate_steps=forecast_horizon
                ) # For all models in config

                if self.logger: self.logger.success("BenchmarkRunner", f"Hyperparameters Optimized for task: {task_name}")
                if self.logger: self.logger.success("BenchmarkRunner", f"Final Model Evaluation Executed for task: {task_name}")

        self.cleanup()

    def cleanup(self):
        """Cleanup TensorBoard writer and ensure all logs are flushed."""
        if self.tf_logger:
            try:
                self.tf_logger.close()
                if self.logger:
                    self.logger.info("BenchmarkRunner", f"Benchmark runner TensorBoard writer closed, TF logs saved to: {self.tensorboard_dir}")
            except Exception as e:
                if self.logger:
                    self.logger.warning("BenchmarkRunner", f"Failed to close benchmark TensorBoard writer: {e}")

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