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

    def _initialize_run(self):
        """Initialize and update all path-related attributes."""
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.config = load_config(self.config_path, self.logs_path)
        self.config_manager = get_config_manager()

        # Update runs_path from validated settings
        self.runs_path = Path(self.config_manager.benchmark_settings.runs_dir)
        self.run_path = self.runs_path / f"run_{self.config_name}_{self.run_timestamp}"
        self.logs_path = self.run_path / 'logs'
        self._setup_logging()

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        self._initialize_run()
        self.logger.info("BenchmarkRunner", f"Run starts - {self.run_timestamp}")
        self.logger.info("BenchmarkRunner", f"Results stored at: {self.run_path}")
        self.logger.info("BenchmarkRunner", "Extracting Configs")

        # Each task produces a separate config
        for config in self.config_manager.generate_configs():
            # Hyper-parameter Tuning
            self.logger.info("BenchmarkRunner", f"Hyperparameter Tuning Starts for task: {task_name} (config idx={idx+1}/{len(task_configs)})")
            hyperparameter_tuner = HyperparameterTuner(
                config=config,
                logs_path=self.logs_path
            )

            # Hyperparameter Tuning - Context + Train + Validate Losses (Rolling Window with strides of validate_steps)
            evals, hyperparameters = hyperparameter_tuner.optimize_hyperparameters(
                context_steps=config['task']['context_window'],
                train_steps=config['task']['forecast_horizon'],
                validate_steps=config['task']['forecast_horizon']
            ) # For all models in config

            self.logger.success("BenchmarkRunner", f"Hyperparameters Optimized for task: {task_name}")
            self.logger.success("BenchmarkRunner", f"Final Model Evaluation Executed for task: {task_name}")

        self.cleanup()

    def cleanup(self):
        """Cleanup TensorBoard writer and ensure all logs are flushed."""
        if self.tf_logger:
            try:
                self.tf_logger.close()
                self.logger.info("BenchmarkRunner", f"Benchmark runner TensorBoard writer closed, TF logs saved to: {self.tensorboard_dir}")
            except Exception as e:
                self.logger.warning("BenchmarkRunner", f"Failed to close benchmark TensorBoard writer: {e}")

    def _setup_logging(self):
        """Setup logging for benchmark runner execution."""
        console_logging = self.config_manager.benchmark_settings.console_logging
        file_logging = self.config_manager.benchmark_settings.file_logging
        tensorboard_logging = self.config_manager.benchmark_settings.tensorboard_logging

        # Setup Python logger for orchestration - always instantiate
        self.logger = get_logger(self.logs_path, console_logging=console_logging, file_logging=file_logging)
        if file_logging:
            self.logger.info("BenchmarkRunner", f"Python logs saved at: {self.logs_path}")

        # Setup TensorBoard logger - always instantiate
        try:
            self.tensorboard_dir = str(Path(self.run_path) / 'tensorboard')
            self.tf_logger = get_tf_logger(self.tensorboard_dir, tensorboard_logging=tensorboard_logging)
            if tensorboard_logging:
                self.logger.info("BenchmarkRunner", f"TensorBoard logging enabled at: {self.tensorboard_dir}")
        except ImportError:
            self.logger.warning("BenchmarkRunner", "TensorBoard not available, benchmark logging disabled")
            self.tf_logger = None
        except Exception as e:
            self.logger.warning("BenchmarkRunner", f"Failed to setup benchmark TensorBoard logging: {e}")
            self.tf_logger = None

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