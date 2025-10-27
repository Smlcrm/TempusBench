import os
import sys
import datetime
import argparse

from tempus_bench.utils.logger import get_logger
from tempus_bench.utils.tf_logger import get_tf_logger
from tempus_bench.utils.paths import get_tasks_dir
from tempus_bench.config import load_config, validate_config_file
from tempus_bench.pipeline.hyperparameter_tuning import HyperparameterTuner

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = get_tasks_dir()

class BenchmarkRunner:
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner with configuration.
        Args:
            config: Configuration dictionary for the pipeline
            config_path: Path to the config file used
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(self.config_path))[0]
        self.run_dir = None # Defined in self.run
        self.logs_dir = None # Defined in self.setup_logging
        self.tasks_dir = TASKS_DIR
        self.tensorflow_dir = None # Defined in self.setup_logging
        self.tf_logger = None # Defined in self.setup_logging
        self.logger = None # Will be initialized in setup_logging
        self.runs_dir = self.config.get("paths", {}).get("runs_dir", "runs")
        self.logs_dir_rel = self.config.get("paths", {}).get("logs_dir", "logs")

    def setup_logging(self):
        """Setup logging for benchmark runner execution."""
        logging = self.config["logging"]["console_logging"]
        if logging:
            # Setup Python logger for orchestration
            self.logs_dir = os.path.join(self.run_dir, 'logs')
            self.logger = get_logger(self.logs_dir)

            try:
                self.tensorboard_dir = os.path.join(self.run_dir, 'tensorboard')
                self.tf_logger = get_tf_logger(self.tensorboard_dir)
                self.logger.info("BenchmarkRunner", f"TensorBoard logging enabled at: {self.tensorboard_dir}")
                self.logger.info("BenchmarkRunner", f"Python logs saved at: {self.logs_dir}")
            except ImportError:
                self.logger.warning("BenchmarkRunner", "TensorBoard not available, benchmark logging disabled")
            except Exception as e:
                self.logger.warning("BenchmarkRunner", f"Failed to setup benchmark TensorBoard logging: {e}")

        return logging

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Directory where the run logs and evaluations are stored
        self.run_dir = os.path.join(self.runs_dir, f"run_{self.config_name}_{self.run_timestamp}")
        self.tasks_dir = TASKS_DIR
        logging = self.setup_logging()

        if logging: self.logger.info("BenchmarkRunner", f"Run starts - {self.run_timestamp}")
        if logging: self.logger.info("BenchmarkRunner", f"Results stored at: {self.run_dir}")

        # Configs
        if logging: self.logger.info("BenchmarkRunner", "Extracting Configs")
        task_config = self.config["task"]
        evaluation_config = self.config["evaluation"]

        # Task Config
        if logging: self.logger.info("BenchmarkRunner", "Extracting Task Configs")
        context_window = task_config["context_window"]
        forecast_horizon = task_config["forecast_horizon"]

        # Hyper-parameter Tuning
        if logging: self.logger.info("BenchmarkRunner", "Hyperparameter Tuning Starts")
        hyperparameter_tuner = HyperparameterTuner(
            config_path=self.config_path,
            run_dir=self.run_dir
        )

        # Hyperparameter Tuning - Context + Train + Validate Losses (Rolling Window with strides of validate_steps)
        evals, hyperparameters = hyperparameter_tuner.optimize_hyperparameters(
            context_steps=context_window,
            train_steps=forecast_horizon,
            validate_steps=forecast_horizon
        ) # For all models in config
        if logging: self.logger.success("BenchmarkRunner", "Hyperparameters Optimized")
        if logging: self.logger.success("BenchmarkRunner", "Final Model Evaluation Executed")

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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(script_dir, "configs", "all_models.yaml")
        config_path = default_config_path

    try:
        validate_config_file(config_path)
        print(f"[INFO] Config file '{config_path}' validated successfully.")
    except Exception as e:
        print(f"[ERROR] Config file validation failed: {e}")
        sys.exit(1)

    # Run the Benchmarks
    runner = BenchmarkRunner(config_path=config_path)

    # Override tasks directory if provided
    if args.tasks_dir:
        runner.tasks_dir = args.tasks_dir

    runner.run()