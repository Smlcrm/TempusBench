import os
import sys
import yaml
import json
import pickle
import pathlib
import datetime
import tempfile
import argparse
import importlib
import subprocess

from tqdm import tqdm

from benchmarking_pipeline.utils.logger import Logger
from benchmarking_pipeline.utils import config_validator
from benchmarking_pipeline.model_executor import ModelExecutor
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT_DIR, "runs")
DATASETS_DIR = os.path.join(ROOT_DIR, "benchmarking_pipeline", "datasets")

class BenchmarkRunner:
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner with configuration.
        Args:
            config: Configuration dictionary for the pipeline
            config_path: Path to the config file used
        """
        self.config = config_validator.load_config(config_path)
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(self.config_path))[0]
        self.run_dir = None # Defined in self.run
        self.logs_dir = None # Defined in self.setup_logging
        self.datasets_dir = None # Defined in self.run
        self.tensorflow_dir = None # Defined in self.setup_logging
        self.tf_logger = None # Defined in self.setup_logging
        self.logger = None # Will be initialized in setup_logging

    def setup_logging(self):
        """Setup logging for benchmark runner execution."""
        logging = self.config["evaluation"]["logging"]
        if logging:
            # Setup Python logger for orchestration
            self.logs_dir = os.path.join(self.run_dir, 'logs')
            self.logger = Logger(logs_dir=self.logs_dir, name='BenchmarkRunner')

            try:
                from benchmarking_pipeline.utils.tf_logger import TFLogger
                self.tensorboard_dir = os.path.join(self.run_dir, 'tensorboard')
                self.tf_logger = TFLogger(tf_logs_dir=self.tensorboard_dir, name='TFBenchmarkRunner')
                self.logger.info(f"TensorBoard (TFBenchmarkRunner) logging enabled at: {self.tensorboard_dir}")
                self.logger.info(f"Python logs saved at: {self.logs_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available, benchmark logging disabled")
            except Exception as e:
                self.logger.warning(f"Failed to setup benchmark TensorBoard logging: {e}")

        return logging

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Directory where the run logs and evaluations are stored
        self.run_dir = os.path.join(RUNS_DIR, f"run_{self.config_name}_{self.run_timestamp}")
        dataset_config =  self.config['task']['dataset']
        self.datasets_dir = DATASETS_DIR
        logging = self.setup_logging()

        if logging: self.logger.info(f"Run starts - {self.run_timestamp}")
        if logging: self.logger.info(f"Results stored at: {self.run_dir}")

        # Configs
        if logging: self.logger.info("Extracting Configs")
        task_config = self.config["task"]
        dataset_config = task_config["dataset"]
        evaluation_config = self.config["evaluation"]

        # Task Config
        if logging: self.logger.info("Extracting Task Configs")
        task_type = task_config["task_type"]
        context_window = task_config["context_window"]
        forecast_horizon = task_config["forecast_horizon"]

        # Hyper-parameter Tuning
        if logging: self.logger.info("Hyperparameter Tuning Starts")
        hyperparameter_tuner = ModelExecutor(
            config_path=self.config_path,
            run_dir=self.run_dir,
            datasets_dir=self.datasets_dir,
            logger=self.logger,
            tf_logger=self.tf_logger
        )

        # Hyperparameter Tuning - Context + Train + Validate Losses (Rolling Window with strides of validate_steps)
        evals, hyperparameters = hyperparameter_tuner.optimize_hyperparameters(
            context_steps=context_window,
            train_steps=forecast_horizon,
            validate_steps=forecast_horizon
        ) # For all models in config
        if logging: self.logger.success("Hyperparameters Optimized")

        # Model Executor
        if logging: self.logger.info("Final Model Evaluation Starts")


        if logging: self.logger.success("Final Model Evaluation Executed")

        self.cleanup()

    def cleanup(self):
        """Cleanup TensorBoard writer and ensure all logs are flushed."""
        if self.tf_logger:
            try:
                self.tf_logger.close()
                if self.logger:
                    self.logger.info(f"Benchmark runner TensorBoard writer closed, TF logs saved to: {self.tensorboard_dir}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to close benchmark TensorBoard writer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarking pipeline with specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config YAML file. If not specified, uses the default config in benchmarking_pipeline/configs/all_models.yaml",
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
        config_validator.validate_config_file(config_path)
        print(f"[INFO] Config file '{config_path}' validated successfully.")
    except Exception as e:
        print(f"[ERROR] Config file validation failed: {e}")
        sys.exit(1)

    # Run the Benchmarks
    runner = BenchmarkRunner(config_path=config_path)
    runner.run()