import argparse
import subprocess
import importlib
import os
import sys
import yaml
import json
import pickle
import tempfile
import pathlib

# Save to temp file

from tqdm import tqdm
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor

import datetime


class BenchmarkRunner:

    def __init__(self, config, config_path=None):
        """
        Initialize benchmark runner with configuration.
        Args:
            config: Configuration dictionary for the pipeline
            config_path: Path to the config file used
        """
        self.config = config
        self.config_path = config_path

        # Setup TensorBoard logging for benchmark runner like we had before
        self.setup_tensorboard_logging()

    def setup_tensorboard_logging(self):
        """Setup TensorBoard logging for benchmark runner execution."""
        # Only enable TensorBoard if config specifies tensorboard: true
        if not self.config.get("tensorboard", False):
            print(
                "[INFO] Benchmark runner TensorBoard logging disabled (tensorboard: false in config)"
            )
            self.writer = None
            self.log_dir = None
            return

        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create benchmark runner logging directory like before
            config_file_name = (
                os.path.splitext(os.path.basename(self.config_path))[0]
                if self.config_path
                else "unknown_config"
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            # Create the exact directory structure we had before
            # Use project root (parent of benchmarking_pipeline) for runs directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            runs_dir = os.path.join(project_root, "runs")
            benchmark_dir = f"benchmark_runner_{config_file_name}_{timestamp}"
            benchmark_runs_dir = os.path.join(runs_dir, benchmark_dir)

            # Ensure directories exist
            os.makedirs(benchmark_runs_dir, exist_ok=True)

            # Create TensorBoard writer
            self.writer = SummaryWriter(benchmark_runs_dir)
            self.log_dir = benchmark_runs_dir
            print(
                f"[INFO] Benchmark runner TensorBoard logging enabled at: {benchmark_runs_dir}"
            )

        except ImportError:
            print("[WARNING] TensorBoard not available, benchmark logging disabled")
            self.writer = None
            self.log_dir = None
        except Exception as e:
            print(f"[WARNING] Failed to setup benchmark TensorBoard logging: {e}")
            self.writer = None
            self.log_dir = None

    def _analyze_dataset(self, dataset_chunk):
        """
        Analyze dataset chunk to determine properties for auto-detection.

        Args:
            dataset_chunk: Dataset chunk object with train/test data

        Returns:
            Dictionary with dataset analysis information
        """
        # Use training targets to determine number of target columns
        train_targets = dataset_chunk.train.targets
        if train_targets is None:
            raise ValueError("Dataset analysis failed: train.targets is None")
        if hasattr(train_targets, "shape"):
            num_targets = (
                train_targets.shape[1]
                if len(getattr(train_targets, "shape")) > 1
                else 1
            )
        else:
            num_targets = 1

        # Basic shape info of targets
        data_shape = train_targets.shape
        total_columns = data_shape[1]

        # More sophisticated detection could be added here
        has_multiple_targets = num_targets > 1

        dataset_info = {
            "num_targets": num_targets,
            "total_columns": total_columns,
            "has_multiple_targets": has_multiple_targets,
            "data_shape": data_shape,
        }

        print(
            f"[DEBUG] Dataset analysis: shape={dataset_info['data_shape']}, targets={num_targets}"
        )
        return dataset_info

    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        # Determine config file name for logging
        config_file_name = (
            os.path.splitext(os.path.basename(self.config_path))[0]
            if self.config_path else "unknown_config"
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        datasets_dir = pathlib.Path(__file__).resolve().parent / 'datasets'
        # Load dataset config
        dataset_cfg = self.config["dataset"]
        
        dataset_paths = []  # Relative paths in dataset root dir containing csvs
        dataset_name = dataset_cfg['name']
        if dataset_name == "*":
            # Recursively find all subdirectories under datasets_dir that contain at least one CSV file
            for root, dirs, files in os.walk(datasets_dir):
                if any(f.endswith(".csv") for f in files):
                    rel_path = os.path.relpath(root, datasets_dir)
                    dataset_paths.append(rel_path)
            if not dataset_paths:
                raise ValueError(f"No dataset directories containing CSV files found in {datasets_dir}")
        elif dataset_name.endswith("/*"):
            # Support for subdirectory wildcard, e.g., "subdir/*"
            subdir = dataset_name[:-2]
            subdir_path = datasets_dir / subdir
            if not os.path.exists(subdir_path) or not os.path.isdir(subdir_path):
                raise ValueError(f"Subdirectory {subdir_path} does not exist or is not a directory")
            # Find all subdirectories under subdir_path that contain at least one CSV file
            for root, dirs, files in os.walk(subdir_path):
                if any(f.endswith(".csv") for f in files):
                    rel_path = os.path.relpath(root, datasets_dir)
                    dataset_paths.append(rel_path)
            if not dataset_paths:
                raise ValueError(f"No dataset directories containing CSV files found in {subdir_path}")
        else:
            dataset_dir_path = datasets_dir / dataset_name
            if not os.path.exists(dataset_dir_path):
                raise ValueError(f"Dataset in path {dataset_dir_path} does not exist")
            # Only add the dataset directory if it directly contains at least one CSV file
            if any(f.endswith(".csv") for f in os.listdir(dataset_dir_path)):
                dataset_paths.append(dataset_name)
            else:
                raise ValueError(f"No CSV files found in dataset directory {dataset_dir_path}")
        print(f"[DEBUG] dataset_paths: {dataset_paths}")
        
        for dd, dataset_name in enumerate(tqdm(dataset_paths)):
            if 18 < dd < 20: continue
            tqdm.write(f"Processing dataset: {dataset_name}")
            split_ratio = dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1])
            num_chunks = dataset_cfg.get("chunks", 1)
            dataset_cfg["name"] = dataset_name
            data_loader = DataLoader(dataset_cfg)

            # Preprocess all dataset chunks
            all_dataset_chunks = data_loader.load_several_chunks(num_chunks)
            preprocessor = Preprocessor(
                {"dataset": {"normalize": dataset_cfg.get("normalize", False)}}
            )
            all_dataset_chunks = [
                preprocessor.preprocess(chunk).data for chunk in all_dataset_chunks
            ]

            # Convert Dataset objects to serializable dictionaries to avoid module import issues
            serializable_chunks = []
            for chunk in all_dataset_chunks:
                serializable_chunk = {
                    "train": {
                        "targets": (
                            chunk.train.targets.values.tolist()
                            if hasattr(chunk.train.targets, "values")
                            else chunk.train.targets.tolist()
                        ),
                        "features": (
                            chunk.train.features.values.tolist()
                            if chunk.train.features is not None
                            and hasattr(chunk.train.features, "values")
                            else chunk.train.features
                        ),
                        "timestamps": (
                            chunk.train.timestamps.tolist()
                            if hasattr(chunk.train.timestamps, "tolist")
                            else list(chunk.train.timestamps)
                        ),
                    },
                    "validation": {
                        "targets": (
                            chunk.validation.targets.values.tolist()
                            if hasattr(chunk.validation.targets, "values")
                            else chunk.validation.targets.tolist()
                        ),
                        "features": (
                            chunk.validation.features.values.tolist()
                            if chunk.validation.features is not None
                            and hasattr(chunk.validation.features, "values")
                            else chunk.validation.features
                        ),
                        "timestamps": (
                            chunk.validation.timestamps.tolist()
                            if hasattr(chunk.validation.timestamps, "tolist")
                            else list(chunk.validation.timestamps)
                        ),
                    },
                    "test": {
                        "targets": (
                            chunk.test.targets.values.tolist()
                            if hasattr(chunk.test.targets, "values")
                            else chunk.test.targets.tolist()
                        ),
                        "features": (
                            chunk.test.features.values.tolist()
                            if chunk.test.features is not None
                            and hasattr(chunk.test.features, "values")
                            else chunk.test.features
                        ),
                        "timestamps": (
                            chunk.test.timestamps.tolist()
                            if hasattr(chunk.test.timestamps, "tolist")
                            else list(chunk.test.timestamps)
                        ),
                    },
                    "name": chunk.name,
                    "metadata": chunk.metadata,
                }
                serializable_chunks.append(serializable_chunk)

            # Temporary file to access data across different subprocesses
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                pickle.dump(serializable_chunks, tmp)
                chunk_path = tmp.name
                print("[DEBUG] Temporary file path", chunk_path)

            with open(chunk_path, "rb") as f:
                yuh = pickle.load(f)
            print(f"[DEBUG] Number of chunks: {len(all_dataset_chunks)}")
            if len(all_dataset_chunks) > 0:
                first = all_dataset_chunks[0]
                targets_shape = getattr(first.train.targets, "shape", None)
                features_shape = (
                    getattr(first.train.features, "shape", None)
                    if first.train.features is not None
                    else None
                )
                print(f"[DEBUG] First chunk train targets shape: {targets_shape}")
                print(f"[DEBUG] First chunk train features shape: {features_shape}")

            # Get model names directly from the model section
            model_names = list(self.config["model"].keys())

            # Import the model router
            from benchmarking_pipeline.models.model_router import ModelRouter

            # Initialize model router
            model_router = ModelRouter()

            # Run each model we have individually
            # Establish a shared, absolute TensorBoard base directory
            # Default to project root (parent of benchmarking_pipeline)
            default_log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
                "tensorboard",
            )
            base_log_dir = self.config['evaluation'].get("log_dir", default_log_dir)
            base_log_dir = os.path.abspath(base_log_dir)
            os.makedirs(base_log_dir, exist_ok=True)

            for model_spec in model_names:
                # Parse the model specification (e.g., 'arima', 'chronos')
                model_name = model_router.parse_model_spec(model_spec)

                # Get parameters for the base model name (without variant)
                if model_name in self.config["model"]:
                    model_params = self.config["model"][model_name]
                else:
                    raise ValueError(
                        f"No parameters found for {model_name}. Please check your configuration."
                    )

                # Analyze dataset to determine target structure
                dataset_chunk = data_loader.load_single_chunk(1)

                # Infer target structure from data
                # All data is treated as multivariate where univariate is just num_targets == 1
                if dataset_chunk.metadata and "num_targets" in dataset_chunk.metadata:
                    num_targets = dataset_chunk.metadata["num_targets"]
                else:
                    raise ValueError(
                        "Dataset metadata missing num_targets. Cannot determine target structure."
                    )

                print(f"Dataset has {num_targets} target variable(s)")
                print(
                    f"Note: All data is treated as multivariate where univariate is just num_targets == 1"
                )

                # Get model path based on inferred target count
                # The model router now treats univariate as a special case of multivariate
                model_path, model_file, model_class = (
                    model_router.get_model_path_by_target_count(model_name, num_targets)
                )

                print(f"[INFO] Processing model: {model_spec}")
                print(f"[INFO] Model name: {model_name}")
                print(f"[INFO] Num targets: {num_targets}")
                print(f"[INFO] Folder path: {model_path}")
                print(f"[INFO] File name: {model_file}")
                print(f"[INFO] Class name: {model_class}")

                # Log model execution start to TensorBoard like we had before
                if self.writer:
                    try:
                        self.writer.add_text(
                            f"model_execution/{model_name}/start_time",
                            datetime.datetime.now().isoformat(),
                            0,
                        )
                        self.writer.add_text(
                            f"model_execution/{model_name}/config",
                            str(full_config_data.get("model", {}).get(model_name, {})),
                            0,
                        )
                        self.writer.add_scalar(
                            f"model_execution/{model_name}/num_targets", num_targets, 0
                        )
                        print(f"[INFO] Logged model execution start to TensorBoard")
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to log model execution start to TensorBoard: {e}"
                        )

                requirements_path = f"{model_path}/requirements.txt"

                # Prepare a per-model temporary config that injects shared log_dir and per-model run_name
                temp_config = config.copy()
                temp_config["log_dir"] = base_log_dir
                temp_config["run_name"] = model_name
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".yaml", mode="w"
                ) as tmp_cfg:
                    yaml.safe_dump(temp_config, tmp_cfg)
                    temp_config_path = tmp_cfg.name

                # Create conda environment name based on model name to avoid conflicts
                conda_env_name = model_name

                # Something on my mind: uv could probably make this process a LOT quicker - definitely something to explore.

                # Create the conda environment if it doesn't already exist.
                try:
                    # Check if environment exists using conda env list
                    result = subprocess.run(
                        ["conda", "env", "list"], check=True, capture_output=True, text=True
                    )

                    # Check if our environment name appears in the list
                    env_exists = conda_env_name in result.stdout

                    if not env_exists:
                        subprocess.run(
                            ["conda", "create", "-n", conda_env_name, "python=3.10", "-y"],
                            check=True,
                        )
                        print(
                            f"[SUCCESS] Conda environment for {conda_env_name} has been made!"
                        )
                    else:
                        print(
                            f"[INFO] Conda environment '{conda_env_name}' already exists. Skipping creation."
                        )

                except subprocess.CalledProcessError:
                    # If conda env list fails, try to create the environment anyway
                    print(
                        f"[WARNING] Could not check existing environments, attempting to create {conda_env_name}"
                    )
                    subprocess.run(
                        ["conda", "create", "-n", conda_env_name, "python=3.10", "-y"],
                        check=True,
                    )
                    print(
                        f"[SUCCESS] Conda environment for {conda_env_name} has been made!"
                    )

                # Install the dependencies of the corresponding model without activating the model.
                subprocess.run(
                    [
                        "conda",
                        "run",
                        "-n",
                        conda_env_name,
                        "pip",
                        "install",
                        "-r",
                        requirements_path,
                    ],
                    check=True,
                )
                print(
                    f"[SUCCESS] Dependencies for {conda_env_name} have been installed in the proper conda environment!"
                )

                # Install the benchmarking_pipeline package in the model environment
                # Install from the root directory of the package where pyproject.toml is located
                # This file is benchmarking_pipeline/run_benchmark.py, so root is parent of 'benchmarking_pipeline'
                root_dir = pathlib.Path(__file__).resolve().parent.parent
                subprocess.run(
                    [
                        "conda",
                        "run",
                        "-n",
                        conda_env_name,
                        "pip",
                        "install",
                        "-e",
                        root_dir,
                    ],
                    check=True,
                )
                print(
                    f"[SUCCESS] Benchmarking pipeline package installed in {conda_env_name} environment!"
                )

                # Temp file for model results to be written by subprocess
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_res:
                    result_path = tmp_res.name
                
                subprocess.run(
                    [
                        "conda",
                        "run",
                        "-n",
                        conda_env_name,
                        "python",
                        "-m",
                        "benchmarking_pipeline.model_executor",
                        "--config",
                        temp_config_path,
                        "--chunk_path",
                        chunk_path,
                        "--model_folder_name",
                        model_path,
                        "--model_file_name",
                        model_file,
                        "--model_class_name",
                        model_class,
                        "--result_path",
                        result_path,
                        "--dataset_name",
                        dataset_name
                    ],
                    check=True,
                )

                # Read back results and log to host TensorBoard
                try:
                    with open(result_path, "r") as rf:
                        payload = json.load(rf)
                    # Lazy import to avoid TF in model envs
                    from benchmarking_pipeline.pipeline.logger import Logger

                    host_logger = Logger({"log_dir": base_log_dir, "run_name": model_name})
                    host_logger.log_hparams(
                        {**payload.get("best_hyperparameters", {}), "model": model_name},
                        payload.get("metrics", {}),
                    )
                    host_logger.log_metrics(
                        payload.get("metrics", {}), step=1, model_name=model_name
                    )
                    # If subprocess produced plots, log them as images
                    plot_val = payload.get("forecast_plot_val_path")
                    plot_test = payload.get("forecast_plot_test_path")
                    if plot_val and os.path.exists(plot_val):
                        host_logger.log_image_file(
                            plot_val, tag=f"{model_name}/forecast_validation", step=1
                        )
                        try:
                            os.remove(plot_val)
                        except Exception:
                            pass
                    if plot_test and os.path.exists(plot_test):
                        host_logger.log_image_file(
                            plot_test, tag=f"{model_name}/forecast_test", step=1
                        )
                        try:
                            os.remove(plot_test)
                        except Exception:
                            pass
                except Exception as host_log_err:
                    print(f"[WARNING] Failed to host-log model results: {host_log_err}")

                # Log model execution completion to TensorBoard like we had before
                if self.writer:
                    try:
                        self.writer.add_text(
                            f"model_execution/{model_name}/completion_time",
                            datetime.datetime.now().isoformat(),
                            0,
                        )
                        self.writer.add_scalar(
                            f"model_execution/{model_name}/status", 1, 0
                        )  # 1 = completed
                        print(f"[INFO] Logged model execution completion to TensorBoard")
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to log model execution completion to TensorBoard: {e}"
                        )

                # Get rid of the environment when we're done with the current model.
                # Keep env to avoid re-creation overhead
                print(f"[INFO] Keeping conda environment '{conda_env_name}' for reuse.")

            os.remove(chunk_path)
            print("All model files ran!")

            # Log benchmark completion and cleanup TensorBoard writer
            if self.writer:
                try:
                    self.writer.add_text(
                        "benchmark/completion_time", datetime.datetime.now().isoformat(), 0
                    )
                    self.writer.add_scalar("benchmark/total_models", len(model_names), 0)
                    self.writer.add_scalar("benchmark/status", 1, 0)  # 1 = completed
                    print(f"[INFO] Logged benchmark completion to TensorBoard")
                except Exception as e:
                    print(
                        f"[WARNING] Failed to log benchmark completion to TensorBoard: {e}"
                    )

            # Cleanup TensorBoard writer
            self.cleanup()

    def cleanup(self):
        """Cleanup TensorBoard writer and ensure all logs are flushed."""
        if self.writer:
            try:
                self.writer.close()
                print(
                    f"[INFO] Benchmark runner TensorBoard writer closed, logs saved to: {self.log_dir}"
                )
            except Exception as e:
                print(f"[WARNING] Failed to close benchmark TensorBoard writer: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarking pipeline with specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarking_pipeline/configs/all_model_univariate.yaml",
        help="Path to the config YAML file",
    )
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    runner = BenchmarkRunner(config=config, config_path=config_path)
    runner.run()
