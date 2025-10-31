"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that executes individual models in
isolated conda environments to avoid dependency conflicts. It handles model execution
with specific hyperparameters and returns evaluation results.

The module can also be invoked directly from the command line:
    python -m tempus_bench.pipeline.model_executor --model-name <name> --hyperparameters '{}' ...
"""

import argparse
import json
import os
import importlib.util
import tempfile
import pickle


from pathlib import Path
from typing import Any, Dict, Optional

from tempus_bench.utils.configs import JobConfig
from tempus_bench.utils.envs import CondaEnvManager
from tempus_bench.utils.log_manager import LogManager
from tempus_bench.utils.paths import get_project_root, get_models_dir



class ModelExecutor:
    """
    Executes a single model inside its dedicated Conda environment.

    The executor prepares the command-line invocation, ensures the requested environment
    is available, and parses JSON results emitted by the child process. This isolation
    allows different models to have conflicting dependencies without interference.

    Attributes:
        job_config (JobConfig): Complete job configuration including task and model settings.
        logger (LoggerManager): Logger instance for logging operations.
    """

    def __init__(
        self,
        job_config: JobConfig,
    ):
        """
        Initialize the executor with execution metadata.

        Args:
            job_config: Job configuration object.
        """
        self.job_config = job_config

    def execute_model(
        self,
        hyperparameters: dict,
        context_steps: int,
        train_steps: int,
        validate_steps: int,
    ) -> dict:
        """
        Execute a single model with specific hyperparameters on dataset windows.

        This method creates or uses an existing conda environment for the model, executes
        the model via CLI with the specified hyperparameters, and processes the results
        across multiple rolling windows of the dataset.

        Args:
            hyperparameters (dict): Concrete hyperparameter assignment to forward to
                the model.
            context_steps (int): Number of context steps extracted from each window.
            train_steps (int): Number of steps used for fitting inside each window.
            validate_steps (int): Number of steps reserved for evaluation.

        Returns:
            dict: List of dictionaries containing evaluation metrics and optional
                artifacts (predictions, true values) produced by the model for each
                window. Each dictionary contains metrics and results for one window.

        Raises:
            RuntimeError: If model execution fails in the conda environment.
            ValueError: If no valid JSON output is found in command results.
        """
        model_name = self.job_config.model_config.model_name
        LogManager.get_logger().info(
            "ModelExecutor",
            f"Executing model {model_name} with hyperparameters {hyperparameters}",
        )

        # Create Conda Environment
        requirements_path = self._get_model_requirements(model_name=model_name)
        python_version = self.job_config.model_setting["python_version"]
        conda_env = CondaEnvManager(
            name=f"benchmark.{model_name}",
            python=python_version,
            requirements_path=requirements_path,
            reinstall=self.job_config.evaluation_setting.reinstall_conda,
        )


        with tempfile.TemporaryDirectory() as temp_dir:
            job_config_path = os.path.join(temp_dir, "job_config.pkl")
            
            with open(job_config_path, "wb") as f:
                pickle.dump(self.job_config, f)

            # Create a temporary directory to store the pickled job_config
            # Build CLI command
            hyperparameters_json = json.dumps(hyperparameters)
            command = (
                f"python -m tempus_bench.pipeline.model_executor "
                f"--task-name {self.job_config.task_config.task_name} "
                f"--model-name {model_name} "
                f"--hyperparameters '{hyperparameters_json}' "
                f"--context-steps {context_steps} "
                f"--train-steps {train_steps} "
                f"--validate-steps {validate_steps} "
                f"--job-config-path {job_config_path} "
            )

            LogManager.get_logger().debug(
                "ModelExecutor", f"Running command: {command}"
            )

            result = conda_env.run(command=command)

            LogManager.get_logger().success(
                "ModelExecutor", f"Command ran successfully for model {model_name}"
            )

            # Parse the JSON output from the command
            # Find the last line that contains JSON (the script outputs debug info first)
            lines = result.stdout.strip().split("\n")
            json_line = None
            for line in reversed(lines):
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    json_line = line.strip()
                    break

            if json_line is None:
                raise ValueError(
                    f"No valid JSON found in command output. Output was: {result.stdout}"
                )

            eval_results = json.loads(json_line)

        return eval_results

    def _get_model_requirements(self, model_name: str) -> str:
        """
        Get the absolute path to requirements.txt for the requested model.

        Args:
            model_name (str): Name of the model to get requirements for.

        Returns:
            str: Absolute path to the requirements.txt file for the model.

        Raises:
            FileNotFoundError: If requirements.txt is not found at the expected path.
        """
        # Models are now directly in the models directory
        models_dir = get_models_dir()
        model_dir = models_dir / model_name

        # Construct requirements path
        req_path = model_dir / "requirements.txt"
        if not req_path.exists():
            raise FileNotFoundError(
                f"requirements.txt not found at expected path: {req_path}"
            )
        return str(req_path.resolve())


def main():
    """
    CLI entry point for executing a model with specific hyperparameters.

    This function parses command-line arguments, loads the job configuration, executes
    the model across multiple rolling windows, and outputs evaluation results as JSON.
    The model is loaded dynamically based on the model name, and predictions are computed
    for each validation window.

    Command-line Arguments:
        --model-name: Name of the model to execute.
        --hyperparameters: JSON string of hyperparameter values.
        --context-steps: Number of context steps.
        --train-steps: Number of training steps.
        --validate-steps: Number of validation steps.
        --job-config-path: Path to pickled job configuration file.

    Returns:
        None: Results are printed to stdout as JSON.

    Raises:
        ImportError: If the model module cannot be loaded.
        ValueError: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Execute a forecasting model with specific hyperparameters on a dataset window"
    )
    parser.add_argument(
        "--task-name", type=str, required=True, help="Temporary task dataset path"
    )
    parser.add_argument(
        "--model-name", required=True, help="Name of the model to execute"
    )
    parser.add_argument(
        "--hyperparameters", required=True, help="JSON string of hyperparameter values"
    )
    parser.add_argument(
        "--context-steps", type=int, required=True, help="Number of context steps"
    )
    parser.add_argument(
        "--train-steps", type=int, required=True, help="Number of training steps"
    )
    parser.add_argument(
        "--validate-steps", type=int, required=True, help="Number of validation steps"
    )
    parser.add_argument(
        "--job-config-path", required=True, help="Path to job configuration file"
    )
    

    args = parser.parse_args()

    # Parse hyperparameters JSON
    hyperparameters = json.loads(args.hyperparameters)


    # Load configuration to get JobConfig

    with open(args.job_config_path, "rb") as f:
        job_config = pickle.load(f)


    task_name = args.task_name
    temp_task_dataset_path = Path(get_project_root()) / "temp_task_datasets" / task_name
    
    with open(temp_task_dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Create data loader
    context_steps = args.context_steps
    train_steps = args.train_steps
    validate_steps = args.validate_steps
    model_name = args.model_name

    steps = [
        ("context", context_steps),
        ("train", train_steps),
        ("validate", validate_steps),
    ]


    window_generator = dataset.generate_dataset_split(
        steps=steps, stride=args.validate_steps, max_windows=job_config.evaluation_config.max_windows
    )

    outputs = []

    for window_idx, dataset_splits in enumerate(window_generator):

        timestamps = dataset.timestamps
        target = dataset.target
        freq = dataset.metadata["time_freq"]  # type: ignore

        # Import model - models are now directly in the models directory
        models_dir = get_models_dir()
        model_dir = models_dir / args.model_name
        model_file = f"{model_name}_model"
        module_path = str(model_dir / f"{model_file}.py")

        # Generate class name (PascalCase + Model suffix)
        class_name = (
            "".join(word.capitalize() for word in args.model_name.split("_")) + "Model"
        )

        spec = importlib.util.spec_from_file_location(model_file, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Failed to load module spec for {model_file} from {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_class = getattr(module, class_name)

        # Extract split indices
        cstart, cend = dataset_splits["context"].start, dataset_splits["context"].end
        tstart, tend = dataset_splits["train"].start, dataset_splits["train"].end
        vstart, vend = dataset_splits["validate"].start, dataset_splits["validate"].end

        # Create and train model
        model = model_class(params=hyperparameters, settings=job_config.model_setting)

        trained_model = model.train(
            y_context=target[cstart:cend],
            y_target=target[tstart:tend],
            timestamps_context=timestamps[cstart:cend],
            timestamps_target=timestamps[tstart:tend],
            freq=freq,
        )

        # Generate predictions
        results = trained_model.predict(
            y_context=target[cstart:tend],
            timestamps_context=timestamps[cstart:tend],
            timestamps_target=timestamps[vstart:vend],
            freq=freq,
        )

        # Compute evaluation metrics
        eval_metrics = trained_model.compute_metrics(
            y_true=target[vstart:vend], y_pred=results
        )

        # Include predictions in output for plotting
        output = {
            **eval_metrics,
            "y_pred": results.tolist(),
            "y_true": target[vstart:vend].tolist(),
            "timestamps_pred": timestamps[vstart:vend].tolist(),
        }

        outputs.append(output)

    # Output results as JSON
    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
