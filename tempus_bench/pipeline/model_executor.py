"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that executes individual models
in isolated conda environments to avoid dependency conflicts.
It handles model execution with specific hyperparameters and returns evaluation results.

Can also be invoked directly from CLI as:
    python -m tempus_bench.pipeline.model_executor --model-name <name> --hyperparameters '{}' ...
"""

import argparse
import json
import os
import importlib.util
import yaml

from pathlib import Path
from typing import Any, Dict, Optional

from ..models.model_router import ModelRouter
from ..utils.envs import CondaEnvManager
from ..utils.logger import Logger
from ..utils.paths import get_models_dir


class ModelExecutor:
    """
    Execute a single model inside its dedicated Conda environment.

    The executor prepares the command-line invocation, ensures the requested environment
    is available, and parses JSON results emitted by the child process.
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
        task_path: str,
        window_idx: int,
    ) -> dict:
        """
        Execute a single model with specific hyperparameters on a specific dataset window.

        Args:
            model_name: Registered name of the model implementation to execute.
            hyperparameters: Concrete hyperparameter assignment to forward to the model.
            context_steps: Number of context steps extracted from each window.
            train_steps: Number of steps used for fitting inside each window.
            validate_steps: Number of steps reserved for evaluation.
            task_path: Absolute path to the dataset folder that holds the CSV payload.
            window_idx: Index of the window to use for execution.

        Returns:
            dict: Evaluation metrics and optional artifacts produced by the model command.
        """
        self.logger.info(
            "ModelExecutor",
            f"Executing model {model_name} with hyperparameters {hyperparameters}",
        )

        # Create Conda Environment
        requirements_path = self._get_model_requirements(model_name=model_name)
        python_version = self.model_settings[model_name]["python_version"]
        conda_env = CondaEnvManager(
            name=f"benchmark.{model_name}",
            python=python_version,
            requirements_path=requirements_path,
        )

        # Build CLI command
        hyperparameters_json = json.dumps(hyperparameters)
        command = (
            f"python -m tempus_bench.pipeline.model_executor "
            f"--model-name {model_name} "
            f"--hyperparameters '{hyperparameters_json}' "
            f"--context-steps {context_steps} "
            f"--train-steps {train_steps} "
            f"--validate-steps {validate_steps} "
            f"--task-path {task_path} "
            f"--window-idx {window_idx} "
        )

        self.logger.debug("ModelExecutor", f"Running command: {command}")

        result = conda_env.run(command=command)

        self.logger.success(
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

    def _get_model_requirements(self, model_name: str):
        """
        Returns the absolute path to requirements.txt for the requested model.
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
    Uses DataLoader to handle complex data formats.
    """
    parser = argparse.ArgumentParser(
        description="Execute a forecasting model with specific hyperparameters on a dataset window"
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
    parser.add_argument("--task-path", required=True, help="Path to the dataset file")
    parser.add_argument(
        "--window-idx", type=int, required=True, help="Index of the window to use"
    )
    parser.add_argument(
        "--config-path", required=True, help="Path to configuration file"
    )

    args = parser.parse_args()

    # Parse hyperparameters JSON
    hyperparameters = json.loads(args.hyperparameters)

    # Find matching job config for the task
    task_name = Path(args.task_path).parent.name
    matching_jobs = [
        job_config
        for job_config, task_idx in manager.generate_run_configs()
        if job_config.task_config.name == task_name
    ]

    if not matching_jobs:
        raise ValueError(f"No job config found for task {task_name}")

    job_config = matching_jobs[0]

    # Use DataLoader to handle data loading properly
    from .data_loader import DataLoader

    data_loader = DataLoader(job_config)

    # Generate windows
    steps = [
        ("context", args.context_steps),
        ("train", args.train_steps),
        ("validate", args.validate_steps),
    ]
    window_iter = data_loader.generate_dataset_split(args.task_path, steps, stride=1)

    # Get the specific window
    window_found = False
    for idx, (win_idx, window) in enumerate(window_iter):
        if idx == args.window_idx:
            window_found = True
            timestamps = window.timestamps
            target = window.target
            freq = window.metadata["freq"]

            # Import model - models are now directly in the models directory
            models_dir = get_models_dir()
            model_dir = models_dir / args.model_name
            model_file = f"{args.model_name}_model"
            module_path = str(model_dir / f"{model_file}.py")

            # Generate class name (PascalCase + Model suffix)
            class_name = (
                "".join(word.capitalize() for word in args.model_name.split("_"))
                + "Model"
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
            cstart, cend = window.context.start, window.context.end
            tstart, tend = window.train.start, window.train.end
            vstart, vend = window.validate.start, window.validate.end

            # Create and train model
            model = model_class(params=hyperparameters)

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
            eval_losses = trained_model.compute_loss(
                y_true=target[vstart:vend], y_pred=results
            )

            # Include predictions in output for plotting
            output = {
                **eval_losses,
                "predictions": results.tolist(),
                "y_true": target[vstart:vend].tolist(),
            }

            # Output results as JSON
            print(json.dumps(output))
            break

    if not window_found:
        raise Exception(
            f"Window {args.window_idx} not found for dataset {args.task_path}"
        )


if __name__ == "__main__":
    main()
