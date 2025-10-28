"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that executes individual models
in isolated conda environments to avoid dependency conflicts.
It handles model execution with specific hyperparameters and returns evaluation results.
"""

import os
import textwrap
import importlib
import tempfile
import pandas as pd

from pathlib import Path
from typing import Dict, Any

from tempus_bench.utils.logger import get_logger
from tempus_bench.utils.envs import CondaEnvManager
from tempus_bench.utils.tf_logger import get_tf_logger
from tempus_bench.utils.paths import get_tasks_dir, get_models_dir, get_project_root
from tempus_bench.models.model_router import ModelRouter
from tempus_bench.utils.model_config import get_python_version_for_model
from tempus_bench.config import load_config, get_config_manager

class ModelExecutor:
    def __init__(self, config_path: str, logs_path: str):
        self.config = load_config(config_path, logs_path)
        self.config_path = config_path
        self.logs_path = logs_path
        self.tasks_dir = get_tasks_dir()
        self.models_dir = get_models_dir()
        self.config_manager = get_config_manager()
        console_logging = self.config_manager.benchmark_settings.console_logging
        file_logging = self.config_manager.benchmark_settings.file_logging
        self.logger = get_logger(logs_path, console_logging=console_logging, file_logging=file_logging)
        self.tf_logger = get_tf_logger(str(Path(logs_path).parent / 'tensorboard'), tensorboard_logging=self.config_manager.benchmark_settings.tensorboard_logging)

    def _generate_model_execution_script(self,
        model_name: str,
        hyperparameters: dict,
        context_steps: int,
        train_steps: int,
        validate_steps: int,
        task_path: str,
        window_idx: int):

        script = textwrap.dedent(f"""
            import os
            import sys
            import json
            import copy
            import numpy as np
            import pandas as pd
            import importlib.util
            from tempus_bench.models.model_router import ModelRouter
            from tempus_bench.pipeline.data_loader import DataLoader

            def main():
                data_loader = DataLoader(
                    config_path={repr(self.config_path)},
                    logs_path={repr(self.logs_path)}
                )

                config = data_loader.config
                tuning_loss = config['evaluation']['tuning_loss']
                evaluation_metrics = config['evaluation']['metrics']
                model_name = {repr(model_name)}
                hyperparameters = {repr(hyperparameters)}
                task_path = {repr(task_path)}
                window_idx = {window_idx}

                # Generate the specific window
                steps = [('context', {context_steps}), ('train', {train_steps}), ('validate', {validate_steps})]
                window_iter = data_loader.generate_task_split(
                    task_path, steps, stride=1
                )

                # Get the specific window
                for idx, (win_idx, window) in enumerate(window_iter):
                    if idx == window_idx:
                        timestamps = window.timestamps
                        target = window.target

                        # Import model here, so we know target shape
                        router = ModelRouter(logs_path={repr(self.logs_path)})
                        task_type = config['task']['task_type']
                        folder_path, file_name, class_name = router.get_model_path_by_task_type(
                            model_name, task_type
                        )
                        module_path = str(Path(folder_path) / f"{{file_name}}.py")
                        spec = importlib.util.spec_from_file_location(file_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        model_class = getattr(module, class_name)

                        # Extract split indices
                        cstart, cend = window.context.start, window.context.end
                        tstart, tend = window.train.start, window.train.end
                        vstart, vend = window.validate.start, window.validate.end
                        freq = window.metadata['freq']

                        # Create model with config_path and hyperparameters
                        model = model_class(config_path={repr(self.config_path)}, logs_path={repr(self.logs_path)}, hyperparameters=hyperparameters)

                        # Set the scaler for inverse transformation if available
                        if hasattr(window, 'scaler') and window.scaler is not None:
                            model.set_scaler(window.scaler)

                        trained_model = model.train(
                            y_context=target[cstart:cend],
                            y_target=target[tstart:tend],
                            timestamps_context=timestamps[cstart:cend],
                            timestamps_target=timestamps[tstart:tend],
                            freq=freq
                        )

                        results = trained_model.predict(
                            y_context=target[cstart:tend],
                            timestamps_context=timestamps[cstart:tend],
                            timestamps_target=timestamps[vstart:vend],
                            freq=freq,
                        )

                        eval_losses = trained_model.compute_loss(
                            y_true=target[vstart:vend],
                            y_pred=results
                        )

                        # Include predictions in output for plotting
                        output = {{
                            **eval_losses,
                            "predictions": results.tolist(),
                            "y_true": target[vstart:vend].tolist()
                        }}

                        # Output results as JSON
                        print(json.dumps(output))
                        return

                raise Exception(f"Window {{window_idx}} not found for dataset {{task_path}}")

            if __name__ == "__main__":
                main()
        """)
        return script

    def execute_model(self,
        model_name: str,
        hyperparameters: dict,
        context_steps: int,
        train_steps: int,
        validate_steps: int,
        task_path: str,
        window_idx: int) -> dict:
        """
        Execute a single model with specific hyperparameters on a specific dataset window.

        Args:
            model_name: Name of the model to execute
            hyperparameters: Dictionary of hyperparameter values
            context_steps: Number of context steps
            train_steps: Number of training steps
            validate_steps: Number of validation steps
            task_path: Path to the dataset file
            window_idx: Index of the window to use

        Returns:
            dict: Evaluation metrics for the model execution
        """
        logging = self.config['logging']['console_logging']

        if logging:
            self.logger.info("ModelExecutor", f"Executing model {model_name} with hyperparameters {hyperparameters}")

        # Create Conda Environment
        requirements_path = self._get_model_requirements(model_name=model_name)
        python_version = get_python_version_for_model(model_name)
        conda_env = CondaEnvManager(
            name=f"benchmark.{model_name}",
            python=python_version,
            requirements_path=requirements_path
        )

        script = self._generate_model_execution_script(
            model_name=model_name,
            hyperparameters=hyperparameters,
            context_steps=context_steps,
            train_steps=train_steps,
            validate_steps=validate_steps,
            task_path=task_path,
            window_idx=window_idx
        )

        if logging:
            self.logger.debug("ModelExecutor", f'Running Script: """\n{script}\n"""')

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = conda_env.run(script=script_path)
            if logging:
                self.logger.success("ModelExecutor", f'Script ran successfully for model {model_name}')
            # Parse the JSON output from the script
            import json
            # Find the last line that contains JSON (the script outputs debug info first)
            lines = result.stdout.strip().split('\n')
            json_line = None
            for line in reversed(lines):
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    json_line = line.strip()
                    break

            if json_line is None:
                raise ValueError(f"No valid JSON found in script output. Output was: {result.stdout}")

            eval_results = json.loads(json_line)
            return eval_results

        finally:
            # Clean up temporary file
            os.unlink(script_path)

    def _get_model_requirements(self, model_name: str):
        """
        Returns the absolute path to requirements.txt for the requested model.
        Uses the model router to determine the correct path based on task type.
        """
        # Get task type from config
        task_type = self.config['task']['task_type']

        # Use model router to get the correct path
        router = ModelRouter(logs_path=self.logs_path)
        folder_path, file_name, class_name = router.get_model_path_by_task_type(
            model_name, task_type
        )

        # Construct requirements path
        req_path = Path(folder_path) / "requirements.txt"
        if not req_path.exists():
            raise FileNotFoundError(f"requirements.txt not found at expected path: {req_path}")
        return str(req_path.resolve())

