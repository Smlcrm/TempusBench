"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that runs individual models
in isolated conda environments to avoid dependency conflicts.
It handles model loading, hyperparameter tuning, and evaluation.
"""

import os
import pdb
import textwrap
import argparse
import importlib
import subprocess
import numpy as np
import pickle, csv, json, yaml
import tempfile
import pandas as pd

from pathlib import Path
from typing import Dict, Any
from itertools import product
from datetime import datetime

from benchmarking_pipeline.utils.envs import CondaEnvManager
from benchmarking_pipeline.utils.logger import Logger
from benchmarking_pipeline.models.base_model import BaseModel
from benchmarking_pipeline.models.model_router import ModelRouter
from benchmarking_pipeline.utils.config_validator import load_config

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class ModelExecutor:
    def __init__(self, config_path: str, run_dir: str, datasets_dir: str, logger=None, tf_logger=None):
        self.config = load_config(config_path)
        self.config_path = config_path
        self.run_dir = run_dir
        self.datasets_dir = datasets_dir
        self.logger = logger
        self.tf_logger = tf_logger

    def _generate_hyperparameter_tuning_script(self,
            model_name: str,
            context_steps: int,
            train_steps: int,
            validate_steps: int,
            hyper_grid: list[dict],
            csv_path: str):

        # Create CSV headers
        csv_headers = ['model_name', 'dataset_path', 'batch_idx', 'row_idx', 'window_idx', 'hyper_parameters']
        tuning_loss = self.config["task"]["tuning_loss"]
        csv_headers.extend([f'{tuning_loss}'] + [f'{metric}' for metric in self.config["evaluation"]["metrics"]])

        # This rewrite fixes the f-string syntax error by avoiding double braces in f-strings (see @file_context_0)
        # and fixes dynamic target/variable issues by moving model import into the split loop where target shape is known.
        # It also ensures run_dir exists, and makes the script ready for stand-alone execution in another Python process.

        script = textwrap.dedent(f"""
            import os
            import sys
            import csv
            import json
            import copy
            import numpy as np
            import pandas as pd
            import importlib.util
            from benchmarking_pipeline.models.model_router import ModelRouter
            from benchmarking_pipeline.pipeline.data_loader import DataLoader

            def main():
                data_loader = DataLoader(
                    config_path={repr(self.config_path)},
                    datasets_dir={repr(self.datasets_dir)},
                    run_dir={repr(self.run_dir)}
                )

                config = data_loader.config
                tuning_loss = config['task']['tuning_loss']
                evaluation_metrics = config['evaluation']['metrics']
                run_dir = {repr(self.run_dir)}
                model_name = {repr(model_name)}

                for dataset_path in data_loader.dataset_paths:

                    # Will collect best hyperparameters and evaluation metrics for each window
                    optimal_hyperparameters = []
                    evaluations = []
                    num_windows = 0

                    window_iter = data_loader.generate_dataset_split(
                        dataset_path, {context_steps}, {train_steps}, {validate_steps}
                    )

                    # For each rolling window
                    for window_idx, window in window_iter:

                        timestamps = window.timestamps
                        target = window.target

                        # Import model here, so we know target shape
                        router = ModelRouter()
                        folder_path, file_name, class_name = router.get_model_path_by_target_count(model_name, target.shape[1] if len(target.shape) > 1 else 1)
                        module_path = os.path.join(folder_path, f"{{file_name}}.py")
                        spec = importlib.util.spec_from_file_location(file_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        model_class = getattr(module, class_name)
                        full_config = copy.deepcopy(config)
                        model = model_class(full_config)

                        # Extract split indices
                        cstart, cend = window.context.start, window.context.end
                        tstart, tend = window.train.start, window.train.end
                        vstart, vend = window.validation.start, window.validation.end
                        freq = window.metadata['freq']

                        tuning_losses = dict()
                        eval_metrics = dict()

                        # Try each hyperparameter combination
                        for params in {hyper_grid}:
                            if params:
                                model.model_config = params

                            trained_model = model.train(
                                y_context=target[cstart:cend, :],
                                y_target=target[tstart:tend, :],
                                timestamps_context=timestamps[cstart:cend],
                                timestamps_target=timestamps[tstart:tend],
                                freq=freq
                            )

                            results = trained_model.predict(
                                y_context=target[cstart:tend, :],
                                timestamps_context=timestamps[cstart:tend],
                                timestamps_target=timestamps[vstart:vend],
                                freq=freq,
                            )

                            eval_losses = trained_model.compute_loss(
                                y_true=target[vstart:vend, :],
                                y_pred=results,
                                y_train=target[tstart:tend, :]
                            )

                            immutable_params = tuple(sorted(params.items()))
                            tuning_losses[immutable_params] = eval_losses[tuning_loss]
                            eval_metrics[immutable_params] = eval_losses

                        # Find the hyperparams with lowest tuning_loss for this window
                        best_params = min(tuning_losses, key=lambda k: tuning_losses[k])
                        optimal_hyperparameters.append(best_params)
                        evaluations.append(eval_metrics)
                        num_windows += 1

                    # Aggregate test loss over all windows, for each metric
                    test_loss = {{ metric: [] for metric in evaluation_metrics }}
                    for window_j in range(num_windows-1):
                        best_params_prev = optimal_hyperparameters[window_j]
                        for metric in evaluation_metrics:
                            test_loss[metric].append(
                                evaluations[window_j+1][best_params_prev][metric]
                            )
                    avg_test_loss = {{ metric: float(np.mean(test_loss[metric])) if test_loss[metric] else float('nan') for metric in evaluation_metrics }}

                    # Construct CSV row
                    header = ["dataset_path"] + [f"avg_test_{{metric}}" for metric in evaluation_metrics] + ["best_params"]
                    row = [dataset_path] + [avg_test_loss[metric] for metric in evaluation_metrics] + [str(optimal_hyperparameters)]

                    # Write to evaluations CSV in run_dir/evals
                    csv_filename = f"evaluations_{{model_name}}.csv"
                    evals_dir = os.path.join(run_dir, "evals")
                    os.makedirs(evals_dir, exist_ok=True)
                    csv_outpath = os.path.join(evals_dir, csv_filename)
                    file_exists = os.path.exists(csv_outpath)
                    with open(csv_outpath, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(header)
                        writer.writerow(row)

            if __name__ == "__main__":
                main()
        """)
        return script

    def _get_model_requirements(self, model_name: str, modality: str):
        """
        Returns the absolute path to requirements.txt for the requested model.
        The structure is: @models/<modality>/<model_name>/requirements.txt
        """

        # Locate root benchmarking_pipeline directory (two levels up from here)
        models_dir = os.path.join(ROOT_DIR, "benchmarking_pipeline", "models")
        req_path = os.path.join(models_dir, modality, model_name, "requirements.txt")
        if not os.path.exists(req_path):
            raise FileNotFoundError(f"requirements.txt not found at expected path: {req_path}")
        return os.path.abspath(req_path)

    def _compute_optimal_hyperparameters(self, model_name: str, csv_path: str, tuning_loss: str) -> tuple:
        """
        Read validation losses CSV and compute optimal hyperparameters.

        Args:
            model_name: Name of the model
            csv_path: Path to the CSV file with validation losses
            tuning_loss: Metric to use for hyperparameter selection

        Returns:
            tuple: (best_hyperparameters, evaluation_results)
                - best_hyperparameters: dict with optimal hyperparameters
                - evaluation_results: dict with average metrics for best config
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV into pandas DataFrame
        df = pd.read_csv(csv_path)

        if df.empty:
            return {}, {}

        # Group by hyper_parameters column and compute mean of tuning_loss
        grouped = df.groupby('hyper_parameters')[tuning_loss].mean()

        # Find hyperparameters with lowest mean tuning_loss
        best_hyperparams_str = grouped.idxmin()

        # Parse the best hyperparameters from JSON string
        try:
            best_hyperparameters = json.loads(best_hyperparams_str) if best_hyperparams_str != 'null' else {}
        except (json.JSONDecodeError, TypeError):
            best_hyperparameters = {}

        # For best hyperparameters, compute mean of all evaluation metrics
        best_rows = df[df['hyper_parameters'] == best_hyperparams_str]
        evaluation_results = {}

        # Get all metric columns (excluding metadata columns)
        metric_columns = [col for col in df.columns if col not in
            ['model_name', 'dataset_path', 'batch_idx', 'row_idx', 'window_idx', 'hyper_parameters']]

        for metric in metric_columns:
            evaluation_results[metric] = best_rows[metric].mean()

        return best_hyperparameters, evaluation_results

    def optimize_hyperparameters(self,
            context_steps: int,
            train_steps: int,
            validate_steps: int):
        """
        Run model evaluation with the provided hyperparameters.

        Args:
            context_steps: Number of context steps
            train_steps: Number of training steps
            validate_steps: Number of validation steps

        Returns:
            tuple: (evals, hyperparameters) - evaluation results and best hyperparameters
        """
        logging = self.config['evaluation']['logging']
        all_evals = {}
        best_hyperparameters = {}

        # Extract configuration parameters
        tuning_loss = self.config['task']['tuning_loss']
        evaluation_metrics = self.config['evaluation']['metrics']

        for model_name, hyperparameters in self.config["model"].items():
            hyper_grid = []
            if hyperparameters:
                keys = list(hyperparameters.keys())
                values_lists = [hyperparameters[k] for k in keys]
                # Compute the Cartesian product of all hyperparameter value lists.
                # For each combination, create a dictionary mapping each key to its value.
                for values_tuple in product(*values_lists):
                    combination = dict(zip(keys, values_tuple))
                    hyper_grid.append(combination)
            else:
                hyper_grid = []

            if logging:
                self.logger.info(f"Preparing to run model: {model_name}")
                self.logger.debug(f"Hyperparameters for {model_name}: {hyperparameters}")
                if not hyperparameters:
                    self.logger.warning(f"{model_name} has no parameters, using empty dict, probably a foundation model")

            # Create Conda Env
            requirements_path = self._get_model_requirements(
                model_name=model_name,
                modality="anyvariate"
            )
            conda_env = CondaEnvManager(
                name = f"benchmark.{model_name}",
                python = "3.11",
                requirements_path = requirements_path
            )

            # Define CSV path for this model
            csv_path = os.path.join(self.run_dir, f'validation_losses_{model_name}.csv')

            script = self._generate_hyperparameter_tuning_script(
                model_name=model_name,
                context_steps=context_steps,
                train_steps=train_steps,
                validate_steps=validate_steps,
                hyper_grid=hyper_grid,
                csv_path=csv_path
            )
            if logging: self.logger.debug(f'Running Script: """\n{script}\n"""')

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                result = conda_env.run(script=script_path)
                if logging: self.logger.success(f'Script ran successfully')

                # Analyze CSV to find optimal hyperparameters
                best_params, eval_results = self._compute_optimal_hyperparameters(
                    model_name, csv_path, tuning_loss
                )

                # Store results
                best_hyperparameters[model_name] = best_params
                all_evals[model_name] = eval_results

                if logging:
                    self.logger.info(f"Best hyperparameters for {model_name}: {best_params}")
                    self.logger.info(f"Evaluation results for {model_name}: {eval_results}")

            finally:
                # Clean up temporary file
                os.unlink(script_path)

        return all_evals, best_hyperparameters
