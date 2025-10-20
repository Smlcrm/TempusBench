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
            hyper_grid: list[dict]):

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

                        # Extract split indices
                        cstart, cend = window.context.start, window.context.end
                        tstart, tend = window.train.start, window.train.end
                        vstart, vend = window.validation.start, window.validation.end
                        freq = window.metadata['freq']

                        tuning_losses = dict()
                        eval_metrics = dict()

                        # Try each hyperparameter combination
                        for params in {repr(hyper_grid)}:
                            # Update the config with the hyperparameters before creating the model
                            full_config_copy = copy.deepcopy(config)
                            # Replace the entire model section with only the current model
                            full_config_copy["model"] = {{model_name: params}}
                            model = model_class(full_config_copy)
                            
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
                                y_pred=results,
                                y_train=target[tstart:tend],
                                freq=freq
                            )

                            del trained_model
                            del model
                            immutable_params = tuple(sorted(params.items()))
                            tuning_losses[immutable_params] = eval_losses[tuning_loss]
                            eval_metrics[immutable_params] = eval_losses

                        # Find the hyperparams with lowest tuning_loss for this window
                        best_params = min(tuning_losses, key=lambda k: tuning_losses[k])
                        optimal_hyperparameters.append(best_params)
                        evaluations.append(eval_metrics)
                        num_windows += 1

                    if num_windows == 0:
                        raise Exception(f"No windows for {{dataset_path}}")

                    # Aggregate test loss over all windows, for each metric
                    test_loss = {{ metric: [] for metric in evaluation_metrics }}
                    for window_j in range(num_windows-1):
                        best_params_prev = optimal_hyperparameters[window_j]
                        for metric in evaluation_metrics:
                            test_loss[metric].append(
                                evaluations[window_j+1][best_params_prev][metric]
                            )
                    avg_test_loss = {{ metric: float(np.mean(test_loss[metric])) if test_loss[metric] else float('nan') for metric in evaluation_metrics }}

                    # Write to evaluations CSV in run_dir/evals
                    csv_filename = f"evaluations.csv"
                    evals_dir = os.path.join(run_dir, "evals")
                    os.makedirs(evals_dir, exist_ok=True)
                    csv_outpath = os.path.join(evals_dir, csv_filename)
                    file_exists = os.path.exists(csv_outpath)
                    row = [model_name, dataset_path] + [avg_test_loss[metric] for metric in evaluation_metrics] + [str(optimal_hyperparameters)]
                    with open(csv_outpath, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists: # write header
                            writer.writerow(["model_name", "dataset_path"] + [f"avg_test_{{metric}}" for metric in evaluation_metrics] + ["best_params"])
                        writer.writerow(row)

            if __name__ == "__main__":
                main()
        """)
        return script

    def _is_valid_combination(self, model_name: str, combination: dict) -> bool:
        """
        Check if a hyperparameter combination is valid for the given model.
        
        Args:
            model_name: Name of the model
            combination: Dictionary of hyperparameter values
            
        Returns:
            bool: True if combination is valid, False otherwise
        """
        if model_name == "exponential_smoothing":
            # For exponential smoothing, if seasonal is not null, seasonal_periods must not be null
            seasonal = combination.get("seasonal")
            seasonal_periods = combination.get("seasonal_periods")
            
            if seasonal is not None and seasonal != "null" and seasonal_periods is None:
                return False
            if seasonal is None or seasonal == "null":
                if seasonal_periods is not None and seasonal_periods != "null":
                    return False
            
            # At least one of trend or seasonal must be specified (not null)
            trend = combination.get("trend")
            if (trend is None or trend == "null") and (seasonal is None or seasonal == "null"):
                return False
                
        elif model_name == "arima":
            # For ARIMA, if seasonal component s > 0, then p, d, q should be reasonable
            s = combination.get("s")
            p = combination.get("p")
            d = combination.get("d")
            q = combination.get("q")
            
            # Basic validation: p, d, q should be non-negative
            if p < 0 or d < 0 or q < 0 or s < 0:
                return False
                
        elif model_name == "theta":
            # For theta, sp should be positive
            sp = combination.get("sp")
            if sp is not None and sp <= 0:
                return False
                
        elif model_name == "seasonal_naive":
            # For seasonal naive, sp should be positive
            sp = combination.get("sp")
            if sp is not None and sp <= 0:
                return False
                
        elif model_name == "croston_classic":
            # For croston classic, alpha and gamma should be between 0 and 1
            alpha = combination.get("alpha")
            gamma = combination.get("gamma")
            if alpha is not None and (alpha <= 0 or alpha >= 1):
                return False
            if gamma is not None and (gamma <= 0 or gamma >= 1):
                return False
                
        return True

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
                    # Check for incompatible combinations and log warnings
                    if self._is_valid_combination(model_name, combination):
                        hyper_grid.append(combination)
                    else:
                        if logging:
                            self.logger.warning(f"Skipping incompatible parameter combination for {model_name}: {combination}")
            else:
                # For foundation models with no hyperparameters, use empty dict
                hyper_grid = [{}]

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

            script = self._generate_hyperparameter_tuning_script(
                model_name=model_name,
                context_steps=context_steps,
                train_steps=train_steps,
                validate_steps=validate_steps,
                hyper_grid=hyper_grid
            )
            if logging: self.logger.debug(f'Running Script: """\n{script}\n"""')

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                result = conda_env.run(script=script_path)
                if logging: self.logger.success(f'Script ran successfully for model {model_name}')

            finally:
                # Clean up temporary file
                os.unlink(script_path)

        return all_evals, best_hyperparameters
