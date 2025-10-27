import os
import csv
import json
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Tuple

from tempus_bench.utils.logger import get_logger
from tempus_bench.utils.tf_logger import get_tf_logger
from tempus_bench.utils.paths import get_tasks_dir, get_project_root
from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.config import load_config
from tempus_bench.pipeline.model_executor import ModelExecutor

class HyperparameterTuner:
    def __init__(self, config_path: str, run_path: str, task_config):
        """
        Initialize the hyperparameter tuner with configuration and directories.

        Args:
            config_path: Path to configuration file
            run_path: Directory for run outputs
            task_config: Task configuration (TaskConfig instance from ConfigManager)
        """
        self.config = load_config(config_path, str(Path(run_path) / 'logs'))
        self.config_path = config_path
        self.tasks_dir = get_tasks_dir()
        self.run_path = run_path
        self.task_config = task_config
        self.logger = get_logger(str(Path(run_path) / 'logs'))
        self.tf_logger = get_tf_logger(str(Path(run_path) / 'tensorboard'))

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

    def _generate_hyperparameter_grid(self, model_name: str, hyperparameters: dict) -> List[dict]:
        """
        Generate hyperparameter grid for a model.
        
        Args:
            model_name: Name of the model
            hyperparameters: Dictionary of hyperparameter lists
            
        Returns:
            List of valid hyperparameter combinations
        """
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
                    if self.logger:
                        self.logger.warning("HyperparameterTuner", f"Skipping incompatible parameter combination for {model_name}: {combination}")
        else:
            # For foundation models with no hyperparameters, use empty dict
            hyper_grid = [{}]
        
        return hyper_grid

    def optimize_hyperparameters(self, context_steps: int, train_steps: int, validate_steps: int) -> Tuple[dict, dict]:
        """
        Optimize hyperparameters for all models in the configuration.
        
        Args:
            context_steps: Number of context steps
            train_steps: Number of training steps
            validate_steps: Number of validation steps
        
        Returns:
            Tuple of (all_evals, best_hyperparameters) - evaluation results and best hyperparameters
        """
        logging = self.config['logging']['console_logging']
        all_evals = {}
        best_hyperparameters = {}

        # Extract configuration parameters
        tuning_loss = self.config['evaluation']['tuning_loss']
        evaluation_metrics = self.config['evaluation']['metrics']

        # Initialize data loader
        data_loader = DataLoader(
            config_path=self.config_path,
            run_path=self.run_path
        )

        # Initialize model executor
        model_executor = ModelExecutor(
            config_path=self.config_path,
            run_path=self.run_path
        )

        for model_name, hyperparameters in self.config["model"].items():
            # Skip models that are None (not configured)
            if hyperparameters is None:
                continue
                
            if logging:
                self.logger.info("HyperparameterTuner", f"Optimizing hyperparameters for model: {model_name}")
                self.logger.debug("HyperparameterTuner", f"Hyperparameters for {model_name}: {hyperparameters}")
                if not hyperparameters:
                    self.logger.warning("HyperparameterTuner", f"{model_name} has no parameters, using empty dict, probably a foundation model")

            # Generate hyperparameter grid
            hyper_grid = self._generate_hyperparameter_grid(model_name, hyperparameters)
            
            # Store results for this model
            model_evals = {}
            model_best_params = {}

            # Process each dataset
            for dataset_path in data_loader.dataset_paths:
                if logging:
                    self.logger.info("HyperparameterTuner", f"Processing dataset: {dataset_path}")

                # Generate windows for this dataset
                steps = [('context', context_steps), ('train', train_steps), ('validate', validate_steps)]
                window_iter = data_loader.generate_dataset_split(
                    dataset_path, steps, stride=validate_steps
                )

                # Store results for each window
                window_results = []
                optimal_hyperparameters = []
                evaluations = []
                num_windows = 0

                # For each rolling window
                for window_idx, window in window_iter:
                    if logging:
                        self.logger.debug("HyperparameterTuner", f"Processing window {window_idx} for dataset {dataset_path}")

                    tuning_losses = {}
                    eval_metrics = {}

                    # Try each hyperparameter combination
                    for params in hyper_grid:
                        try:
                            # Execute model with these hyperparameters
                            eval_losses = model_executor.execute_model(
                                model_name=model_name,
                                hyperparameters=params,
                                context_steps=context_steps,
                                train_steps=train_steps,
                                validate_steps=validate_steps,
                                dataset_path=dataset_path,
                                window_idx=window_idx
                            )

                            immutable_params = tuple(sorted(params.items()))
                            tuning_losses[immutable_params] = eval_losses[tuning_loss]
                            eval_metrics[immutable_params] = eval_losses
                            
                            # Log hyperparameters and metrics to TensorBoard
                            self.tf_logger.log_hparams(params, eval_losses)

                        except Exception as e:
                            if logging:
                                self.logger.error("HyperparameterTuner", f"Error executing model {model_name} with params {params}: {e}")
                            continue

                    # Find the hyperparams with lowest tuning_loss for this window
                    if tuning_losses:
                        best_params = min(tuning_losses, key=lambda k: tuning_losses[k])
                        optimal_hyperparameters.append(best_params)
                        evaluations.append(eval_metrics)
                        num_windows += 1
                        
                        # Generate forecast plot for best hyperparameters
                        self._generate_forecast_plot(
                            model_name=model_name,
                            hyperparameters=dict(best_params),
                            context_steps=context_steps,
                            train_steps=train_steps,
                            validate_steps=validate_steps,
                            dataset_path=dataset_path,
                            window_idx=window_idx
                        )

                if num_windows == 0:
                    if logging:
                        self.logger.warning("HyperparameterTuner", f"No valid windows for dataset {dataset_path}")
                    continue

                # Aggregate test loss over all windows, for each metric
                test_loss = {metric: [] for metric in evaluation_metrics}
                for window_j in range(num_windows-1):
                    best_params_prev = optimal_hyperparameters[window_j]
                    for metric in evaluation_metrics:
                        if best_params_prev in evaluations[window_j+1]:
                            test_loss[metric].append(
                                evaluations[window_j+1][best_params_prev][metric]
                            )
                
                avg_test_loss = {
                    metric: float(np.mean(test_loss[metric])) if test_loss[metric] else float('nan') 
                    for metric in evaluation_metrics
                }

                # Write to evaluations CSV in run_path/evals
                csv_filename = f"evaluations.csv"
                evals_dir = Path(self.run_path) / "evals"
                evals_dir.mkdir(exist_ok=True)
                csv_outpath = evals_dir / csv_filename
                file_exists = csv_outpath.exists()
                row = [model_name, dataset_path] + [avg_test_loss[metric] for metric in evaluation_metrics] + [str(optimal_hyperparameters)]
                with open(csv_outpath, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:  # write header
                        writer.writerow(["model_name", "dataset_path"] + [f"avg_test_{metric}" for metric in evaluation_metrics] + ["best_params"])
                    writer.writerow(row)

                # Store results for this dataset
                model_evals[dataset_path] = avg_test_loss
                model_best_params[dataset_path] = optimal_hyperparameters

            # Store results for this model
            all_evals[model_name] = model_evals
            best_hyperparameters[model_name] = model_best_params

        if logging:
            self.logger.success("HyperparameterTuner", "Hyperparameter optimization completed")

        return all_evals, best_hyperparameters

    def _generate_forecast_plot(self, model_name, hyperparameters, context_steps, train_steps, validate_steps, dataset_path, window_idx):
        """
        Generate time series forecast plot showing context, train, validate data and predictions.
        """
        try:
            # Import here to avoid circular imports
            from .data_loader import DataLoader
            from .model_executor import ModelExecutor
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            
            # Create data loader to get the window data
            data_loader = DataLoader(
                config_path=self.config_path,
                run_path=self.run_path
            )
            
            # Get the specific window data
            steps = [('context', context_steps), ('train', train_steps), ('validate', validate_steps)]
            window_iter = data_loader.generate_dataset_split(dataset_path, steps, stride=1)
            
            # Find the specific window
            window_data = None
            for idx, window in window_iter:
                if idx == window_idx:
                    window_data = window
                    break
            
            if window_data is None:
                if self.logger:
                    self.logger.warning("HyperparameterTuner", f"Window {window_idx} not found for plotting")
                return
            
            # Create model executor to get predictions
            model_executor = ModelExecutor(
                config_path=self.config_path,
                run_path=self.run_path,
                datasets_dir=self.datasets_dir
            )
            
            # Execute model to get predictions
            eval_results = model_executor.execute_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                context_steps=context_steps,
                train_steps=train_steps,
                validate_steps=validate_steps,
                dataset_path=dataset_path,
                window_idx=window_idx
            )
            
            # Create plots directory
            plots_dir = Path(self.run_path) / 'tensorboard' / 'plots' / model_name
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Get data from window using indices
            context_data = window_data.target[window_data.context.start:window_data.context.end]
            train_data = window_data.target[window_data.train.start:window_data.train.end]
            validate_data = window_data.target[window_data.validate.start:window_data.validate.end]
            
            # Get predictions from model execution
            eval_results = model_executor.execute_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                context_steps=context_steps,
                train_steps=train_steps,
                validate_steps=validate_steps,
                dataset_path=dataset_path,
                window_idx=window_idx
            )
            
            # Extract actual predictions from results
            predictions = np.array(eval_results.get('predictions'))
            y_true_validate = np.array(eval_results.get('y_true'))
            
            # Create subplots for each target
            num_targets = context_data.shape[1] if context_data.ndim > 1 else 1
            fig, axes = plt.subplots(num_targets, 1, figsize=(15, 4*num_targets))
            if num_targets == 1:
                axes = [axes]
            
            # Create continuous time indices for smooth plotting
            context_len = len(context_data)
            train_len = len(train_data)
            validate_len = len(validate_data)
            
            # Create continuous time series by concatenating all segments
            if context_data.ndim == 1:
                full_data = np.concatenate([context_data, train_data, validate_data])
            else:
                full_data = np.concatenate([context_data, train_data, validate_data], axis=0)
            
            # Create continuous time indices
            full_time = np.arange(len(full_data))
            context_time = full_time[:context_len]
            train_time = full_time[context_len:context_len + train_len]
            validate_time = full_time[context_len + train_len:]
            
            # For each target, create a subplot
            for target_idx in range(num_targets):
                ax = axes[target_idx]
                
                # Plot context data
                if context_data.ndim == 1:
                    ax.plot(context_time, context_data, 'b-', label='Context', linewidth=2, alpha=0.8)
                else:
                    ax.plot(context_time, context_data[:, target_idx], 'b-', label='Context', linewidth=2, alpha=0.8)
                
                # Plot training data
                if train_data.ndim == 1:
                    ax.plot(train_time, train_data, 'g-', label='Train', linewidth=2, alpha=0.8)
                else:
                    ax.plot(train_time, train_data[:, target_idx], 'g-', label='Train', linewidth=2, alpha=0.8)
                
                # Plot validation data (true values)
                if validate_data.ndim == 1:
                    ax.plot(validate_time, validate_data, 'r-', label='True Values', linewidth=2, alpha=0.8)
                else:
                    ax.plot(validate_time, validate_data[:, target_idx], 'r-', label='True Values', linewidth=2, alpha=0.8)
                
                # Plot actual model predictions
                if len(predictions) > 0:
                    if predictions.ndim == 1:
                        ax.plot(validate_time, predictions, 'orange', linestyle='--', 
                               label='Model Predictions', linewidth=2, alpha=0.8)
                    else:
                        ax.plot(validate_time, predictions[:, target_idx], 'orange', linestyle='--', 
                               label='Model Predictions', linewidth=2, alpha=0.8)
                else:
                    # Fallback to simple prediction if no actual predictions available
                    if validate_data.ndim == 1:
                        last_train_val = train_data[-1] if len(train_data) > 0 else context_data[-1]
                        fallback_predictions = np.full_like(validate_data, last_train_val)
                        ax.plot(validate_time, fallback_predictions, 'orange', linestyle='--', 
                               label='Predictions (Fallback)', linewidth=2, alpha=0.8)
                    else:
                        last_train_vals = train_data[-1, :] if len(train_data) > 0 else context_data[-1, :]
                        fallback_predictions = np.full_like(validate_data, last_train_vals)
                        ax.plot(validate_time, fallback_predictions[:, target_idx], 'orange', linestyle='--', 
                               label='Predictions (Fallback)', linewidth=2, alpha=0.8)
                
                # Customize subplot
                ax.set_title(f'{model_name} - Target {target_idx + 1} (Window {window_idx})')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Add hyperparameters info to the figure
            fig.suptitle(f'Best Hyperparameters: {hyperparameters}', fontsize=12, y=0.98)
            
            # Save plot
            plot_path = plots_dir / f'window_{window_idx}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to TensorBoard
            self.tf_logger.log_image_file(
                image_path=str(plot_path),
                tag=f'{model_name}/forecast',
                step=window_idx
            )
            
            if self.logger:
                self.logger.info("HyperparameterTuner", f"Generated time series plot for {model_name} window {window_idx}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error("HyperparameterTuner", f"Error generating time series plot for {model_name}: {e}")


