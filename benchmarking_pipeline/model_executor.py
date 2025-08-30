"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that runs individual models
in isolated conda environments to avoid dependency conflicts.
It handles model loading, hyperparameter tuning, and evaluation.
"""

import argparse
import yaml
import pickle
import importlib
import os
from datetime import datetime

from benchmarking_pipeline.models.base_model import BaseModel
from benchmarking_pipeline.models.foundation_model import FoundationModel
import json

from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.trainer.foundation_model_tuning import FoundationModelTuner


class ModelExecutor:

    def __init__(
        self,
        config,
        chunk_path,
        model_folder_name,
        model_file_name,
        model_class_name,
        result_path=None,
    ):
        self.config = config
        self.chunk_path = chunk_path
        self.model_folder_name = model_folder_name
        self.model_file_name = model_file_name
        self.model_class_name = model_class_name
        # Prepare a results output path if provided via config
        self.result_path = result_path or self.config.get("result_path")

        # Setup TensorBoard logging like we had before
        self.setup_tensorboard_logging()

    def setup_tensorboard_logging(self):
        """Setup TensorBoard logging with the exact structure we had before."""
        # Only enable TensorBoard if config specifies tensorboard: true
        if not self.config.get("tensorboard", False):
            print("[INFO] TensorBoard logging disabled (tensorboard: false in config)")
            self.writer = None
            self.log_dir = None

            return

        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create runs directory structure like before: runs/MODEL_NAME/TIMESTAMP/train/
            model_name = self.model_folder_name.split("/")[-1]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Create the exact directory structure we had before
            runs_dir = "runs"
            model_runs_dir = os.path.join(runs_dir, model_name)
            timestamp_dir = os.path.join(model_runs_dir, timestamp)
            train_dir = os.path.join(timestamp_dir, "train")

            # Ensure directories exist
            os.makedirs(train_dir, exist_ok=True)

            # Create TensorBoard writer
            self.writer = SummaryWriter(train_dir)
            self.log_dir = train_dir
            print(f"[INFO] TensorBoard logging enabled at: {train_dir}")

        except ImportError:
            print("[WARNING] TensorBoard not available, logging disabled")
            self.writer = None
            self.log_dir = None
        except Exception as e:
            print(f"[WARNING] Failed to setup TensorBoard logging: {e}")
            self.writer = None
            self.log_dir = None

    def run(self):
        # Extract the model name from the folder path for parameter lookup
        # The folder path is now an absolute path like '/path/to/benchmarking_pipeline/models/multivariate/arima'
        # We need to extract just 'arima' for parameter lookup
        model_name = self.model_folder_name.split("/")[-1]

        # Build the module path for import
        # The model_folder_name is now an absolute path, so we need to extract the relative part
        # from the models directory to construct the module path
        if "/models/" in self.model_folder_name:
            # Extract the part after 'models/' to get the relative path
            models_index = self.model_folder_name.find("/models/")
            relative_path = self.model_folder_name[
                models_index + 8 :
            ]  # +8 to skip '/models/'
            # Replace forward slashes with dots for proper Python module path
            relative_path = relative_path.replace("/", ".")
            module_path = (
                f"benchmarking_pipeline.models.{relative_path}.{self.model_file_name}"
            )
        else:
            raise ValueError(
                f"Invalid model folder path: {self.model_folder_name}. Must contain '/models/'"
            )

        print(f"[INFO] Importing module: {module_path}")
        module = importlib.import_module(module_path)
        model_class = getattr(module, self.model_class_name)
        print(f"[INFO] Executing {model_name} model...")

        with open(self.chunk_path, "rb") as f:
            serializable_chunks = pickle.load(f)

        # Reconstruct Dataset objects from serializable chunks
        from benchmarking_pipeline.pipeline.data_types import Dataset, DatasetSplit
        import numpy as np

        all_dataset_chunks = []
        for chunk_data in serializable_chunks:
            # Convert back to numpy arrays and reconstruct Dataset objects
            # For univariate data, ensure targets are 1D arrays (time series)
            train_targets = np.array(chunk_data["train"]["targets"])
            if train_targets.ndim == 2 and train_targets.shape[0] == 1:
                # If shape is (1, time_steps), squeeze to (time_steps,)
                train_targets = train_targets.squeeze()

            validation_targets = np.array(chunk_data["validation"]["targets"])
            if validation_targets.ndim == 2 and validation_targets.shape[0] == 1:
                validation_targets = validation_targets.squeeze()

            test_targets = np.array(chunk_data["test"]["targets"])
            if test_targets.ndim == 2 and test_targets.shape[0] == 1:
                test_targets = test_targets.squeeze()

            train_split = DatasetSplit(
                targets=train_targets,
                features=(
                    np.array(chunk_data["train"]["features"])
                    if chunk_data["train"]["features"] is not None
                    else None
                ),
                timestamps=np.array(chunk_data["train"]["timestamps"]),
            )

            validation_split = DatasetSplit(
                targets=validation_targets,
                features=(
                    np.array(chunk_data["validation"]["features"])
                    if chunk_data["validation"]["features"] is not None
                    else None
                ),
                timestamps=np.array(chunk_data["validation"]["timestamps"]),
            )

            test_split = DatasetSplit(
                targets=test_targets,
                features=(
                    np.array(chunk_data["test"]["features"])
                    if chunk_data["test"]["features"] is not None
                    else None
                ),
                timestamps=np.array(chunk_data["test"]["timestamps"]),
            )

            dataset = Dataset(
                train=train_split,
                validation=validation_split,
                test=test_split,
                name=chunk_data["name"],
                metadata=chunk_data["metadata"],
            )
            all_dataset_chunks.append(dataset)

        # Get the hyperparameter grid for this model (no auto-injection)
        hyper_grid = self.config["model"][model_name] or {}

        print(f"[INFO] Hyperparameter grid for {model_name}: {hyper_grid}")

        if issubclass(model_class, BaseModel):
            print(f"{model_name} is a Base Model!")
            # Handle case where model has no parameters (empty model)
            if not hyper_grid:
                print(f"[INFO] {model_name} has no parameters, using empty hyper_grid")

            print(f"{model_name} hyper grid: {hyper_grid}")

            model_params = {
                k: v[0] if isinstance(v, list) else v for k, v in hyper_grid.items()
            }
            # Include dataset configuration for other parameters
            model_params["dataset"] = self.config["dataset"]
            print(f"{model_name} initial model_params: {model_params}")

            # Create a full config that includes evaluation metrics
            full_config = self.config.copy()
            # Update the model section with the current model parameters
            if "model" not in full_config:
                full_config["model"] = {}
            full_config["model"][model_name] = model_params

            base_model = model_class(full_config)

            model_hyperparameter_tuner = HyperparameterTuner(
                base_model, hyper_grid, False
            )
            validation_score_hyperparameter_tuple = model_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(
                all_dataset_chunks
            )
            print(
                f"{model_name} validation score hyperparameter tuple: {validation_score_hyperparameter_tuple}"
            )
            best_hyperparameters_dict = {
                k: validation_score_hyperparameter_tuple[1][i]
                for i, k in enumerate(hyper_grid.keys())
            }
            print(
                f"{model_name} best hyperparameters dict: {best_hyperparameters_dict}"
            )

            # Log hyperparameter search results to TensorBoard like we had before
            if self.writer:
                try:
                    # Log the best validation score
                    best_score = validation_score_hyperparameter_tuple[0]
                    self.writer.add_scalar(
                        "hyperparameter_search/best_validation_score", best_score, 0
                    )

                    # Log each hyperparameter value
                    for param_name, param_value in best_hyperparameters_dict.items():
                        if isinstance(param_value, (int, float)):
                            self.writer.add_scalar(
                                f"hyperparameters/{param_name}", param_value, 0
                            )
                        else:
                            self.writer.add_text(
                                f"hyperparameters/{param_name}", str(param_value), 0
                            )

                    print(f"[INFO] Logged hyperparameter search results to TensorBoard")
                except Exception as e:
                    print(
                        f"[WARNING] Failed to log hyperparameter results to TensorBoard: {e}"
                    )

            results = model_hyperparameter_tuner.final_evaluation(
                best_hyperparameters_dict, all_dataset_chunks
            )

            print(f"{model_name} results: {results}")
            print(f"[SUCCESS] {model_name} execution completed successfully!")

            # Log final evaluation results to TensorBoard like we had before
            if self.writer:
                try:
                    # Log each metric
                    for metric_name, metric_value in results.items():
                        if isinstance(metric_value, (int, float)):
                            self.writer.add_scalar(
                                f"evaluation/{metric_name}", metric_value, 0
                            )
                        else:
                            self.writer.add_text(
                                f"evaluation/{metric_name}", str(metric_value), 0
                            )

                    # Log model configuration
                    self.writer.add_text("model/config", str(self.config), 0)
                    self.writer.add_text("model/name", model_name, 0)

                    print(f"[INFO] Logged final evaluation results to TensorBoard")
                except Exception as e:
                    print(
                        f"[WARNING] Failed to log evaluation results to TensorBoard: {e}"
                    )

            # Persist results for host process to log to TensorBoard
            if self.result_path:
                try:
                    payload = {
                        "model": model_name,
                        "best_hyperparameters": best_hyperparameters_dict,
                        "metrics": results,
                    }
                    # Try to create forecast plots for validation and test from last chunk
                    try:
                        last_chunk = all_dataset_chunks[-1]
                        y_context = last_chunk.train.targets
                        y_val = last_chunk.validation.targets
                        y_test = last_chunk.test.targets
                        trained_model = model_hyperparameter_tuner.model_class
                        # Get frequency from the dataset
                        freq = last_chunk.metadata["freq"]

                        # Get timestamps for proper feature creation
                        y_context_timestamps = last_chunk.train.timestamps
                        y_val_timestamps = last_chunk.validation.timestamps
                        y_test_timestamps = last_chunk.test.timestamps

                        # Check if this is a RandomForest model that should use rolling_predict
                        is_random_forest = (
                            "RandomForest" in trained_model.__class__.__name__
                        )

                        # Predict on validation
                        if is_random_forest:
                            # For RandomForest, use rolling_predict to get full-length predictions
                            preds_val = trained_model.rolling_predict(
                                y_context=y_context[-trained_model.lookback_window :],
                                y_target=y_val,
                                y_context_timestamps=y_context_timestamps[
                                    -trained_model.lookback_window :
                                ],
                                y_target_timestamps=y_val_timestamps,
                                freq=freq,
                            )
                        else:
                            # For other models, use regular predict
                            preds_val = trained_model.predict(
                                y_context=y_context,
                                y_target=y_val,
                                y_context_timestamps=y_context_timestamps,
                                y_target_timestamps=y_val_timestamps,
                                freq=freq,
                            )

                        # Predict on test using train+val as context when applicable
                        import numpy as np

                        y_ctx_plus_val = (
                            np.concatenate([y_context, y_val])
                            if y_context is not None and y_val is not None
                            else y_context
                        )
                        y_ctx_plus_val_timestamps = (
                            np.concatenate([y_context_timestamps, y_val_timestamps])
                            if y_context_timestamps is not None
                            and y_val_timestamps is not None
                            else y_context_timestamps
                        )

                        if is_random_forest:
                            # For RandomForest, use rolling_predict to get full-length predictions
                            preds_test = trained_model.rolling_predict(
                                y_context=y_ctx_plus_val[
                                    -trained_model.lookback_window :
                                ],
                                y_target=y_test,
                                y_context_timestamps=y_ctx_plus_val_timestamps[
                                    -trained_model.lookback_window :
                                ],
                                y_target_timestamps=y_test_timestamps,
                                freq=freq,
                            )
                        else:
                            # For other models, use regular predict
                            preds_test = trained_model.predict(
                                y_context=y_ctx_plus_val,
                                y_target=y_test,
                                y_context_timestamps=y_ctx_plus_val_timestamps,
                                y_target_timestamps=y_test_timestamps,
                                freq=freq,
                            )
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        colors = [
                            "#1f77b4",
                            "#ff7f0e",
                            "#2ca02c",
                            "#d62728",
                            "#9467bd",
                            "#8c564b",
                            "#e377c2",
                            "#7f7f7f",
                            "#bcbd22",
                            "#17becf",
                        ]

                        def _save_plot(y_true_arr, preds_arr, title_suffix):
                            y_true_arr = np.asarray(y_true_arr)
                            preds_arr = np.asarray(preds_arr)

                            # Fix: Ensure predictions have the right shape for plotting
                            if preds_arr.ndim == 2 and preds_arr.shape[0] == 1:
                                # ARIMA returns (1, 300), convert to (300,)
                                preds_arr = preds_arr.flatten()

                            fig, ax = plt.subplots(figsize=(12, 6))
                            if y_true_arr.ndim == 1:
                                min_len = min(len(y_true_arr), len(preds_arr))
                                ax.plot(
                                    y_true_arr[:min_len],
                                    color=colors[0],
                                    label="True",
                                    linewidth=2,
                                )
                                ax.plot(
                                    preds_arr[:min_len],
                                    color=colors[1],
                                    label="Predicted",
                                    linewidth=2,
                                    linestyle="--",
                                    alpha=0.9,
                                )
                            else:
                                num_targets = y_true_arr.shape[1]
                                for i in range(num_targets):
                                    c_true = colors[i % len(colors)]
                                    ax.plot(
                                        y_true_arr[:, i],
                                        color=c_true,
                                        label=f"True Target {i}",
                                        linewidth=2,
                                        alpha=0.8,
                                    )
                                    if preds_arr.ndim == 1 and i == 0:
                                        ax.plot(
                                            preds_arr,
                                            color=colors[(i + 1) % len(colors)],
                                            label=f"Predicted Target {i}",
                                            linewidth=2,
                                            linestyle="--",
                                            alpha=0.9,
                                        )
                                    elif preds_arr.ndim == 2 and i < preds_arr.shape[1]:
                                        pred_vals = preds_arr[:, i]
                                        if not np.all(np.isnan(pred_vals)):
                                            ax.plot(
                                                pred_vals,
                                                color=colors[(i + 1) % len(colors)],
                                                label=f"Predicted Target {i}",
                                                linewidth=2,
                                                linestyle="--",
                                                alpha=0.9,
                                            )
                            ax.set_title(
                                f"{model_name} Predictions vs True Values ({title_suffix})",
                                fontsize=14,
                                fontweight="bold",
                            )
                            ax.set_xlabel("Time Steps", fontsize=12)
                            ax.set_ylabel("Values", fontsize=12)
                            ax.grid(True, alpha=0.3)
                            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                            plt.tight_layout()
                            import tempfile

                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".png"
                            ) as tmp_png:
                                p = tmp_png.name
                            fig.savefig(p)
                            plt.close(fig)
                            return p

                        plot_val = _save_plot(
                            y_true_arr=y_val,
                            preds_arr=preds_val,
                            title_suffix="Validation",
                        )
                        plot_test = _save_plot(
                            y_true_arr=y_test, preds_arr=preds_test, title_suffix="Test"
                        )
                        payload["forecast_plot_val_path"] = plot_val
                        payload["forecast_plot_test_path"] = plot_test
                    except Exception as e:
                        print(f"[WARNING] Failed to create forecast plots: {e}")

                    with open(self.result_path, "w") as rf:
                        json.dump(payload, rf)
                except Exception as write_err:
                    print(
                        f"[WARNING] Failed to write results to {self.result_path}: {write_err}"
                    )
        elif issubclass(model_class, FoundationModel):
            print(f"{model_name} is a Foundation Model!")

            # Handle case where model has no parameters (empty model)
            if not hyper_grid:
                print(f"[INFO] {model_name} has no parameters, using empty hyper_grid")

            model_params = {
                k: v[0] if isinstance(v, list) else v for k, v in hyper_grid.items()
            }
            # Include dataset configuration for other parameters
            model_params["dataset"] = self.config["dataset"]

            # Create a full config that includes evaluation metrics (like we do for base models)
            full_config = self.config.copy()
            # Update the model section with the current model parameters
            if "model" not in full_config:
                full_config["model"] = {}
            full_config["model"][model_name] = model_params

            foundation_model = model_class(full_config)

            model_hyperparameter_tuner = FoundationModelTuner(
                foundation_model, hyper_grid, False
            )
            validation_score_hyperparameter_tuple = model_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(
                all_dataset_chunks
            )
            best_hyperparameters_dict = {
                k: validation_score_hyperparameter_tuple[1][i]
                for i, k in enumerate(hyper_grid.keys())
            }
            print(
                f"{model_name} best hyperparameters dict: {best_hyperparameters_dict}"
            )

            # Log hyperparameter search results to TensorBoard like we had before
            if self.writer:
                try:
                    # Log the best validation score
                    best_score = validation_score_hyperparameter_tuple[0]
                    self.writer.add_scalar(
                        "hyperparameter_search/best_validation_score", best_score, 0
                    )

                    # Log each hyperparameter value
                    for param_name, param_value in best_hyperparameters_dict.items():
                        if isinstance(param_value, (int, float)):
                            self.writer.add_scalar(
                                f"hyperparameters/{param_name}", param_value, 0
                            )
                        else:
                            self.writer.add_text(
                                f"hyperparameters/{param_name}", str(param_value), 0
                            )

                    print(
                        f"[INFO] Logged foundation model hyperparameter search results to TensorBoard"
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Failed to log foundation model hyperparameter results to TensorBoard: {e}"
                    )

            results = model_hyperparameter_tuner.final_evaluation(
                best_hyperparameters_dict, all_dataset_chunks
            )

            print(f"{model_name} results: {results}")
            print(f"[SUCCESS] {model_name} execution completed successfully!")

            # Log final evaluation results to TensorBoard like we had before
            if self.writer:
                try:
                    # Log each metric
                    for metric_name, metric_value in results.items():
                        if isinstance(metric_value, (int, float)):
                            self.writer.add_scalar(
                                f"evaluation/{metric_name}", metric_value, 0
                            )
                        else:
                            self.writer.add_text(
                                f"evaluation/{metric_name}", str(metric_value), 0
                            )

                    # Log model configuration
                    self.writer.add_text("model/config", str(self.config), 0)
                    self.writer.add_text("model/name", model_name, 0)

                    print(
                        f"[INFO] Logged foundation model final evaluation results to TensorBoard"
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Failed to log foundation model evaluation results to TensorBoard: {e}"
                    )

            # Persist results for host process to log to TensorBoard
            if self.result_path:
                try:
                    payload = {
                        "model": model_name,
                        "best_hyperparameters": best_hyperparameters_dict,
                        "metrics": results,
                    }
                    # Create plots using last chunk (validation and test)
                    try:
                        last_chunk = all_dataset_chunks[-1]
                        y_context = last_chunk.train.targets
                        y_val = last_chunk.validation.targets
                        y_test = last_chunk.test.targets
                        trained_model = model_hyperparameter_tuner.model_class
                        import numpy as np

                        # Get frequency from the dataset
                        freq = last_chunk.metadata["freq"]
                        preds_val = trained_model.predict(
                            y_context=y_context,
                            y_target=y_val,
                            y_context_timestamps=last_chunk.train.timestamps,
                            y_target_timestamps=last_chunk.validation.timestamps,
                            freq=freq,
                        )
                        y_ctx_plus_val = (
                            np.concatenate([y_context, y_val])
                            if y_context is not None and y_val is not None
                            else y_context
                        )
                        preds_test = trained_model.predict(
                            y_context=y_ctx_plus_val,
                            y_target=y_test,
                            y_context_timestamps=np.concatenate(
                                [
                                    last_chunk.train.timestamps,
                                    last_chunk.validation.timestamps,
                                ]
                            ),
                            y_target_timestamps=last_chunk.test.timestamps,
                            freq=freq,
                        )
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        colors = [
                            "#1f77b4",
                            "#ff7f0e",
                            "#2ca02c",
                            "#d62728",
                            "#9467bd",
                            "#8c564b",
                            "#e377c2",
                            "#7f7f7f",
                            "#bcbd22",
                            "#17becf",
                        ]

                        def _save_plot(y_true_arr, preds_arr, title_suffix):
                            y_true_arr = np.asarray(y_true_arr)
                            preds_arr = np.asarray(preds_arr)
                            fig, ax = plt.subplots(figsize=(12, 6))
                            if y_true_arr.ndim == 1:
                                min_len = min(len(y_true_arr), len(preds_arr))
                                ax.plot(
                                    y_true_arr[:min_len],
                                    color=colors[0],
                                    label="True",
                                    linewidth=2,
                                )
                                ax.plot(
                                    preds_arr[:min_len],
                                    color=colors[1],
                                    label="Predicted",
                                    linewidth=2,
                                    linestyle="--",
                                    alpha=0.9,
                                )
                            else:
                                num_targets = y_true_arr.shape[1]
                                for i in range(num_targets):
                                    c_true = colors[i % len(colors)]
                                    ax.plot(
                                        y_true_arr[:, i],
                                        color=c_true,
                                        label=f"True Target {i}",
                                        linewidth=2,
                                        alpha=0.8,
                                    )
                                    if preds_arr.ndim == 1 and i == 0:
                                        ax.plot(
                                            preds_arr,
                                            color=colors[(i + 1) % len(colors)],
                                            label=f"Predicted Target {i}",
                                            linewidth=2,
                                            linestyle="--",
                                            alpha=0.9,
                                        )
                                    elif preds_arr.ndim == 2 and i < preds_arr.shape[1]:
                                        pred_vals = preds_arr[:, i]
                                        if not np.all(np.isnan(pred_vals)):
                                            ax.plot(
                                                pred_vals,
                                                color=colors[(i + 1) % len(colors)],
                                                label=f"Predicted Target {i}",
                                                linewidth=2,
                                                linestyle="--",
                                                alpha=0.9,
                                            )
                            ax.set_title(
                                f"{model_name} Predictions vs True Values ({title_suffix})",
                                fontsize=14,
                                fontweight="bold",
                            )
                            ax.set_xlabel("Time Steps", fontsize=12)
                            ax.set_ylabel("Values", fontsize=12)
                            ax.grid(True, alpha=0.3)
                            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                            plt.tight_layout()
                            import tempfile

                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".png"
                            ) as tmp_png:
                                p = tmp_png.name
                            fig.savefig(p)
                            plt.close(fig)
                            return p

                        plot_val = _save_plot(
                            y_true_arr=y_val,
                            preds_arr=preds_val,
                            title_suffix="Validation",
                        )
                        plot_test = _save_plot(
                            y_true_arr=y_test, preds_arr=preds_test, title_suffix="Test"
                        )
                        payload["forecast_plot_val_path"] = plot_val
                        payload["forecast_plot_test_path"] = plot_test
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to create forecast plots (foundation): {e}"
                        )

                    with open(self.result_path, "w") as rf:
                        json.dump(payload, rf)
                except Exception as write_err:
                    print(
                        f"[WARNING] Failed to write results to {self.result_path}: {write_err}"
                    )

        # Cleanup TensorBoard writer to ensure all logs are flushed
        self.cleanup()

    def cleanup(self):
        """Cleanup TensorBoard writer and ensure all logs are flushed."""
        if self.writer:
            try:
                self.writer.close()
                print(
                    f"[INFO] TensorBoard writer closed, logs saved to: {self.log_dir}"
                )
            except Exception as e:
                print(f"[WARNING] Failed to close TensorBoard writer: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute a single model in an isolated environment for benchmarking."
    )
    parser.add_argument("--config", type=str, help="Path to the config YAML file")
    parser.add_argument(
        "--chunk_path",
        type=str,
        help="Path to the temporary pickle file containing dataset chunks.",
    )
    parser.add_argument(
        "--model_folder_name", type=str, help="Name of the model folder to reference."
    )
    parser.add_argument(
        "--model_file_name",
        type=str,
        help="Name of the model file containing the model class.",
    )
    parser.add_argument(
        "--model_class_name", type=str, help="Name of the model class to instantiate."
    )
    parser.add_argument(
        "--result_path", type=str, help="Path to write JSON results for host logging."
    )
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_executor = ModelExecutor(
        config=config,
        chunk_path=args.chunk_path,
        model_folder_name=args.model_folder_name,
        model_file_name=args.model_file_name,
        model_class_name=args.model_class_name,
        result_path=args.result_path,
    )
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Chunk Path: {args.chunk_path}")
    print(f"[INFO] Model Folder: {args.model_folder_name}")
    print(f"[INFO] Model File: {args.model_file_name}")
    print(f"[INFO] Model Class: {args.model_class_name}")
    model_executor.run()
