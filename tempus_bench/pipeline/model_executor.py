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
import sys
import importlib.util
import tempfile
import pickle
import re
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional


from tempus_bench.utils.envs import CondaEnvManager
from tempus_bench.utils.paths import get_project_root, get_models_dir
from tempus_bench.utils.model_settings import (
    get_past_only_covariate_models,
    get_no_covariate_models,
)

# Derived from each model's settings.yaml ``capabilities.covariates``.
PAST_ONLY_COVARIATE_MODELS = get_past_only_covariate_models()
NO_COVARIATE_MODELS = get_no_covariate_models()

# Model names whose class names don't follow default PascalCase derivation (e.g. nbeats -> NBEATSModel)
MODEL_CLASS_NAME_OVERRIDES = {
    "nbeats": "NBEATSModel",
    "nhits": "NHITSModel",
    "itransformer": "ITransformerModel",
    "timesnet": "TimesNetModel",
    "tft": "TFTModel",
    "deepar": "DeepARModel",
}
from tempus_bench.utils.utils import validate_forecast_sanity
from tempus_bench.pipeline.data_types import Dataset

# Printed on its own line before ``json.dumps(outputs)`` so the parent can find the payload
# even when libraries emit other ``[...]``-shaped lines earlier on stdout.
_JSON_OUTPUT_SENTINEL = "__TEMPUSBENCH_MODEL_EXECUTOR_OUTPUT_JSON__"
_LEGACY_JSON_OUTPUT_SENTINELS = ("__TEMPUSBENCH_MODEL_EXECUTOR_RESULTS__",)
_JSON_OUTPUT_SENTINELS = (_JSON_OUTPUT_SENTINEL,) + _LEGACY_JSON_OUTPUT_SENTINELS


def parse_subprocess_eval_stdout(stdout: str) -> list:
    """Parse ``model_executor`` subprocess stdout into the per-window evaluation list.

    Prefer the sentinel line (current protocol). Otherwise scan **from the end**: the
    real ``json.dumps(outputs)`` line is almost always last, while stray ``[...]`` lines
    from dependencies often appear earlier and used to be mistaken for the payload.
    """
    raw = (stdout or "").strip()
    if not raw:
        raise ValueError("Subprocess stdout is empty")
    lines = raw.split("\n")
    dec = json.JSONDecoder()

    for sentinel in _JSON_OUTPUT_SENTINELS:
        for i, line in enumerate(lines):
            if line.strip() != sentinel:
                continue
            remainder = "\n".join(lines[i + 1 :]).strip()
            if not remainder:
                raise ValueError(
                    f"After {sentinel!r}, expected JSON payload. Stdout tail: {raw[-500:]!r}"
                )
            try:
                out, end = dec.raw_decode(remainder)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"After {sentinel!r}, JSON decode failed: {e}. "
                    f"Payload prefix: {remainder[:240]!r}"
                ) from e
            trailing = remainder[end:].strip()
            if trailing:
                raise ValueError(
                    f"After {sentinel!r}, trailing garbage after JSON: {trailing[:120]!r}"
                )
            if not isinstance(out, list):
                raise ValueError(
                    f"After {sentinel!r}, expected a JSON array, got {type(out).__name__}"
                )
            return out

    for line in reversed(lines):
        s = line.strip()
        if not s or s == "[]":
            continue
        try:
            out, end = dec.raw_decode(s)
        except json.JSONDecodeError:
            continue
        if end < len(s):
            continue
        if isinstance(out, list) and len(out) > 0 and all(
            isinstance(x, dict) for x in out
        ):
            return out

    raise ValueError(
        "No evaluation results found in subprocess stdout (no sentinel and no parseable "
        f"non-empty JSON array of objects). Stdout tail: {raw[-500:]!r}"
    )


def _scaler_from_dataset_metadata(metadata: Optional[Dict[str, Any]]):
    """Reconstruct StandardScaler from DataLoader metadata (pickle-safe)."""
    if not metadata or "target_scaler_mean" not in metadata:
        return None
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required to inverse-transform normalized targets for export"
        ) from e
    s = StandardScaler()
    s.mean_ = np.asarray(metadata["target_scaler_mean"], dtype=np.float64)
    s.scale_ = np.asarray(metadata["target_scaler_scale"], dtype=np.float64)
    s.var_ = s.scale_ ** 2
    s.n_features_in_ = int(s.mean_.shape[0])
    s.feature_names_in_ = None
    return s


def _inverse_target_scale(y: np.ndarray, scaler) -> np.ndarray:
    """Inverse StandardScaler per column (same as training targets)."""
    if scaler is None:
        return np.asarray(y, dtype=np.float64)
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        return scaler.inverse_transform(a.reshape(-1, 1)).ravel()
    if a.ndim == 2:
        return scaler.inverse_transform(a)
    if a.ndim == 3:
        s0, s1, s2 = a.shape
        flat = a.reshape(-1, s2)
        inv = scaler.inverse_transform(flat)
        return inv.reshape(s0, s1, s2)
    return a


class ModelExecutor:
    """
    Executes a single model inside its dedicated Conda environment.

    The executor prepares the command-line invocation, ensures the requested environment
    is available, and parses JSON results emitted by the child process. This isolation
    allows different models to have conflicting dependencies without interference.

    Attributes:
        job_config (JobConfig): Complete job configuration including task and model settings.
    """

    def __init__(
        self,
        job_config: Dict,
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
        model_name = self.job_config["model_config"]["model_name"]

        # Get model requirements path
        requirements_path = self._get_model_requirements(model_name=model_name)

        # Create temporary directory for augmented requirements and job config
        # Note: The temporary directory and any augmented requirements file
        # created within it are automatically deleted when this 'with' block exits
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure all packages required by main() function are included
            final_requirements_path = self._ensure_required_packages(
                requirements_path=requirements_path, temp_dir=temp_dir
            )

            # Create Conda Environment
            # Note: CondaEnvManager reads the requirements file synchronously during
            # initialization if a new environment is created or reinstall is requested.
            # The temporary file (if created) will exist until this 'with' block exits.
            python_version = self.job_config["model_setting"]["python_version"]
            conda_env = CondaEnvManager(
                name=f"benchmark.{model_name}",
                python=python_version,
                requirements_path=final_requirements_path,
                reinstall=self.job_config["evaluation_setting"]["reinstall_conda"],
            )
            job_config_path = os.path.join(temp_dir, "job_config.json")

            # Add absolute path to task dataset so subprocess finds it regardless of cwd
            task_name = self.job_config["task_config"]["task_name"]
            tdir = self.job_config.get("task_datasets_dir")
            if tdir:
                task_dataset_path = str(Path(tdir) / f"{task_name}.pkl")
            else:
                task_dataset_path = str(
                    Path(get_project_root()) / "temp_task_datasets" / f"{task_name}.pkl"
                )
            job_config_to_write = {
                **self.job_config,
                "task_dataset_path": task_dataset_path,
            }

            # Write job config dict to JSON file
            with open(job_config_path, "w") as f:
                json.dump(job_config_to_write, f)

            # Build CLI command
            hyperparameters_json = json.dumps(hyperparameters)

            command = (
                f"python -m tempus_bench.pipeline.model_executor "
                f"--task-name {self.job_config['task_config']['task_name']} "
                f"--model-name {model_name} "
                f"--hyperparameters '{hyperparameters_json}' "
                f"--context-steps {context_steps} "
                f"--train-steps {train_steps} "
                f"--validate-steps {validate_steps} "
                f"--job-config-path {job_config_path} "
            )

            verbose = self.job_config.get("evaluation_setting", {}).get("verbose", False)
            result = None
            for attempt in range(2):
                try:
                    result = conda_env.run(command=command, verbose=verbose)
                    break
                except RuntimeError as e:
                    err_text = str(e)
                    if attempt == 0 and (
                        "ModuleNotFoundError" in err_text or "ImportError" in err_text
                    ):
                        conda_env.install(final_requirements_path)
                        continue
                    raise
            if result is None:
                raise RuntimeError("Model execution failed after install retry")
            outputs = parse_subprocess_eval_stdout(result.stdout)
        return outputs

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

    def _ensure_required_packages(self, requirements_path: str, temp_dir: str) -> str:
        """
        Ensure that all packages required by main() function are in the requirements file.

        This method checks if packages used in the main() function (lines 183-390) are
        present in the model's requirements.txt. If any are missing, it creates a
        temporary requirements file that includes both the original requirements and
        the missing packages.

        Packages checked:
        - numpy: Used in main() for array operations

        Args:
            requirements_path (str): Path to the original model requirements.txt file.
            temp_dir (str): Temporary directory where augmented requirements file
                can be created if needed.

        Returns:
            str: Path to requirements file to use (original if no additions needed,
                temporary augmented file if packages were missing).
        """
        # Packages required by main() function (lines 183-390)
        # Only include third-party packages (standard library packages don't need installation)
        required_packages = {
            "numpy": "numpy",  # numpy is used as np in main()
        }

        # Read the original requirements file
        with open(requirements_path, "r") as f:
            original_requirements = f.read()

        # Check which required packages are missing
        missing_packages = []
        for package_name, install_name in required_packages.items():
            # Check if package is in requirements (case-insensitive, allow version specs)
            # Match package name at start of line or after a newline
            pattern = rf"(^|\n)\s*{package_name}(\s|==|>=|<=|>|<|$)"

            if not re.search(
                pattern, original_requirements, re.IGNORECASE | re.MULTILINE
            ):
                missing_packages.append(install_name)

        # If no packages are missing, return original path
        if not missing_packages:
            return requirements_path

        # Create augmented requirements file
        augmented_path = os.path.join(temp_dir, "augmented_requirements.txt")
        with open(augmented_path, "w") as f:
            # Write original requirements
            f.write(original_requirements)
            # Add missing packages if file doesn't end with newline
            if original_requirements and not original_requirements.endswith("\n"):
                f.write("\n")
            # Append missing packages
            for package in missing_packages:
                f.write(f"{package}\n")

        return augmented_path


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
        --job-config-path: Path to JSON job configuration file.

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

    print("In Model Executor")

    args = parser.parse_args()

    # Parse hyperparameters JSON
    hyperparameters = json.loads(args.hyperparameters)

    # Load configuration as dictionary
    with open(args.job_config_path, "r") as f:
        job_config = json.load(f)

    task_name = args.task_name
    temp_task_dataset_path = job_config.get("task_dataset_path")
    if not temp_task_dataset_path:
        tdir = job_config.get("task_datasets_dir")
        if tdir:
            temp_task_dataset_path = str(Path(tdir) / f"{task_name}.pkl")
        else:
            temp_task_dataset_path = str(
                Path(get_project_root()) / "temp_task_datasets" / f"{task_name}.pkl"
            )

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
        steps=steps,
        stride=args.validate_steps,
        max_windows=job_config["evaluation_config"]["max_windows"],
    )

    outputs = []

    # Import model - models are now directly in the models directory
    models_dir = get_models_dir()
    model_dir = models_dir / args.model_name
    model_file = f"{model_name}_model"
    module_path = str(model_dir / f"{model_file}.py")

    # Generate class name (PascalCase + Model suffix); use override for special cases
    class_name = MODEL_CLASS_NAME_OVERRIDES.get(
        args.model_name,
        "".join(word.capitalize() for word in args.model_name.split("_")) + "Model",
    )

    spec = importlib.util.spec_from_file_location(model_file, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Failed to load module spec for {model_file} from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    model_class = getattr(module, class_name)

    for window_idx, dataset_splits in enumerate(window_generator):
        timestamps = np.asarray(dataset.timestamps)
        target = np.asarray(dataset.target)
        freq = dataset.metadata["time_freq"]  # type: ignore
        if freq:
            freq = str(freq)
            # Models that need pandas 2.0 freq (M->ME, Q->QE, A->YE): neuralforecast
            NEEDS_PANDAS2_FREQ = frozenset({"deepar", "itransformer", "timesnet"})
            # Models that need legacy M (gluonts Period, toto freq_to_seconds): don't map
            NEEDS_LEGACY_FREQ = frozenset({"lagllama", "toto"})
            if model_name in NEEDS_PANDAS2_FREQ:
                freq = re.sub(r"(\d*)M$", r"\1ME", freq)
                freq = re.sub(r"(\d*)Q$", r"\1QE", freq)
                freq = re.sub(r"(\d*)A$", r"\1YE", freq)
                freq = re.sub(r"(\d*)Y$", r"\1YE", freq)
            elif model_name in NEEDS_LEGACY_FREQ:
                # Keep M, Q, A; only map Q->QE, A->YE for lagllama (gluonts) compatibility
                if model_name == "lagllama":
                    freq = {"Q": "QE", "A": "YE", "Y": "YE", "AS": "YS"}.get(
                        freq, freq
                    )
            else:
                # Default: apply pandas 2.0 mapping for compatibility
                freq = re.sub(r"(\d*)M$", r"\1ME", freq)
                freq = re.sub(r"(\d*)Q$", r"\1QE", freq)
                freq = re.sub(r"(\d*)A$", r"\1YE", freq)
                freq = re.sub(r"(\d*)Y$", r"\1YE", freq)

        # Extract split indices
        cstart, cend = dataset_splits["context"].start, dataset_splits["context"].end
        tstart, tend = dataset_splits["train"].start, dataset_splits["train"].end
        vstart, vend = dataset_splits["validate"].start, dataset_splits["validate"].end

        # Create and train model
        model = model_class(
            params=hyperparameters, settings=job_config["model_setting"]
        )

        x_context_train = None
        x_target_train = None
        x_context_predict = None
        x_target_predict = None

        if dataset.covariate is not None and model_name not in NO_COVARIATE_MODELS:
            covariate = np.asarray(dataset.covariate)
            x_context_train = covariate[cstart:cend]
            x_target_train = covariate[tstart:tend] if model_name not in PAST_ONLY_COVARIATE_MODELS else None

            # For predict: past covs = context+train period, future covs = validation period
            # Chronos-2 expects past_covariates (history_length,) and future_covariates (prediction_length,)
            x_context_predict = covariate[cstart:tend]
            x_target_predict = covariate[vstart:vend] if model_name not in PAST_ONLY_COVARIATE_MODELS else None

        train_call_kwargs: dict = {
            "x_context": x_context_train,
            "x_target": x_target_train,
            "y_context": target[cstart:cend],
            "y_target": target[tstart:tend],
            "timestamps_context": timestamps[cstart:cend],
            "timestamps_target": timestamps[tstart:tend],
            "freq": freq,
            "tuning_loss": job_config["evaluation_config"]["tuning_loss"],
            "num_samples": job_config["evaluation_config"]["num_samples"],
        }
        # Moirai models must size prediction_length for the validation horizon; train span alone
        # can disagree with uni2ts patching and yield y_pred shorter than y_true (no BQ rows).
        if model_name in ("moirai_moe", "moirai2"):
            train_call_kwargs["validate_horizon"] = int(vend - vstart)

        trained_model = model.train(**train_call_kwargs)

        # Generate predictions
        results = trained_model.predict(
            x_context=x_context_predict,
            x_target=x_target_predict,
            y_context=target[cstart:tend],
            timestamps_context=timestamps[cstart:tend],
            timestamps_target=timestamps[vstart:vend],
            freq=freq,
            num_samples=job_config["evaluation_config"]["num_samples"],
        )

        # Covariate sensitivity check (first window only, when covariates present)
        if (
            dataset.covariate is not None
            and window_idx == 0
            and hasattr(trained_model, "predict")
        ):
            try:
                pred_no_cov = trained_model.predict(
                    x_context=x_context_predict,
                    x_target=x_target_predict,
                    y_context=target[cstart:tend],
                    timestamps_context=timestamps[cstart:tend],
                    timestamps_target=timestamps[vstart:vend],
                    freq=freq,
                    num_samples=job_config["evaluation_config"]["num_samples"],
                    use_covariates=False,
                )
                max_diff = float(np.abs(np.asarray(results) - np.asarray(pred_no_cov)).max())
                if max_diff < 1e-6:
                    print(
                        "[Covariate check] WARNING: Predictions identical with/without covariates "
                        f"(max_diff={max_diff:.2e}). Covariates may not be used.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[Covariate check] OK: Predictions differ with vs without covariates (max_diff={max_diff:.4f})",
                        file=sys.stderr,
                    )
            except TypeError:
                # Model doesn't support use_covariates kwarg (e.g. non-Chronos2)
                pass

        # Compute evaluation metrics
        eval_metrics = trained_model.compute_metrics(
            y_true=target[vstart:vend],
            y_pred=results,
            point_forecast_statistic=job_config["evaluation_config"][
                "point_forecast_statistic"
            ],
            num_quantiles=job_config["evaluation_config"]["num_quantiles"],
        )

        # Forecast sanity validation
        y_true_window = target[vstart:vend]
        validation = validate_forecast_sanity(
            y_true=y_true_window,
            y_pred=results,
            point_forecast_statistic=job_config["evaluation_config"][
                "point_forecast_statistic"
            ],
        )
        if not validation["ok"]:
            for w in validation["warnings"]:
                print(f"[Forecast validation] window {window_idx}: {w}", file=sys.stderr)

        # Metrics above use normalized y_true / y_pred. Export denormalized series for
        # BigQuery + UI alignment with raw task CSV (inverse StandardScaler when present).
        md = dataset.metadata if isinstance(dataset.metadata, dict) else {}
        tc = job_config.get("task_config") or {}
        ds_cfg = tc.get("dataset") or {}
        normalize = bool(ds_cfg.get("normalize", False))
        scaler = _scaler_from_dataset_metadata(md)
        if normalize and scaler is None:
            raise RuntimeError(
                "task dataset.normalize is true but the pickled Dataset metadata is missing "
                "target_scaler_mean / target_scaler_scale; rebuild task datasets with the "
                "current tempus_bench DataLoader."
            )

        if job_config["model_setting"]["model_type"] == "hybrid":
            pt = _inverse_target_scale(np.asarray(results[0]), scaler)
            results_out = pt.tolist()
        else:
            results_out = _inverse_target_scale(np.asarray(results), scaler).tolist()

        y_true_out = _inverse_target_scale(target[vstart:vend], scaler).tolist()
        ctx_out = _inverse_target_scale(target[cstart:tend], scaler).tolist()

        # Include predictions and context in output (for plotting and downstream export)
        output = {
            **eval_metrics,
            "y_pred": results_out,
            "y_true": y_true_out,
            "timestamps_pred": timestamps[vstart:vend].tolist(),
            "context_values": ctx_out,
            "context_timestamps": timestamps[cstart:tend].tolist(),
            "forecast_validation": {
                "ok": validation["ok"],
                "warnings": validation["warnings"],
                "stats": validation["stats"],
            },
        }

        outputs.append(output)

    # Output results as JSON (metrics may include datetimes / numpy scalars).
    # Sentinel line lets the parent parse stdout even when deps print ``[]``-like noise.
    print(_JSON_OUTPUT_SENTINEL)
    print(json.dumps(outputs, default=str))


if __name__ == "__main__":
    main()
