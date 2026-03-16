#!/usr/bin/env python3
"""
Run covariate benchmarks using the unified config (benchmark_covariates.yaml).
Logs output to logs/covariate_benchmarks.log and exports a summary.

Usage:
  python scripts/run_covariate_benchmarks.py                    # Run all models
  python scripts/run_covariate_benchmarks.py --models a,b,c    # Run specific models only

Logs:
  - logs/covariate_benchmarks.log         Full stdout/stderr
  - logs/covariate_benchmarks_summary.log Summary (timestamps, exit status)
"""
import argparse
import subprocess
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "tempus_bench" / "config"
UNIFIED_CONFIG = CONFIG_DIR / "benchmark_covariates.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"

# Models that need external credentials or have unfixable env issues (always skip)
SKIP_MODELS = {
    "lafn",  # Requires Chronarium remote model, GCP credentials
}


def load_unified_config():
    """Load the unified covariate config."""
    with open(UNIFIED_CONFIG) as f:
        return yaml.safe_load(f)


def get_models_from_config(config_data):
    """Extract model names from config, excluding SKIP_MODELS."""
    return [m for m in config_data["model"].keys() if m not in SKIP_MODELS]


def create_filtered_config(model_subset):
    """Create a temp config with only the specified models."""
    config_data = load_unified_config()
    filtered_models = {m: config_data["model"][m] for m in model_subset if m in config_data["model"]}
    if not filtered_models:
        return None
    config_data["model"] = filtered_models
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="covariate_")
    with open(fd, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    return path


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="?",
        help="Comma-separated list of models to run (default: all)."
    )
    args = parser.parse_args()

    config_path = str(UNIFIED_CONFIG)
    temp_config_path = None

    if args.models:
        model_set = {m.strip() for m in args.models.split(",")}
        config_data = load_unified_config()
        available = set(config_data["model"].keys()) - SKIP_MODELS
        requested = model_set & available
        if not requested:
            print(f"No matching models. Available: {', '.join(sorted(available))}")
            sys.exit(1)
        temp_config_path = create_filtered_config(requested)
        config_path = temp_config_path
        print(f"Running {len(requested)} selected models: {', '.join(sorted(requested))}")
    else:
        models = get_models_from_config(load_unified_config())
        print(f"Running {len(models)} models from {UNIFIED_CONFIG.name}")

    log_path = LOGS_DIR / "covariate_benchmarks.log"
    start_time = datetime.now().isoformat()

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            [sys.executable, "-m", "tempus_bench.run_benchmark", "--config", config_path],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    end_time = datetime.now().isoformat()

    if temp_config_path:
        Path(temp_config_path).unlink(missing_ok=True)

    # Export summary
    summary_path = LOGS_DIR / "covariate_benchmarks_summary.log"
    status = "PASSED" if result.returncode == 0 else "FAILED"
    with open(summary_path, "w") as f:
        f.write(f"Covariate benchmarks run: {start_time} -> {end_time}\n")
        f.write(f"Config: {UNIFIED_CONFIG.name}\n")
        f.write(f"Status: {status} (exit {result.returncode})\n")

    print(f"\nLog: {log_path}")
    print(f"Summary: {summary_path}")
    print(f"Status: {status}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
