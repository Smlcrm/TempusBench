#!/usr/bin/env python3
"""
Generate benchmark config files: one per model for multivariate and univariate tasks.
Each model will have two config files:
  - benchmark_gcp_{model}_multivariate.yaml (runs multivariate/*)
  - benchmark_gcp_{model}_univariate.yaml (runs univariate/*)
"""

import yaml
from pathlib import Path

# Model configurations from the original config
MODEL_CONFIGS = {
    "seasonal_naive": {
        "sp": [12, 24]
    },
    "arima": {
        "p": [1, 2],
        "d": [1],
        "q": [1, 2],
        "s": [2]
    },
    "croston_classic": {
        "alpha": [0.1, 0.3, 0.5],
        "gamma": [0.1, 0.3, 0.5]
    },
    "exponential_smoothing": {
        "trend": ["null", "add"],
        "seasonal": ["null"],
        "seasonal_periods": [4, 8, 16, 32]
    },
    "lstm": {
        "learning_rate": [0.01, 0.001, 0.0001]
    },
    "prophet": {
        "seasonality_mode": ["additive"],
        "changepoint_prior_scale": [0.05],
        "seasonality_prior_scale": [10.0]
    },
    "random_forest": {
        "n_estimators": [10],
        "max_depth": [2]
    },
    "svr": {
        "kernel": ["rbf"],
        "C": [1.0],
        "epsilon": [0.1],
        "gamma": ["scale"]
    },
    "theta": {
        "sp": [12],
        "theta_method": ["correlation_optimal"],
        "use_reduced_rank": [False]
    },
    "varmax": {
        "p": [1, 2],
        "q": [1, 2],
        "trend": ["c", "t"]
    },
    "xgboost": {
        "n_estimators": [200],
        "max_depth": [4],
        "learning_rate": [0.05, 0.01, 0.001]
    },
    "moment": {},
    "timesfm": {},
    "tiny_time_mixer": {},
    "chronos": {},
    "moirai": {},
    "moirai_moe": {},
    "lagllama": {},
    "toto": {},
    "tabpfn": {}
}

# Base evaluation configuration
EVALUATION_CONFIG = {
    "tuning_loss": "mae",
    "max_windows": 4,
    "max_num_variates": 10,
    "num_samples": 100,
    "num_quantiles": 10,
    "point_forecast_statistic": "mean"
}

def generate_config(model_name: str, task_type: str) -> dict:
    """Generate a config file for a specific model and task type."""
    config = {
        "evaluation": {
            "task_path": f"{task_type}/*",
            **EVALUATION_CONFIG
        },
        "model": {
            model_name: MODEL_CONFIGS[model_name]
        }
    }
    return config

def represent_list_flow_style(dumper, data):
    """Represent lists in flow style [a, b, c] instead of block style."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Register custom representer for lists
yaml.add_representer(list, represent_list_flow_style)

def main():
    config_dir = Path("tempus_bench/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete old benchmark_gcp_*.yaml files (keep benchmark_test_gcp.yaml)
    print("Cleaning up old benchmark_gcp_*.yaml files...")
    for old_file in config_dir.glob("benchmark_gcp_*.yaml"):
        if old_file.name != "benchmark_test_gcp.yaml":
            old_file.unlink()
            print(f"  Deleted: {old_file.name}")
    
    print("\nGenerating new config files...")
    
    # Generate configs for each model
    for model_name in MODEL_CONFIGS.keys():
        # Multivariate config
        multivariate_config = generate_config(model_name, "multivariate")
        multivariate_file = config_dir / f"benchmark_gcp_{model_name}_multivariate.yaml"
        
        with open(multivariate_file, 'w') as f:
            yaml.dump(multivariate_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  Created: {multivariate_file.name}")
        
        # Univariate config
        univariate_config = generate_config(model_name, "univariate")
        univariate_file = config_dir / f"benchmark_gcp_{model_name}_univariate.yaml"
        
        with open(univariate_file, 'w') as f:
            yaml.dump(univariate_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  Created: {univariate_file.name}")
    
    total_files = len(MODEL_CONFIGS) * 2
    print(f"\n✅ Generated {total_files} config files ({len(MODEL_CONFIGS)} models × 2 task types)")
    print(f"   - Multivariate configs: {len(MODEL_CONFIGS)}")
    print(f"   - Univariate configs: {len(MODEL_CONFIGS)}")

if __name__ == "__main__":
    main()

