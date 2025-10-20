#!/usr/bin/env python3
"""
Generate individual configuration files for all datasets based on analysis results.
"""

import json
import os
import yaml
from pathlib import Path

def represent_list(dumper, data):
    """Represent lists as inline format."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Register the custom representer
yaml.add_representer(list, represent_list)

class NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that disables anchors and aliases."""
    def ignore_aliases(self, data):
        return True

# Add the list representer to the NoAliasDumper
NoAliasDumper.add_representer(list, represent_list)

def generate_hyperparameters(dataset_info):
    """Generate hyperparameters based on dataset characteristics."""
    
    # Base hyperparameters with 2-4 options per parameter for speed/accuracy balance
    # Based on all_models.yaml - includes ALL models even those without hyperparameters
    base_hyperparams = {
        # Statistical Models - exponential_smoothing with safe combinations
        'exponential_smoothing': {
            'trend': ['add'],
            'seasonal': [None],
            'seasonal_periods': [None],
            'damped_trend': [False, True]
        },
        'seasonal_naive': {
            'sp': [12, 24, 168]
        },
        'croston_classic': {
            'alpha': [0.1, 0.3, 0.5],
            'gamma': [0.1, 0.3]
        },
        'theta': {
            'sp': [12, 24]
        },
        'arima': {
            'p': [1, 2],
            'd': [1],
            'q': [1, 2],
            's': [2]
        },
        
        # Machine Learning Models
        'xgboost': {
            'lookback_window': [30, 50],
            'n_estimators': [100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.9],
            'colsample_bytree': [0.9],
            'random_state': [42],
            'n_jobs': [1]
        },
        'random_forest': {
            'lookback_window': [30, 50],
            'n_estimators': [10, 20],
            'max_depth': [2, 3],
            'random_state': [42],
            'n_jobs': [1],
            'forecast_horizon': [50]
        },
        'svr': {
            'training_loss': ['mae'],
            'kernel': ['rbf', 'linear'],
            'C': [1.0, 10.0],
            'epsilon': [0.1],
            'gamma': ['scale'],
            'lookback_window': [50],
            'forecast_horizon': [10]
        },
        'prophet': {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.05, 0.1],
            'seasonality_prior_scale': [10.0],
            'yearly_seasonality': [True],
            'weekly_seasonality': [True],
            'daily_seasonality': [False]
        },
        
        # Deep Learning Models
        'lstm': {
            'units': [32, 64],
            'layers': [1, 2],
            'dropout': [0.1, 0.2],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32],
            'epochs': [50, 100],
            'sequence_length': [10],
            'training_loss': ['mae'],
            'forecast_horizon': [10]
        },
        
        # Foundation Models (no hyperparameters - will be empty dict)
        'chronos': {},
        'tiny_time_mixer': {},
        'moirai': {},
        'moirai_moe': {},
        'moment': {},
        'timesfm': {},
        'lagllama': {},
        'toto': {},
        'tabpfn': {}
    }
    
    # Adjust hyperparameters based on dataset characteristics
    time_steps = dataset_info.get('time_steps', 1000)
    is_univariate = dataset_info.get('is_univariate', True)
    
    # Adjust seasonal parameters based on frequency
    freq = dataset_info.get('time_freq', '1D')
    if 'H' in freq:  # Hourly data
        seasonal_periods = [24, 168]  # Daily, weekly
    elif 'D' in freq:  # Daily data
        seasonal_periods = [7, 30]  # Weekly, monthly
    elif 'W' in freq:  # Weekly data
        seasonal_periods = [4, 12]  # Monthly, quarterly
    elif 'M' in freq:  # Monthly data
        seasonal_periods = [12]  # Yearly
    else:
        seasonal_periods = [12, 24]
    
    # Update seasonal parameters
    base_hyperparams['exponential_smoothing']['seasonal_periods'] = [None] + seasonal_periods
    base_hyperparams['seasonal_naive']['sp'] = seasonal_periods
    base_hyperparams['theta']['sp'] = seasonal_periods[:2]  # Use first 2
    
    # Adjust LSTM parameters based on data size
    if time_steps < 1000:
        base_hyperparams['lstm']['epochs'] = [30, 50]
        base_hyperparams['lstm']['batch_size'] = [8, 16]
    elif time_steps > 10000:
        base_hyperparams['lstm']['epochs'] = [100, 150]
        base_hyperparams['lstm']['batch_size'] = [32, 64]
    
    # Adjust lookback windows based on data size
    if time_steps < 500:
        lookback_windows = [10, 20]
    elif time_steps < 2000:
        lookback_windows = [20, 30]
    else:
        lookback_windows = [30, 50]
    
    base_hyperparams['xgboost']['lookback_window'] = lookback_windows
    base_hyperparams['random_forest']['lookback_window'] = lookback_windows
    base_hyperparams['svr']['lookback_window'] = lookback_windows
    
    return base_hyperparams

def generate_config_file(dataset_info, output_dir):
    """Generate a single config file for a dataset."""
    
    dataset_name = dataset_info['dataset_name']
    dataset_type = dataset_info['dataset_type']
    
    # Generate hyperparameters
    hyperparams = generate_hyperparameters(dataset_info)
    
    # Create config structure with proper YAML formatting
    config = {
        'task': {
            'task_type': 'deterministic',
            'forecast_horizon': dataset_info['forecast_horizon'],
            'context_window': dataset_info['context_window'],
            'max_windows': dataset_info['max_windows'],
            'tuning_loss': 'mae',
            'dataset': {
                'name': f"{dataset_type}/{dataset_name}",
                'normalize': False,
                'handle_missing': 'interpolate'
            }
        },
        'evaluation': {
            'metrics': ['mae', 'mase', 'rmse', 'mape'],
            'logging': True
        },
        'model': hyperparams
    }
    
    # Create output directory
    type_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(type_dir, exist_ok=True)
    
    # Write config file with inline list formatting and no anchors
    output_file = os.path.join(type_dir, f"{dataset_name}.yaml")
    with open(output_file, 'w') as f:
        yaml.dump(config, f, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return output_file

def main():
    """Generate all config files."""
    
    # Load analysis results
    with open('dataset_analysis.json', 'r') as f:
        datasets = json.load(f)
    
    # Create output directory
    output_dir = 'benchmarking_pipeline/configs/datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate configs for successful datasets
    successful = [d for d in datasets if 'error' not in d]
    failed = [d for d in datasets if 'error' in d]
    
    print(f"Generating configs for {len(successful)} datasets...")
    
    generated_files = []
    for dataset_info in successful:
        try:
            output_file = generate_config_file(dataset_info, output_dir)
            generated_files.append(output_file)
            print(f"Generated: {output_file}")
        except Exception as e:
            print(f"Failed to generate config for {dataset_info['dataset_name']}: {e}")
    
    print(f"\nConfig generation complete!")
    print(f"  Generated: {len(generated_files)} config files")
    print(f"  Failed datasets: {len(failed)}")
    
    if failed:
        print(f"\nFailed datasets:")
        for d in failed:
            print(f"  {d['dataset_name']}: {d['error']}")
    
    # Create summary
    summary = {
        'total_datasets': len(datasets),
        'successful': len(successful),
        'failed': len(failed),
        'generated_configs': len(generated_files),
        'config_files': generated_files
    }
    
    with open('config_generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to config_generation_summary.json")

if __name__ == "__main__":
    main()
