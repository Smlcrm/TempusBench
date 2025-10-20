#!/usr/bin/env python3
"""
Script to update all config files to include all deterministic evaluation metrics.
"""

import os
import yaml
from pathlib import Path

def update_config_metrics(config_path):
    """Update a single config file to include all deterministic metrics."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update evaluation.metrics to include all deterministic metrics
    if 'evaluation' in config and 'metrics' in config['evaluation']:
        config['evaluation']['metrics'] = ['mae', 'mase', 'rmse', 'mape']
        
        # Write the updated config back to the file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Updated {config_path}")
        return True
    else:
        print(f"No evaluation.metrics found in {config_path}")
        return False

def main():
    """Update all config files in the configs directory."""
    configs_dir = Path("benchmarking_pipeline/configs/datasets")
    
    if not configs_dir.exists():
        print(f"Configs directory not found: {configs_dir}")
        return
    
    updated_count = 0
    total_count = 0
    
    # Process all YAML files in subdirectories
    for yaml_file in configs_dir.rglob("*.yaml"):
        total_count += 1
        if update_config_metrics(yaml_file):
            updated_count += 1
    
    print(f"\nSummary:")
    print(f"Total config files found: {total_count}")
    print(f"Successfully updated: {updated_count}")

if __name__ == "__main__":
    main()
