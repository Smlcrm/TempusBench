#!/usr/bin/env python3
"""
Analyze all datasets to extract characteristics and calculate optimal task parameters.
"""

import os
import sys
import pandas as pd
import ast
import numpy as np
import re
import json
from pathlib import Path

# Add the benchmarking_pipeline to the path
sys.path.append('/Users/gandresr/Documents/GitHub/benchmark')

def parse_target_data(target_raw):
    """Parse and clean target data, handling both univariate and multivariate formats."""
    # Clean the target string to handle empty values before parsing
    target_cleaned_str = target_raw
    while ', ,' in target_cleaned_str or ',,' in target_cleaned_str:
        target_cleaned_str = re.sub(r',\s*,', ', None,', target_cleaned_str)
    target_cleaned_str = re.sub(r'\[\s*,', '[None,', target_cleaned_str)
    target_cleaned_str = re.sub(r',\s*\]', ', None]', target_cleaned_str)
    
    try:
        target_parsed = ast.literal_eval(target_cleaned_str)
    except SyntaxError:
        # If still fails, try more aggressive cleaning
        target_cleaned_str = target_cleaned_str.replace('""', 'None').replace("''", 'None')
        target_parsed = ast.literal_eval(target_cleaned_str)
    
    # Detect if univariate (1D list) or multivariate (2D list)
    is_univariate = isinstance(target_parsed, list) and len(target_parsed) > 0 and not isinstance(target_parsed[0], list)
    
    if is_univariate:
        # Convert 1D list to 2D: [v1, v2, ...] -> [[v1, v2, ...]]
        target_parsed = [target_parsed]
    
    # Convert to numpy array for analysis
    target_cleaned = []
    for row in target_parsed:
        cleaned_row = []
        for val in row:
            if val is None or val == "" or val == "None":
                cleaned_row.append(np.nan)
            else:
                try:
                    cleaned_row.append(float(val))
                except (ValueError, TypeError):
                    cleaned_row.append(np.nan)
        target_cleaned.append(cleaned_row)
    
    target = np.array(target_cleaned)
    return target, is_univariate

def analyze_dataset(csv_path):
    """Analyze a single dataset and return its characteristics."""
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Extract basic information
        time_start = df['start'].iloc[0]
        time_freq = df['freq'].iloc[0]
        target_raw = df['target'].iloc[0]
        
        # Parse target data
        target, is_univariate = parse_target_data(target_raw)
        
        # Calculate characteristics
        if is_univariate:
            time_steps = target.shape[1]  # For univariate: (1, time_steps)
            num_series = 1
        else:
            time_steps = target.shape[1]  # For multivariate: (num_series, time_steps)
            num_series = target.shape[0]
        
        # Calculate 10% window parameters
        context_window = max(1, int(time_steps * 0.10))
        forecast_horizon = min(128, context_window // 2)  # Min of 128 or half context_window
        
        # Calculate number of windows
        window_size = context_window + forecast_horizon
        stride = forecast_horizon
        num_windows = max(0, (time_steps - window_size) // stride + 1)
        
        # Set max_windows to 20
        max_windows = 20
        
        return {
            'dataset_name': os.path.basename(os.path.dirname(csv_path)),
            'csv_path': csv_path,
            'time_start': time_start,
            'time_freq': time_freq,
            'is_univariate': is_univariate,
            'num_series': num_series,
            'time_steps': time_steps,
            'context_window': context_window,
            'forecast_horizon': forecast_horizon,
            'max_windows': max_windows,
            'window_size': window_size,
            'stride': stride,
            'num_windows': num_windows,
            'target_shape': target.shape,
            'has_nan': np.isnan(target).any(),
            'nan_count': np.isnan(target).sum()
        }
    
    except Exception as e:
        return {
            'dataset_name': os.path.basename(os.path.dirname(csv_path)),
            'csv_path': csv_path,
            'error': str(e)
        }

def main():
    """Analyze all datasets and generate analysis report."""
    datasets_dir = 'benchmarking_pipeline/datasets'
    results = []
    
    print("Analyzing datasets...")
    
    # Analyze univariate datasets
    univariate_dir = os.path.join(datasets_dir, 'univariate')
    if os.path.exists(univariate_dir):
        for dataset_name in os.listdir(univariate_dir):
            dataset_path = os.path.join(univariate_dir, dataset_name)
            if os.path.isdir(dataset_path):
                csv_file = os.path.join(dataset_path, f"{dataset_name}.csv")
                if os.path.exists(csv_file):
                    print(f"Analyzing univariate: {dataset_name}")
                    result = analyze_dataset(csv_file)
                    result['dataset_type'] = 'univariate'
                    results.append(result)
    
    # Analyze multivariate datasets
    multivariate_dir = os.path.join(datasets_dir, 'multivariate')
    if os.path.exists(multivariate_dir):
        for dataset_name in os.listdir(multivariate_dir):
            dataset_path = os.path.join(multivariate_dir, dataset_name)
            if os.path.isdir(dataset_path):
                csv_file = os.path.join(dataset_path, f"{dataset_name}.csv")
                if os.path.exists(csv_file):
                    print(f"Analyzing multivariate: {dataset_name}")
                    result = analyze_dataset(csv_file)
                    result['dataset_type'] = 'multivariate'
                    results.append(result)
    
    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    # Convert numpy types in results
    for result in results:
        for key, value in result.items():
            result[key] = convert_numpy_types(value)
    
    output_file = 'dataset_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    
    # Print summary
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\nSummary:")
    print(f"  Total datasets: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed datasets:")
        for r in failed:
            print(f"  {r['dataset_name']}: {r['error']}")
    
    if successful:
        print(f"\nDataset characteristics:")
        print(f"  Univariate: {len([r for r in successful if r['dataset_type'] == 'univariate'])}")
        print(f"  Multivariate: {len([r for r in successful if r['dataset_type'] == 'multivariate'])}")
        
        time_steps = [r['time_steps'] for r in successful if 'time_steps' in r]
        if time_steps:
            print(f"  Time steps - Min: {min(time_steps)}, Max: {max(time_steps)}, Avg: {np.mean(time_steps):.0f}")
        
        windows = [r['num_windows'] for r in successful if 'num_windows' in r]
        if windows:
            print(f"  Windows - Min: {min(windows)}, Max: {max(windows)}, Avg: {np.mean(windows):.1f}")

if __name__ == "__main__":
    main()
