#!/usr/bin/env python3
"""
Script to unify all metrics from nested folder structure into a single CSV file.

The script reads metrics from the structure: metrics/model_type/task_type/model/metrics.csv
and aggregates them into a single CSV with headers: task_type, model, MASE, MSE, RMSE
"""

import os
import pandas as pd
from pathlib import Path
import argparse


def find_metrics_files(metrics_dir):
    """
    Recursively find all metrics.csv files in the metrics directory.

    Args:
        metrics_dir (str): Path to the metrics directory

    Returns:
        list: List of tuples (file_path, model_type, task_type, model_name)
    """
    metrics_files = []
    metrics_path = Path(metrics_dir)

    if not metrics_path.exists():
        print(f"Error: Metrics directory '{metrics_dir}' does not exist")
        return []

    # Walk through the directory structure
    for model_type_dir in metrics_path.iterdir():
        if not model_type_dir.is_dir():
            continue

        model_type = model_type_dir.name

        for task_type_dir in model_type_dir.iterdir():
            if not task_type_dir.is_dir():
                continue

            task_type = task_type_dir.name

            for model_dir in task_type_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name
                metrics_file = model_dir / "metrics.csv"

                if metrics_file.exists():
                    metrics_files.append(
                        (metrics_file, model_type, task_type, model_name)
                    )

    return metrics_files


def extract_metrics_from_file(file_path, model_type, task_type, model_name):
    """
    Extract metrics from a single metrics.csv file.

    Args:
        file_path (Path): Path to the metrics.csv file
        model_type (str): Type of model (e.g., univariate, multivariate)
        task_type (str): Type of task/dataset
        model_name (str): Name of the specific model

    Returns:
        dict: Dictionary with extracted metrics
    """
    try:
        # Read the metrics CSV file
        df = pd.read_csv(file_path)

        # Extract the required metrics
        metrics = {
            "task_type": task_type,
            "model": model_name,
            "MASE": None,
            "MSE": None,
            "RMSE": None,
        }

        # Check if the required columns exist and extract values
        if "mase" in df.columns:
            metrics["MASE"] = df["mase"].iloc[0] if len(df) > 0 else None

        if "mse" in df.columns:
            metrics["MSE"] = df["mse"].iloc[0] if len(df) > 0 else None
        elif "rmse" in df.columns:
            # If MSE is not available, we can't calculate it from RMSE without additional data
            # So we'll leave it as None
            pass

        if "rmse" in df.columns:
            metrics["RMSE"] = df["rmse"].iloc[0] if len(df) > 0 else None

        return metrics

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def unify_metrics(metrics_dir, output_file):
    """
    Main function to unify all metrics into a single CSV file.

    Args:
        metrics_dir (str): Path to the metrics directory
        output_file (str): Path to the output CSV file
    """
    print(f"Searching for metrics files in: {metrics_dir}")

    # Find all metrics files
    metrics_files = find_metrics_files(metrics_dir)

    if not metrics_files:
        print("No metrics files found!")
        return

    print(f"Found {len(metrics_files)} metrics files")

    # Extract metrics from each file
    all_metrics = []
    for file_path, model_type, task_type, model_name in metrics_files:
        print(f"Processing: {model_type}/{task_type}/{model_name}")

        metrics = extract_metrics_from_file(
            file_path, model_type, task_type, model_name
        )
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        print("No valid metrics extracted!")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_metrics)

    # Reorder columns to match the required format
    df = df[["task_type", "model", "MASE", "MSE", "RMSE", "MAPE"]]

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Unified metrics saved to: {output_file}")
    print(f"Total records: {len(df)}")

    # Display summary
    print("\nSummary:")
    print(f"Unique task types: {df['task_type'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Records with MASE: {df['MASE'].notna().sum()}")
    print(f"Records with MSE: {df['MSE'].notna().sum()}")
    print(f"Records with RMSE: {df['RMSE'].notna().sum()}")
    print(f"Records with MAPE: {df['MAPE'].notna().sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Unify metrics from nested folder structure"
    )
    parser.add_argument(
        "--metrics-dir",
        default="benchmarking_pipeline/metrics",
        help="Path to the metrics directory (default: benchmarking_pipeline/metrics)",
    )
    parser.add_argument(
        "--output",
        default="unified_metrics.csv",
        help="Output CSV file name (default: unified_metrics.csv)",
    )

    args = parser.parse_args()

    # Run the unification
    unify_metrics(args.metrics_dir, args.output)


if __name__ == "__main__":
    main()
