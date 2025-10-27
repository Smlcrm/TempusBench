"""
Pipeline package for the benchmarking framework.

This package contains the core components for data processing, model training, and evaluation
in the time series forecasting benchmarking pipeline.

Components:
- data_loader: Loads and processes time series data chunks
- data_types: Defines data structures for datasets and splits
- evaluator: Computes evaluation metrics
- logger: Handles logging and metrics storage
- preprocessor: Applies data preprocessing steps
- model_executor: Executes model training and evaluation in isolated environments
- visualizer: Creates plots and visualizations
- batch_utils: Utilities for batch processing
- forecast_horizon: Handles forecast horizon calculations

Usage:
    from tempus_bench.pipeline import DataLoader, ModelExecutor
    
    # Load data
    data_loader = DataLoader(config)
    datasets = data_loader.load_several_chunks(3)
    
    # Execute models
    model_executor = ModelExecutor(config_path, run_dir, datasets_dir)
    results = model_executor.execute_model(model_name, hyperparameters, context_steps, train_steps, validate_steps, dataset_path, window_idx)
"""
