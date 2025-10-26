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
- trainer: Manages model training and hyperparameter tuning
- visualizer: Creates plots and visualizations
- batch_utils: Utilities for batch processing
- forecast_horizon: Handles forecast horizon calculations
- model_executor: Executes model training and evaluation

Usage:
    from tempus_bench.pipeline import DataLoader, Trainer, Evaluator
    
    # Load data
    data_loader = DataLoader(config)
    datasets = data_loader.load_several_chunks(3)
    
    # Train models
    trainer = Trainer(config)
    results = trainer.run_benchmark(datasets)
"""
