"""
Benchmarking Pipeline Package

A comprehensive framework for benchmarking time series forecasting models, including both traditional
statistical models and modern foundation models.

This package provides:
- Model routing and execution for univariate/multivariate time series forecasting
- Support for traditional models (ARIMA, LSTM, XGBoost, etc.) and foundation models (Chronos, LagLlama, Moirai, etc.)
- Automated data loading, preprocessing, and evaluation pipelines
- Hyperparameter tuning and model comparison capabilities
- Comprehensive evaluation metrics and visualization tools

Package Structure:
- models/: Model implementations organized by type (deterministic, stochastic)
- pipeline/: Core data processing and evaluation pipeline components
- configs/: Configuration files for different model types and datasets
- metrics/: Evaluation metrics and scoring functions
- trainer/: Hyperparameter tuning and model training utilities
- utils/: Utility functions and configuration validation

Usage:
    from tempus_bench import model_router
    from tempus_bench.pipeline import DataLoader, ModelExecutor
    
    # Load data and run benchmarks
    data_loader = DataLoader(config)
    datasets = data_loader.load_several_chunks(3)
    
    # Execute models
    model_executor = ModelExecutor(config_path, run_dir, datasets_dir)
    results = model_executor.execute_model(model_name, hyperparameters, context_steps, train_steps, validate_steps, dataset_path, window_idx)
"""

__version__ = "1.0.0"
__author__ = "Benchmarking Pipeline Team"
__description__ = "Comprehensive time series forecasting model benchmarking framework"
