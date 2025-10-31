"""
Pipeline package for the benchmarking framework.

This package contains the core components for data processing, model training, and evaluation
in the time series forecasting benchmarking pipeline.

Components:
- data_loader: Loads and processes time series data chunks
- data_types: Defines data structures for datasets and splits
- metric_registry: Computes evaluation metrics
- preprocessor: Applies data preprocessing steps
- model_executor: Executes model training and evaluation in isolated environments
- visualizer: Creates plots and visualizations
- hyperparameter_tuning: Handles hyperparameter optimization

Usage:
    from tempus_bench.pipeline import DataLoader, ModelExecutor

    # Load data
    data_loader = DataLoader(job_config.task_config, job_config.evaluation_config)
    datasets = data_loader.load_several_chunks(3)

    # Execute models (runs in isolated conda environment via CLI)
    model_executor = ModelExecutor(job_config)
    results = model_executor.execute_model(
        model_name='arima',
        hyperparameters={'p': 2, 'd': 1, 'q': 2},
        context_steps=50,
        train_steps=100,
        validate_steps=20,
        task_path='path/to/task.csv',
        window_idx=0,
        config_path=job_config.config_path
    )
"""
