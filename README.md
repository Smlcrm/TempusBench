# Time Series Forecasting Benchmarking Pipeline

A comprehensive framework for benchmarking time series forecasting models, including both traditional statistical models and modern foundation models.

## Overview

This project provides a unified benchmarking framework for evaluating the performance of various time series forecasting models. It supports:

- **Traditional Models**: ARIMA, LSTM, XGBoost, SVR, Prophet, Random Forest, Theta, DeepAR, Exponential Smoothing, Croston Classic, Seasonal Naive, TabPFN
- **Foundation Models**: Chronos, LagLlama, Moirai, TimesFM, Tiny Time Mixer, Toto, Moment
- **Deterministic and Stochastic Forecasting**: Automatic routing based on task type
- **Univariate and Multivariate Forecasting**: All models handle both seamlessly
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Isolated Execution**: Each model runs in its own conda environment to avoid dependency conflicts

## Project Structure

```
tempus_bench/
├── __init__.py                 # Package initialization
├── run_benchmark.py           # Main entry point for benchmarks
├── config/                    # Configuration system
│   ├── benchmark.yaml         # Default configuration
│   ├── settings.yaml          # System configuration
│   ├── manager.py             # Configuration management
│   ├── models.py              # Model configuration handling
│   └── validator.py           # Configuration validation
├── tasks/                     # Time series datasets
│   ├── univariate/           # 25 univariate time series tasks
│   │   ├── chickenpox_dense_univariate/
│   │   ├── coinbase_days_univariate/
│   │   └── ... (23 more)
│   └── multivariate/         # 23 multivariate time series tasks
│       ├── baggage_100_multivariate/
│       ├── madrid_transport_multivariate/
│       └── ... (21 more)
├── metrics/                   # Evaluation metrics
│   ├── __init__.py
│   ├── crps.py                # Continuous Ranked Probability Score
│   ├── weighted_interval_score.py
│   ├── mae.py
│   ├── mape.py
│   ├── mase.py
│   ├── rmse.py
│   └── evaluation.py
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── base_model.py          # Base class for all models
│   ├── arima/                 # ARIMA model
│   ├── lstm/                  # LSTM model
│   ├── xgboost/               # XGBoost model
│   ├── prophet/               # Prophet model
│   ├── chronos/               # Chronos foundation model
│   ├── lagllama/              # LagLlama foundation model
│   ├── moirai/                # Moirai foundation model
│   ├── moirai_moe/            # Moirai MoE foundation model
│   ├── toto/                  # Toto foundation model
│   ├── moment/                # Moment foundation model
│   ├── deepar/                # DeepAR model
│   └── ... (other models)
├── pipeline/                  # Core pipeline components
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── data_types.py         # Data structures and types
│   ├── preprocessor.py       # Data preprocessing
│   ├── model_executor.py     # Model execution in isolated environments
│   ├── hyperparameter_tuning.py
│   └── visualizer.py
└── utils/                     # Utility functions
    ├── __init__.py
    ├── envs.py               # Conda environment management
    ├── log_manager.py        # Unified logging (standard and TensorBoard)
    ├── model_config.py       # Model configuration handling
    └── paths.py              # Path management
```

## Key Features

### 1. Automatic Model Discovery

The framework automatically discovers available models from the models directory. Each model has a `settings.yaml` file that specifies its type (deterministic, stochastic, or hybrid).

All models handle both univariate and multivariate datasets internally.

```python
from tempus_bench.utils import get_available_models

# Get all available models
available_models = get_available_models()
print(available_models)
# Output: {'arima', 'lstm', 'xgboost', 'prophet', 'chronos', 'lagllama', 'moirai', ...}
```

### 2. Unified Model Interface

All models implement a consistent interface through base classes:

- **`BaseModel`**: Base class for all models with standard methods
- **`BaseModel`**: Enhanced base class for stochastic models

```python
from tempus_bench.models.base_model import BaseModel

# Deterministic model implementation
class MyModel(BaseModel):
    def train(self, y_context, y_target, **kwargs):
        # Training implementation
        pass
    
    def predict(self, y_context, **kwargs):
        # Prediction implementation
        pass
    
    def compute_metrics(self, y_true, y_pred):
        # Metrics computation
        pass
```

### 3. Comprehensive Data Handling

The pipeline automatically handles:
- **Multiple task formats** (CSV with metadata)
- **Flexible windowing** (context, train, validate splits)
- **Automatic frequency detection** from data
- **Data normalization** (optional)
- **Rolling window evaluation**

```python
from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import TaskConfig, EvaluationConfig, DatasetConfig

# Create task and evaluation configurations
task_config = TaskConfig(
    name="chickenpox_dense_univariate",
    task_path="tempus_bench/tasks/univariate/chickenpox_dense_univariate",
    forecast_horizon=25,
    context_window=50,
    dataset=DatasetConfig(
        file_name="chickenpox_dense_univariate.csv",
        normalize=True,
        handle_missing="interpolate"
    )
)

evaluation_config = EvaluationConfig(
    task_path="tempus_bench/tasks/univariate/chickenpox_dense_univariate",
    max_windows=10,
    max_num_variates=None
)

# Initialize data loader
data_loader = DataLoader(task_config, evaluation_config)

# Generate rolling windows
steps = [
    ('context', 50),
    ('train', 25),
    ('validate', 25)
]
window_iter = data_loader.generate_dataset_split(
    steps=steps,
    stride=1
)

# Iterate over windows
for window_idx, window_splits in window_iter:
    print(f"Window {window_idx}: {window_splits}")
```

### 4. Flexible Configuration

Configuration files support:
- **Model-specific parameters** (hyperparameter grids)
- **Task configuration** (context window, forecast horizon)
- **Evaluation settings** (metrics, loss functions)
- **System configuration** (paths, logging, TensorBoard)

```yaml
# benchmark.yaml
task_path: '*'  # Use all tasks

evaluation:
  tuning_loss: mae
  point_forecast_statistic: mean
  max_num_variates: 4
  max_windows: 5

model:
  # Traditional models with hyperparameter grids
  arima:
    p: [1, 2]
    d: [1]
    q: [1, 2]
    s: [2]
  
  xgboost:
    lookback_window: [30]
    n_estimators: [200]
    max_depth: [4]
    learning_rate: [0.05]
  
  # Foundation models (no hyperparameters needed)
  chronos: {}
  lagllama: {}
  moirai: {}
```

## Installation

### Prerequisites

- Python 3.8+
- Conda

**Note**: All models added to the `tempus_bench/models` directory must be compatible with Python 3.0 or later (Python 3.x series).

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd benchmark
   ```

2. **Install dependencies**:
   ```bash
   # Run the installation script
   source install.sh
   ```

3. **Verify installation**:
   ```bash
   python -c "from tempus_bench.utils import get_available_models; print('Installation successful!')"
   ```

## Usage

### Basic Usage via Command Line

```bash
# Activate the conda environment
conda activate sim.benchmarks

# Run benchmark with default configuration
python -m tempus_bench.run_benchmark

# Run with custom configuration
python -m tempus_bench.run_benchmark --config tempus_bench/config/benchmark.yaml

# The system will automatically:
# 1. Discover all available models (deterministic and stochastic)
# 2. Load all tasks from tempus_bench/tasks/
# 3. Perform hyperparameter tuning for each model
# 4. Evaluate models on rolling windows
# 5. Store results in runs/ directory
```

### Python API Usage

```python
from tempus_bench.run_benchmark import BenchmarkRunner

# Initialize and run benchmarks
config_path = "tempus_bench/config/benchmark.yaml"
runner = BenchmarkRunner(config_path=config_path)
runner.run()

# The system handles:
# - Hyperparameter optimization
# - Rolling window evaluation  
# - Multiple model execution
# - Result storage
```

### Running Individual Models

```python
from tempus_bench.utils.manager import Manager
from tempus_bench.pipeline.model_executor import ModelExecutor
from tempus_bench.utils.log_manager import LogManager

# First, create a Manager to load configurations
manager = Manager(
    config_path="tempus_bench/config/benchmark.yaml",
)

# Get a job config
job_config, _ = next(manager.generate_run_configs())

# Initialize executor
executor = ModelExecutor(
    job_config=job_config
)

# Execute a single model with specific hyperparameters
results = executor.execute_model(
    model_name='arima',
    hyperparameters={'p': 2, 'd': 1, 'q': 2, 's': 2},
    context_steps=50,
    train_steps=25,
    validate_steps=25,
    task_path="tempus_bench/tasks/univariate/chickenpox_dense_univariate",
    window_idx=0,
    config_path=job_config.config_path
)
```

### Scripts and Automation

```bash
# Run all benchmarks for all models
bash scripts/bash/run_all_benchmarks.sh

# Clean up conda environments
bash scripts/bash/cleanup_conda_envs.sh

# Each model runs in isolation in its own conda environment
# This prevents dependency conflicts between different models
```

### Adding New Models

**Important**: All models must be compatible with Python 3.0 or later (Python 3.x series). Ensure your model implementation and dependencies work with Python 3.x.

1. **Choose model type** (deterministic, stochastic, or hybrid):
   - **Deterministic**: Point forecasts (mean/median predictions)
   - **Stochastic**: Probabilistic forecasts (samples/quantiles)
   - **Hybrid**: Both point forecasts and samples

2. **Create model directory**:
   ```
   tempus_bench/models/my_model/
   ├── my_model_model.py
   ├── requirements.txt
   └── settings.yaml
   ```

3. **Create settings.yaml**:
   ```yaml
   # settings.yaml
   model_type: deterministic  # or 'stochastic' or 'hybrid'
   python_version: "3.11.13"  # Must be Python 3.0 or later (3.x series)
   ```
   
   **Note**: The `python_version` specified in `settings.yaml` must be Python 3.0 or later. The framework requires all models to be compatible with Python 3.x series.

4. **Implement model class**:
   ```python
   from tempus_bench.models.base_model import BaseModel
   
   class MyModelModel(BaseModel):
       def __init__(self, params, settings):
           super().__init__(params, settings)
           # Store model-specific params
           self.params = params
       
       def train(self, y_context, y_target, timestamps_context, timestamps_target, freq, **kwargs):
           # Training implementation
           # Must return self for method chaining
           return self
       
       def predict(self, y_context, timestamps_context, timestamps_target, freq, **kwargs):
           # Prediction implementation
           # Return numpy array of predictions
           return predictions
       
       def compute_metrics(self, y_true, y_pred):
           # Compute evaluation metrics
           return {'mae': mae, 'rmse': rmse}
   ```

5. **Add to configuration**:
   ```yaml
   # In tempus_bench/config/benchmark.yaml
   model:
     my_model:
       param1: [value1, value2]
       param2: [value3, value4]
   ```

The model will be automatically discovered and available for benchmarking!

## Model Categories

### Traditional Models

- **Statistical Models**: ARIMA, Theta, Seasonal Naive
- **Machine Learning**: XGBoost, Random Forest, SVR
- **Deep Learning**: LSTM, DeepAR
- **Ensemble**: TabPFN

### Foundation Models

- **Chronos**: Amazon's time series foundation model
- **LagLlama**: Large language model for time series
- **Moirai**: Microsoft's foundation model for forecasting
- **TimesFM**: Google's time series foundation model
- **Tiny Time Mixer**: Lightweight transformer model
- **Toto**: Multi-modal foundation model

## Evaluation Metrics

The framework supports various evaluation metrics:

- **Point Forecast Metrics**: MAE, RMSE, MAPE, SMAPE
- **Probabilistic Metrics**: CRPS, Interval Score
- **Custom Metrics**: Easy to add new evaluation functions

## Contributing

1. **Follow the project structure** for consistency
2. **Add comprehensive documentation** for new features
3. **Include tests** for new functionality
4. **Use type hints** for better code quality
5. **Follow PEP 8** style guidelines

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=tempus_bench
```

## License

[Add your license information here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tempus_bench,
  title={Time Series Forecasting Benchmarking Pipeline},
  author={[Your Name/Organization]},
  year={2024},
  url={[Repository URL]}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `configs/CONFIG_DOCUMENTATION.md`
- Review the test examples for usage patterns

## Roadmap

- [ ] Support for more foundation models
- [ ] Advanced hyperparameter optimization
- [ ] Distributed training capabilities
- [ ] Real-time forecasting pipeline
- [ ] Model interpretability tools
- [ ] Automated model selection
