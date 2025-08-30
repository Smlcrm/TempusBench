# Time Series Forecasting Benchmarking Pipeline

A comprehensive framework for benchmarking time series forecasting models, including both traditional statistical models and modern foundation models.

## Overview

This project provides a unified benchmarking framework for evaluating the performance of various time series forecasting models. It supports:

- **Traditional Models**: ARIMA, LSTM, XGBoost, SVR, Prophet, Random Forest, Theta, DeepAR, TabPFN
- **Foundation Models**: Chronos, LagLlama, Moirai, TimesFM, Tiny Time Mixer, Toto
- **Univariate and Multivariate Forecasting**: Automatic routing based on data characteristics
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Hyperparameter Tuning**: Automated optimization of model parameters

## Project Structure

```
benchmarking_pipeline/
├── __init__.py                 # Package initialization and documentation
├── cli.py                     # Command-line interface
├── configs/                   # Configuration files
│   ├── all_model_multivariate.yaml
│   ├── all_model_univariate.yaml
│   └── CONFIG_DOCUMENTATION.md
├── datasets/                  # Time series datasets
│   ├── australian_electricity_demand/
│   ├── azure_vm_traces_2017/
│   ├── bitcoin_with_missing/
│   ├── china_air_quality/
│   ├── covid_deaths/
│   ├── tourism_monthly/
│   └── ... (many more)
├── metrics/                   # Evaluation metrics
│   ├── __init__.py
│   ├── crps.py              # Continuous Ranked Probability Score
│   ├── interval_score.py    # Interval score for prediction intervals
│   └── ... (other metrics)
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── base_model.py        # Base class for traditional models
│   ├── foundation_model.py  # Base class for foundation models
│   ├── model_router.py      # Intelligent model routing
│   ├── univariate/          # Univariate-only models
│   │   ├── arima/
│   │   ├── lstm/
│   │   ├── prophet/
│   │   └── ... (other models)
│   ├── multivariate/        # Models with separate multivariate implementations
│   │   ├── arima/
│   │   ├── lstm/
│   │   ├── xgboost/
│   │   └── ... (other models)
│   └── anyvariate/          # Models that handle both univariate and multivariate
│       ├── chronos/         # Amazon Chronos foundation model
│       ├── lagllama/        # LagLlama foundation model
│       ├── moirai/          # Moirai foundation model
│       └── ... (other models)
├── pipeline/                  # Core pipeline components
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── data_types.py        # Data structures and types
│   ├── evaluator.py         # Model evaluation
│   ├── logger.py            # Logging and metrics storage
│   ├── preprocessor.py      # Data preprocessing
│   ├── trainer.py           # Model training and hyperparameter tuning
│   ├── visualizer.py        # Plots and visualizations
│   └── ... (other components)
├── trainer/                   # Training utilities
│   ├── foundation_model_tuning.py
│   └── hyperparameter_tuning.py
└── utils/                     # Utility functions
    ├── __init__.py
    └── config_validator.py   # Configuration validation
```

## Key Features

### 1. Intelligent Model Routing

The `ModelRouter` automatically discovers available models and routes requests based on:
- **Data characteristics** (univariate vs multivariate)
- **Model capabilities** (univariate-only, multivariate-only, or anyvariate)
- **Folder structure** (automatic discovery)

```python
from benchmarking_pipeline.models import model_router

# Get available models
available = model_router.get_available_models()
print(available)
# Output: {'anyvariate': ['chronos', 'lagllama', 'moirai'], 
#          'multivariate': ['arima', 'lstm', 'xgboost'], 
#          'univariate_only': ['prophet', 'theta']}

# Get model path for specific data
folder_path, file_name, class_name = model_router.get_model_path_by_target_count(
    'arima', num_targets=3, variant='multivariate'
)
```

### 2. Unified Model Interface

All models implement a consistent interface through base classes:

- **`BaseModel`**: For traditional statistical and ML models
- **`FoundationModel`**: For large pre-trained foundation models

```python
from benchmarking_pipeline.models import BaseModel, FoundationModel

# Traditional model
class MyModel(BaseModel):
    def train(self, y_context, x_context=None, **kwargs):
        # Implementation
        pass
    
    def predict(self, y_context=None, x_target=None, **kwargs):
        # Implementation
        pass

# Foundation model
class MyFoundationModel(FoundationModel):
    def train(self, y_context, **kwargs):
        # Initialize pre-trained model
        pass
    
    def predict(self, y_context, **kwargs):
        # Generate predictions
        pass
```

### 3. Comprehensive Data Handling

The pipeline automatically handles:
- **Multiple data formats** (CSV chunks with metadata)
- **Automatic splitting** (train/validation/test)
- **Missing data handling** (interpolation, deletion)
- **Data normalization** (optional)
- **Frequency alignment** (automatic detection)

```python
from benchmarking_pipeline.pipeline import DataLoader

# Load data
data_loader = DataLoader(config)
datasets = data_loader.load_several_chunks(3)

# Each dataset contains train/validation/test splits
for dataset in datasets:
    print(f"Train: {len(dataset.train.targets)} samples")
    print(f"Validation: {len(dataset.validation.targets)} samples")
    print(f"Test: {len(dataset.test.targets)} samples")
```

### 4. Flexible Configuration

Configuration files support:
- **Model-specific parameters** (hyperparameter grids)
- **Dataset configuration** (paths, frequencies, split ratios)
- **Evaluation settings** (metrics, validation strategies)
- **Training parameters** (batch sizes, learning rates, epochs)

```yaml
# Example configuration
test_type: deterministic
dataset:
  name: china_air_quality
  path: datasets/china_air_quality
  frequency: H
  forecast_horizon: [10, 25, 50]
  split_ratio: [0.8, 0.1, 0.1]
  normalize: false
  handle_missing: interpolate

model:
  # Foundation models
  chronos:
    model_size: ['small', 'base']
    context_length: [8, 16]
    num_samples: [5, 10]
  
  # Traditional models
  arima:
    p: [0, 1, 2]
    d: [0, 1]
    q: [0, 1, 2]
    s: [2, 4]

evaluation:
  type: deterministic
  metrics: [mae, rmse, mape]
```

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd benchmark
   ```

2. **Install dependencies**:
   ```bash
   # Create conda environment
   conda create -n sim.benchmarks python=3.11.13
   conda activate sim.benchmarks
   
   ./install.sh
   ```

3. **Verify installation**:
   ```bash
   python -c "from benchmarking_pipeline import model_router; print('Installation successful!')"
   ```

## Usage

### Basic Usage

```python
from benchmarking_pipeline.pipeline import DataLoader, Trainer
from benchmarking_pipeline.models import model_router

# Load configuration
import yaml
with open('configs/all_model_multivariate.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
data_loader = DataLoader(config)
datasets = data_loader.load_several_chunks(3)

# Train and evaluate models
trainer = Trainer(config)
results = trainer.run_benchmark(datasets)
```

### Command Line Interface

```bash
# Run benchmark with default configuration
python -m benchmarking_pipeline.cli

# Run with custom configuration
python -m benchmarking_pipeline.cli --config configs/my_config.yaml

# Specify output directory
python -m benchmarking_pipeline.cli --output-dir results/
```

### Adding New Models

1. **Create model directory**:
   ```
   models/univariate/my_model/
   ├── my_model_model.py
   └── requirements.txt
   ```

2. **Implement model class**:
   ```python
   from benchmarking_pipeline.models.base_model import BaseModel
   
   class MyModelModel(BaseModel):
       def __init__(self, config=None, config_file=None):
           super().__init__(config, config_file)
           # Initialize model-specific parameters
       
       def train(self, y_context, **kwargs):
           # Training implementation
           pass
       
       def predict(self, **kwargs):
           # Prediction implementation
           pass
   ```

3. **Add to configuration**:
   ```yaml
   model:
     my_model:
       param1: [value1, value2]
       param2: [value3, value4]
   ```

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
pytest --cov=benchmarking_pipeline
```

## License

[Add your license information here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{benchmarking_pipeline,
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
