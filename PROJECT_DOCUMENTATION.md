# Project Documentation

This document provides comprehensive documentation for the Time Series Forecasting Benchmarking Pipeline project, including architecture, design decisions, and implementation details.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Design Decisions](#design-decisions)
4. [Implementation Details](#implementation-details)
5. [Data Flow](#data-flow)
6. [Model System](#model-system)
7. [Pipeline Components](#pipeline-components)
8. [Configuration System](#configuration-system)
9. [Testing Strategy](#testing-strategy)
10. [Performance Considerations](#performance-considerations)
11. [Security and Best Practices](#security-and-best-practices)
12. [Deployment](#deployment)
13. [Monitoring and Logging](#monitoring-and-logging)
14. [Troubleshooting](#troubleshooting)
15. [Future Enhancements](#future-enhancements)

## Project Overview

### Purpose

The Time Series Forecasting Benchmarking Pipeline is a comprehensive framework designed to evaluate and compare the performance of various time series forecasting models. The project addresses the need for standardized benchmarking in the time series forecasting domain, which has become increasingly important with the emergence of foundation models and the need to compare them against traditional statistical and machine learning approaches.

### Key Features

- **Unified Interface**: Consistent API for both traditional and foundation models
- **Intelligent Routing**: Automatic model selection based on data characteristics
- **Comprehensive Evaluation**: Multiple metrics and validation strategies
- **Flexible Configuration**: YAML-based configuration system
- **Extensible Architecture**: Easy addition of new models and metrics
- **Reproducible Results**: Deterministic execution and result storage

### Target Users

- **Researchers**: Academic and industrial researchers in time series forecasting
- **Data Scientists**: Practitioners evaluating forecasting models for production use
- **ML Engineers**: Engineers building forecasting systems and pipelines
- **Students**: Learners studying time series forecasting and model evaluation

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Configuration │    │   Data Sources  │    │   Model Store   │
│      System     │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Pipeline                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Data Loader │ │Preprocessor │ │   Trainer   │ │ Evaluator   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Results &     │    │   Logging &     │    │   Visualization │
│   Metrics       │    │   Monitoring    │    │   & Reporting   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

#### 1. Configuration Layer
- **YAML Configuration Files**: Human-readable configuration format
- **Configuration Validator**: Ensures configuration correctness
- **Environment Management**: Handles different deployment environments

#### 2. Data Layer
- **Data Loader**: Handles various data formats and sources
- **Data Preprocessor**: Applies transformations and cleaning
- **Data Types**: Structured data representations

#### 3. Model Layer
- **Model Router**: Intelligent model selection and routing
- **Base Classes**: Abstract interfaces for model implementations
- **Model Implementations**: Concrete model implementations

#### 4. Pipeline Layer
- **Training Pipeline**: Model training and optimization
- **Evaluation Pipeline**: Performance assessment and metrics
- **Inference Pipeline**: Prediction generation

#### 5. Output Layer
- **Results Storage**: Structured result storage
- **Logging System**: Comprehensive logging and monitoring
- **Visualization**: Charts, plots, and reports

## Design Decisions

### 1. Model Architecture

#### Why Separate Base Classes?

The decision to have separate `BaseModel` and `FoundationModel` classes was driven by:

- **Different Training Paradigms**: Traditional models require training from scratch, while foundation models are typically fine-tuned
- **Resource Requirements**: Foundation models have different memory and computational requirements
- **Interface Consistency**: Both types need similar interfaces for prediction and evaluation
- **Future Extensibility**: Allows for different optimization strategies and evaluation methods

#### Implementation Details

```python
# Traditional models inherit from BaseModel
class ArimaModel(BaseModel):
    def train(self, y_context, **kwargs):
        # Train from scratch
        pass
    
    def predict(self, **kwargs):
        # Generate predictions
        pass

# Foundation models inherit from FoundationModel
class ChronosModel(FoundationModel):
    def train(self, y_context, **kwargs):
        # Initialize pre-trained model
        pass
    
    def predict(self, **kwargs):
        # Generate predictions
        pass
```

### 2. Model Routing System

#### Why Intelligent Routing?

The `ModelRouter` provides several benefits:

- **Automatic Discovery**: Models are automatically discovered from the folder structure
- **Flexible Categorization**: Supports univariate, multivariate, and anyvariate models
- **Consistent Interface**: All models are accessed through the same interface
- **Easy Extension**: Adding new models requires only folder structure changes

#### Routing Logic

```python
def get_model_path_by_target_count(self, model_name, num_targets, variant=None):
    # 1. Check if model is anyvariate (handles both univariate and multivariate)
    if model_name in self.anyvariate_models:
        return self._route_to_anyvariate(model_name)
    
    # 2. Check if model has separate implementations
    elif model_name in self.multivariate_models:
        return self._route_to_multivariate(model_name, variant)
    
    # 3. Check if model is univariate-only
    elif model_name in self.univariate_models:
        return self._route_to_univariate(model_name, variant)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

### 3. Data Handling Strategy

#### Why Treat All Data as Multivariate?

The decision to treat all data as multivariate (where univariate is just `num_targets == 1`) provides:

- **Consistent Interface**: All models use the same data interface
- **Future Flexibility**: Easy to extend univariate models to multivariate
- **Reduced Complexity**: Single data path for all model types
- **Better Performance**: Avoids unnecessary data transformations

#### Implementation

```python
def _extract_target_structure(self, target_data):
    """Infer target structure from data."""
    if isinstance(target_data, str):
        parsed = eval(target_data)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                # Multivariate: [[1,2,3], [4,5,6]]
                self.num_targets = len(parsed)
            else:
                # Univariate: [1,2,3,4,5] (multivariate with 1 target)
                self.num_targets = 1
        else:
            # Single value (multivariate with 1 target)
            self.num_targets = 1
```

### 4. Configuration System

#### Why YAML?

YAML was chosen for configuration because:

- **Human Readable**: Easy to understand and modify
- **Hierarchical Structure**: Natural representation of nested configurations
- **Wide Support**: Excellent Python support with PyYAML
- **Comments**: Support for inline documentation
- **Validation**: Easy to validate structure and content

#### Configuration Structure

```yaml
# Top-level configuration
test_type: deterministic
tensorboard: true

# Dataset configuration
dataset:
  name: china_air_quality
  path: datasets/china_air_quality
  frequency: H
  forecast_horizon: [10, 25, 50]
  split_ratio: [0.8, 0.1, 0.1]

# Model configuration
model:
  chronos:
    model_size: ['small', 'base']
    context_length: [8, 16]
  
  arima:
    p: [0, 1, 2]
    d: [0, 1]
    q: [0, 1, 2]

# Evaluation configuration
evaluation:
  type: deterministic
  metrics: [mae, rmse, mape]
```

## Implementation Details

### 1. Model Implementation Pattern

All models follow a consistent implementation pattern:

```python
class ModelName(BaseModel):
    def __init__(self, config=None, config_file=None):
        super().__init__(config, config_file)
        # Extract model-specific parameters
        self.param1 = self.config.get('param1', default_value)
        self.param2 = self.config.get('param2', default_value)
        
        # Initialize model state
        self.model_ = None
        self.is_fitted = False
    
    def train(self, y_context, **kwargs):
        # Training implementation
        # Set self.is_fitted = True when complete
        return self
    
    def predict(self, **kwargs):
        # Prediction implementation
        # Return numpy array with shape (n_samples, forecast_horizon)
        pass
    
    def get_params(self):
        # Return current parameters
        return {'param1': self.param1, 'param2': self.param2}
    
    def set_params(self, **params):
        # Update parameters
        # Reset model if parameters change
        return self
```

### 2. Data Flow Implementation

The data flow is implemented through a series of transformations:

```python
# 1. Load raw data
raw_dataset = data_loader.load_single_chunk(chunk_id)

# 2. Preprocess data
preprocessed_data = preprocessor.preprocess(raw_dataset)

# 3. Train model
model.train(preprocessed_data.train.targets)

# 4. Generate predictions
predictions = model.predict(
    y_context=preprocessed_data.validation.targets,
    forecast_horizon=len(preprocessed_data.test.targets)
)

# 5. Evaluate performance
metrics = evaluator.evaluate(predictions, preprocessed_data.test.targets)
```

### 3. Error Handling Strategy

The project uses a fail-fast approach with comprehensive error handling:

```python
def load_single_chunk(self, chunk_id):
    chunk_file = f"chunk{chunk_id:03d}.csv"
    chunk_path = os.path.join(self.dataset_path, chunk_file)
    
    print(f"[DEBUG] Loading chunk file: {chunk_file}")
    exit()
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
    
    try:
        chunk_data = pd.read_csv(chunk_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read chunk {chunk_id}: {e}")
    
    # Validate data structure
    required_columns = ['item_id', 'start', 'freq', 'target']
    missing_columns = [col for col in required_columns if col not in chunk_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return self._create_dataset(chunk_data)
```

## Data Flow

### 1. Data Loading Flow

```
Raw Data Files → DataLoader → Dataset Objects → Preprocessor → Clean Data
     ↓              ↓            ↓              ↓           ↓
  CSV Chunks   Parse Files   Create Splits   Transform   Ready Data
```

### 2. Model Training Flow

```
Clean Data → Model Selection → Model Training → Model Validation → Trained Model
    ↓            ↓              ↓              ↓              ↓
Train/Val   Router Logic   Fit Algorithm   Performance   Ready for
  Split                    to Data         Assessment    Prediction
```

### 3. Evaluation Flow

```
Trained Model → Test Data → Generate Predictions → Compute Metrics → Results
      ↓            ↓              ↓                  ↓           ↓
Model State   Test Split    Forecast Values      Calculate   Store and
  Ready       Loaded        for Horizon          Scores      Report
```

### 4. Complete Pipeline Flow

```
Configuration → Data Loading → Preprocessing → Model Training → Evaluation → Results
      ↓            ↓              ↓              ↓              ↓           ↓
Load YAML    Load Chunks    Clean/Transform   Train Models   Assess      Store
  Config      from Files     Data              on Train      Performance  Output
```

## Model System

### 1. Model Categories

#### Traditional Models
- **Statistical Models**: ARIMA, Theta, Seasonal Naive
- **Machine Learning**: XGBoost, Random Forest, SVR
- **Deep Learning**: LSTM, DeepAR
- **Ensemble**: TabPFN

#### Foundation Models
- **Chronos**: Amazon's time series foundation model
- **LagLlama**: Large language model for time series
- **Moirai**: Microsoft's foundation model
- **TimesFM**: Google's time series foundation model
- **Tiny Time Mixer**: Lightweight transformer
- **Toto**: Multi-modal foundation model

### 2. Model Interface Design

The model interface is designed to be:

- **Consistent**: All models implement the same methods
- **Flexible**: Supports different input/output formats
- **Extensible**: Easy to add new model types
- **Efficient**: Minimizes data copying and transformations

```python
@abstractmethod
def train(self, y_context, x_context=None, y_target=None, x_target=None, **kwargs):
    """Train the model on given data."""
    pass

@abstractmethod
def predict(self, y_context=None, x_target=None, forecast_horizon=None, **kwargs):
    """Generate predictions."""
    pass

@abstractmethod
def get_params(self):
    """Get current model parameters."""
    pass

@abstractmethod
def set_params(self, **params):
    """Set model parameters."""
    pass
```

### 3. Model Lifecycle Management

Models follow a clear lifecycle:

1. **Initialization**: Load configuration and set parameters
2. **Training**: Fit the model to training data
3. **Validation**: Assess performance on validation data
4. **Prediction**: Generate forecasts for test data
5. **Evaluation**: Compute performance metrics
6. **Persistence**: Save model state for later use

## Pipeline Components

### 1. Data Loader

The `DataLoader` handles:

- **File Discovery**: Automatically finds data chunks
- **Format Parsing**: Handles CSV and other formats
- **Data Validation**: Ensures data integrity
- **Split Creation**: Creates train/validation/test splits

### 2. Preprocessor

The `Preprocessor` provides:

- **Data Cleaning**: Handle missing values and outliers
- **Normalization**: Scale features appropriately
- **Feature Engineering**: Create derived features
- **Data Validation**: Ensure data quality

### 3. Trainer

The `Trainer` manages:

- **Model Selection**: Choose appropriate models
- **Hyperparameter Tuning**: Optimize model parameters
- **Training Orchestration**: Coordinate training process
- **Validation**: Monitor training progress

### 4. Evaluator

The `Evaluator` computes:

- **Point Forecast Metrics**: MAE, RMSE, MAPE, SMAPE
- **Probabilistic Metrics**: CRPS, Interval Score
- **Statistical Tests**: Significance testing
- **Performance Analysis**: Detailed model assessment

## Configuration System

### 1. Configuration Structure

The configuration system is organized hierarchically:

```
Configuration
├── Global Settings
│   ├── test_type
│   ├── tensorboard
│   └── output_directory
├── Dataset Configuration
│   ├── name
│   ├── path
│   ├── frequency
│   ├── forecast_horizon
│   ├── split_ratio
│   ├── normalize
│   ├── handle_missing
│   └── chunks
├── Model Configuration
│   ├── Foundation Models
│   │   ├── chronos
│   │   ├── lagllama
│   │   └── moirai
│   └── Traditional Models
│       ├── arima
│       ├── lstm
│       └── xgboost
└── Evaluation Configuration
    ├── type
    ├── metrics
    ├── validation_strategy
    └── cross_validation_folds
```

### 2. Configuration Validation

The configuration system includes comprehensive validation:

```python
def validate_config(config):
    """Validate configuration structure and values."""
    # Check required sections
    required_sections = ['dataset', 'model', 'evaluation']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate dataset configuration
    validate_dataset_config(config['dataset'])
    
    # Validate model configuration
    validate_model_config(config['model'])
    
    # Validate evaluation configuration
    validate_evaluation_config(config['evaluation'])
    
    return True
```

### 3. Environment-Specific Configuration

The system supports environment-specific configurations:

```yaml
# Development configuration
development:
  dataset:
    chunks: 1
  model:
    arima:
      p: [1]
      d: [1]
      q: [1]

# Production configuration
production:
  dataset:
    chunks: 10
  model:
    arima:
      p: [0, 1, 2]
      d: [0, 1]
      q: [0, 1, 2]
```

## Testing Strategy

### 1. Test Categories

The testing strategy is organized into multiple categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Smoke Tests**: Quick health checks
- **Performance Tests**: Test system performance

### 2. Test Data Management

Test data is managed through:

- **Synthetic Data**: Programmatically generated test data
- **Fixture System**: Reusable test configurations
- **Mock Objects**: Simulated external dependencies
- **Data Cleanup**: Automatic cleanup after tests

### 3. Test Coverage

The testing strategy aims for:

- **High Coverage**: Target 80%+ code coverage
- **Critical Path Coverage**: Ensure all critical paths are tested
- **Edge Case Coverage**: Test boundary conditions and error cases
- **Performance Coverage**: Test system performance under various conditions

## Performance Considerations

### 1. Memory Management

The system is designed for efficient memory usage:

- **Chunked Processing**: Process large datasets in manageable chunks
- **Lazy Loading**: Load data only when needed
- **Memory Pooling**: Reuse memory for similar operations
- **Garbage Collection**: Explicit cleanup of large objects

### 2. Computational Efficiency

Performance optimizations include:

- **Parallel Processing**: Use multiple cores for independent operations
- **Vectorized Operations**: Use numpy/pandas vectorized operations
- **Caching**: Cache expensive computations
- **Lazy Evaluation**: Defer computation until needed

### 3. Scalability

The system is designed to scale:

- **Horizontal Scaling**: Distribute work across multiple machines
- **Vertical Scaling**: Optimize for single-machine performance
- **Resource Management**: Efficient use of available resources
- **Load Balancing**: Distribute load evenly across resources

## Security and Best Practices

### 1. Input Validation

All inputs are validated:

- **Configuration Validation**: Ensure configuration correctness
- **Data Validation**: Validate data integrity and format
- **Parameter Validation**: Check parameter ranges and types
- **File Validation**: Validate file existence and permissions

### 2. Error Handling

Comprehensive error handling:

- **Graceful Degradation**: Continue operation when possible
- **Detailed Error Messages**: Provide helpful error information
- **Error Logging**: Log all errors for debugging
- **Recovery Mechanisms**: Attempt to recover from errors

### 3. Code Quality

The project follows best practices:

- **Type Hints**: Use type hints for better code quality
- **Documentation**: Comprehensive docstrings and comments
- **Code Style**: Follow PEP 8 style guidelines
- **Testing**: Comprehensive test coverage

## Deployment

### 1. Environment Setup

The system supports multiple deployment environments:

```bash
# Development environment
conda create -n sim.benchmarks python=3.9
conda activate sim.benchmarks
pip install -e .

# Production environment
pip install benchmarking_pipeline

# Docker environment
docker build -t benchmarking_pipeline .
docker run -it benchmarking_pipeline
```

### 2. Configuration Management

Configuration management supports:

- **Environment Variables**: Use environment variables for sensitive data
- **Configuration Files**: YAML configuration files
- **Default Values**: Sensible defaults for all parameters
- **Validation**: Configuration validation at startup

### 3. Resource Requirements

The system has flexible resource requirements:

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **High Performance**: 32GB+ RAM, 16+ CPU cores, GPU support
- **Scalable**: Support for distributed deployment

## Monitoring and Logging

### 1. Logging System

The logging system provides:

- **Structured Logging**: JSON-formatted log messages
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Automatic log file rotation
- **Log Aggregation**: Centralized log collection

### 2. Performance Monitoring

Performance monitoring includes:

- **Execution Time**: Track operation execution times
- **Memory Usage**: Monitor memory consumption
- **CPU Usage**: Track CPU utilization
- **I/O Operations**: Monitor file and network I/O

### 3. Error Tracking

Error tracking provides:

- **Error Aggregation**: Group similar errors
- **Error Context**: Provide context for error debugging
- **Error Reporting**: Report errors to monitoring systems
- **Error Recovery**: Track error recovery success rates

## Troubleshooting

### 1. Common Issues

#### Configuration Issues
- **Invalid YAML**: Check YAML syntax and indentation
- **Missing Parameters**: Ensure all required parameters are provided
- **Invalid Values**: Check parameter value ranges and types

#### Data Issues
- **File Not Found**: Verify file paths and permissions
- **Data Format**: Ensure data is in expected format
- **Missing Columns**: Check for required data columns

#### Model Issues
- **Training Failures**: Check data quality and model parameters
- **Memory Errors**: Reduce batch sizes or data chunks
- **Performance Issues**: Optimize hyperparameters and data preprocessing

### 2. Debugging Tools

The system provides several debugging tools:

- **Verbose Logging**: Enable detailed logging output
- **Debug Mode**: Run in debug mode for additional information
- **Performance Profiling**: Profile system performance
- **Memory Profiling**: Analyze memory usage patterns

### 3. Support Resources

Support is available through:

- **Documentation**: Comprehensive project documentation
- **Test Examples**: Working examples in test files
- **Issue Tracking**: GitHub issues for bug reports
- **Community Support**: Community forums and discussions

## Future Enhancements

### 1. Planned Features

- **Advanced Hyperparameter Optimization**: Bayesian optimization and neural architecture search
- **Distributed Training**: Support for distributed model training
- **Real-time Forecasting**: Real-time prediction capabilities
- **Model Interpretability**: Tools for understanding model decisions

### 2. Performance Improvements

- **GPU Acceleration**: Enhanced GPU support for deep learning models
- **Memory Optimization**: Better memory management for large datasets
- **Parallel Processing**: Improved parallel execution capabilities
- **Caching System**: Intelligent caching for expensive operations

### 3. Integration Enhancements

- **Cloud Integration**: Support for cloud platforms (AWS, GCP, Azure)
- **Database Integration**: Direct database connectivity
- **API Interface**: RESTful API for external access
- **Web Interface**: Web-based user interface

### 4. Model Support

- **Additional Foundation Models**: Support for more foundation models
- **Custom Model Support**: Framework for custom model implementations
- **Model Ensembles**: Support for model combination strategies
- **Transfer Learning**: Enhanced transfer learning capabilities

## Conclusion

The Time Series Forecasting Benchmarking Pipeline provides a comprehensive, extensible, and user-friendly framework for evaluating time series forecasting models. The architecture is designed to be:

- **Modular**: Easy to add new components and models
- **Scalable**: Support for both small and large-scale experiments
- **Maintainable**: Clear separation of concerns and comprehensive testing
- **Extensible**: Support for new model types and evaluation metrics

The project follows software engineering best practices and is designed to grow with the evolving field of time series forecasting. Through its intelligent design and comprehensive testing, it provides a solid foundation for research and production use.

## References

1. [Time Series Forecasting: Principles and Practice](https://otexts.com/fpp3/)
2. [Forecasting: Principles and Practice](https://otexts.com/fpp2/)
3. [Foundation Models for Time Series](https://arxiv.org/abs/2310.13525)
4. [Chronos: Time Series Forecasting with Large Language Models](https://arxiv.org/abs/2403.07815)
5. [LagLlama: Large Language Models for Time Series](https://arxiv.org/abs/2310.18607)
6. [Moirai: Foundation Models for Time Series](https://arxiv.org/abs/2311.09823)
7. [TimesFM: Foundation Models for Time Series](https://arxiv.org/abs/2310.10648)
8. [Tiny Time Mixer: Lightweight Time Series Forecasting](https://arxiv.org/abs/2310.10648)
9. [Toto: Multi-modal Foundation Models](https://arxiv.org/abs/2310.10648)
10. [ARIMA Models for Time Series Forecasting](https://otexts.com/fpp2/arima.html)
11. [LSTM for Time Series Forecasting](https://arxiv.org/abs/1506.00019)
12. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
13. [Prophet: Forecasting at Scale](https://peerj.com/preprints/3190/)
14. [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)
15. [TabPFN: A Transformer That Solves Small Tabular Classification Problems](https://arxiv.org/abs/2207.01848)
