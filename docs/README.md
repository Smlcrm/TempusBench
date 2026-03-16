# TempusBench Documentation

Central documentation for the TempusBench time series forecasting benchmarking pipeline.

## Documentation Index

### Models

| Document | Description |
|----------|-------------|
| [TempusBench Models](models/tempusbench_models.md) | All forecasting models in TempusBench (statistical, ML, foundation) |
| [StatsForecast Models](models/statsforecast_models.md) | Reference: models available in Nixtla StatsForecast |
| [Chronax Models](models/chronax_models.md) | Reference: models available in Chronax (JAX) |
| [Model Comparison](models/model_comparison.md) | TempusBench ∪ Chronax vs StatsForecast |

### Covariates

| Document | Description |
|----------|-------------|
| [Foundation Models Covariate Support](covariates/foundation_models_covariate_support.md) | Covariate support reference for all foundation models |
| [Covariate Testing Guide](covariates/covariate_testing_guide.md) | Manual and automated testing strategies for covariates |
| [Chronos-2 Covariate Flow](covariates/chronos2_covariate_flow.md) | Chronos-2 specific covariate data flow |

### Development

| Document | Description |
|----------|-------------|
| [Config Coverage Summary](development/CONFIG_COVERAGE_SUMMARY.md) | Test coverage for config module |
| [TODO](TODO.md) | Pipeline improvements and future work |

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Model implementations**: `tempus_bench/models/`
- **Benchmark config**: `tempus_bench/config/benchmark.yaml`
- **Covariate tests**: `tests/unit/test_covariate_support.py`
