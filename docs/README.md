# Tempus Bench — library documentation

Docs in this tree describe the **`tempus_bench`** package (configs, tasks, models, testing). They are intended to ship with an open-source / PyPI distribution.

## Models

| Document | Description |
| -------- | ----------- |
| [Tempus Bench Models](models/tempusbench_models.md) | Models shipped in this repo |
| [StatsForecast Models](models/statsforecast_models.md) | Reference: Nixtla StatsForecast |
| [Chronax Models](models/chronax_models.md) | Reference: Chronax (JAX) |
| [Model Comparison](models/model_comparison.md) | Tempus Bench ∪ Chronax vs StatsForecast |
| [Model capabilities (`settings.yaml`)](models_capabilities.md) | `capabilities` schema and validation |

## Covariates

| Document | Description |
| -------- | ----------- |
| [Foundation Models Covariate Support](covariates/foundation_models_covariate_support.md) | Per-model covariate modes |
| [Covariate Testing Guide](covariates/covariate_testing_guide.md) | Manual and automated testing |
| [Chronos-2 Covariate Flow](covariates/chronos2_covariate_flow.md) | Chronos-2 data flow |

## Development

| Document | Description |
| -------- | ----------- |
| [Config coverage](development/CONFIG_COVERAGE_SUMMARY.md) | Config module tests |
| [TODO](TODO.md) | Pipeline improvements |

## Quick links

- [Library README](../README.md) (install, CLI, structure)
- Cloud UI, worker, and GCP deploy tooling: separate repository **`inference-tempusbench-cloud`** (clone TempusBench into `tempusbench_open/` beside that repo’s `deployment/` and `tempusbench_cloud/` trees)
- Model code: `tempus_bench/models/`
- Default benchmark config: `tempus_bench/config/benchmark.yaml`
- Covariate unit tests: `tests/unit/test_covariate_support.py`
