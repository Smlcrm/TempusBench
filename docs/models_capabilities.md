# Model capabilities (`settings.yaml`)

Each model under `tempus_bench/models/<name>/` must declare execution settings in `settings.yaml`, including a **`capabilities`** block. If it is missing or invalid, `ConfigManager` raises a clear error at benchmark start. Parsing and covariate helpers live in **`tempus_bench/utils/model_settings.py`** (with `ConfigManager` in `config_manager.py` orchestrating loads for full benchmark configs).

## Schema

```yaml
capabilities:
  covariates: past_future | past_only | future_only | none
  univariate: true | false    # may run on tasks under tasks/univariate/
  multivariate: true | false  # may run on tasks under tasks/multivariate/
```

- **`covariates`**: How past/future covariate tensors are passed into the model (`none` ⇒ not declared for **covariate** task datasets under `tasks/covariate/`).
- **`univariate` / `multivariate`**: Declare whether the model is allowed on that task family. The pipeline checks every (model × task) pair in the active benchmark config and fails fast if incompatible.

## Adding a new model

1. Add `tempus_bench/models/<name>/settings.yaml` with `capabilities` set truthfully.
2. Do not add a central registry file; discovery uses the models directory and `get_available_models()`.

Downstream UIs can generate a JSON snapshot of tasks and models from **`tempus_bench.utils.orchestrator_metadata`** (`get_tasks`, `get_models`); that file should be treated as generated from `settings.yaml`, not edited by hand.
