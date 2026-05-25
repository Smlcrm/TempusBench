# tasks — benchmark catalog

Runtime data loading uses this tree (`tempus_bench/tasks/`).

## Layout

```
tasks/
  univariate/<task_id>/
    task.yaml
    metadata.json
    <csv>
  multivariate/<task_id>/
    task.yaml
    metadata.json
    <csv>
  datasets_catalog.csv
  all_tasks.yaml
```

## CSV schema

Each task CSV has one row per variate:

| column | description |
|--------|-------------|
| `variable_name` | variate identifier (matches yaml lists) |
| `variable_unit` | unit string for the series |
| `timestamps` | JSON array of ISO 8601 UTC strings |
| `values` | JSON array of numbers (or null for missing) |

There is **no** `variable_type` column. Targets and covariates are selected from
`task.yaml` at load time.

## Flat `task.yaml` schema

### Univariate

```yaml
task:
  context_window: 512
  forecast_horizon: 64
  handle_missing: interpolate
  normalization_method: standard   # standard | none
  file_name: <csv>
  target_variable_names: [name]
  covariate_variable_names: []
```

### Multivariate (joint + covariate modes)

```yaml
task:
  context_window: 32
  forecast_horizon: 8
  handle_missing: interpolate
  normalization_method: standard
  file_name: <csv>
  multivariate_target_variable_names: [all variates]
  covariate_target_variable_name: <primary target for covariate mode>
  covariate_variable_names: [remaining variates used as covariates]
```

## Logical task fan-out (`__covariate`)

Each **multivariate folder** becomes two logical tasks at discovery:

| logical id | `task_mode` | targets | covariates |
|------------|-------------|---------|------------|
| `<folder>` | `multivariate` | all variates | `[]` |
| `<folder>__covariate` | `covariate` | `covariate_target_variable_name` | `covariate_variable_names` |

Both modes read the **same on-disk CSV**. The `__covariate` suffix appears only in
logical task names, pickle filenames, and external result sink ids — not in folder
paths.

Examples:

- `multivariate/multivariate_transport_monthly_airline_baggage_complaints` — joint multivariate
- `multivariate/multivariate_transport_monthly_airline_baggage_complaints__covariate` — single target + covariates

Wildcard discovery (`*`, `multivariate/*`) emits both logical ids for each multivariate folder.

## Regenerating catalog artifacts

```bash
python tempus_bench/scripts/generate_tasks_metadata.py
python tempus_bench/scripts/generate_tasks_task_yaml.py
```
