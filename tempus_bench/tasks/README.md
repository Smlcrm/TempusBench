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
  covariate/<task_id>/
    task.yaml
    metadata.json
    <csv>   # copy of sibling multivariate data (covariate_*.csv)
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
```

### Multivariate (joint forecasting)

```yaml
task:
  context_window: 32
  forecast_horizon: 8
  handle_missing: interpolate
  normalization_method: standard
  file_name: multivariate_<suffix>.csv
  target_variable_names: [all variates]
```

### Covariate (single target + covariates)

```yaml
task:
  context_window: 32
  forecast_horizon: 8
  handle_missing: interpolate
  normalization_method: standard
  file_name: covariate_<suffix>.csv
  target_variable_name: PM25
  covariate_variable_names: [remaining variates used as covariates]
```

Naming: `multivariate_transport_monthly_foo` pairs with
`covariate/covariate_transport_monthly_foo` and `covariate_transport_monthly_foo.csv`.

## Discovery

Wildcard discovery (`*`) registers every folder under `univariate/`, `multivariate/`,
and `covariate/` (30 tasks total: 10 + 10 + 10).

Examples:

- `multivariate/multivariate_transport_monthly_airline_baggage_complaints` — joint multivariate
- `covariate/covariate_transport_monthly_airline_baggage_complaints` — single target + covariates

## Regenerating catalog artifacts

```bash
python tempus_bench/scripts/generate_tasks_metadata.py
python tempus_bench/scripts/generate_tasks_task_yaml.py
```

One-off migration from the legacy `__covariate` fan-out layout:

```bash
python tempus_bench/scripts/migrate_covariate_folders.py
```
