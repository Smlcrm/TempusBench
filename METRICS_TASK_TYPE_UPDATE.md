# Metrics Task Type Validation Update

## Summary

Updated all metrics in the `benchmarking_pipeline/metrics/` module to accept an optional `task_type` kwarg and validate that the metric can be applied to that task type. Metrics now default to their compatible task type if not specified.

## Changes Made

### 1. Updated All Metric Classes

**Files Modified:**
- `benchmarking_pipeline/metrics/mae.py`
- `benchmarking_pipeline/metrics/rmse.py`
- `benchmarking_pipeline/metrics/mape.py`
- `benchmarking_pipeline/metrics/mase.py`
- `benchmarking_pipeline/metrics/crps.py`
- `benchmarking_pipeline/metrics/quantile_loss.py`
- `benchmarking_pipeline/metrics/interval_score.py`

**Changes:**
- Added optional `task_type` parameter validation in all `__call__` methods
- Each metric now validates that the provided `task_type` is compatible with the metric
- Metrics default to their compatible task type if `task_type` is not provided
- Updated docstrings to document the optional parameter and defaults

### 2. Task Type Compatibility

Based on `config_validator.py` validation rules:

**Deterministic Task Type:**
- ✅ `mae` - Mean Absolute Error
- ✅ `rmse` - Root Mean Squared Error  
- ✅ `mape` - Mean Absolute Percentage Error
- ✅ `mase` - Mean Absolute Scaled Error

**Stochastic Task Type:**
- ✅ `mae` - Mean Absolute Error
- ✅ `rmse` - Root Mean Squared Error
- ✅ `crps` - Continuous Ranked Probability Score
- ✅ `quantile_loss` - Quantile Loss (Pinball Loss)
- ✅ `interval_score` - Interval Score (Winkler Score)

### 3. Updated Evaluator

**File Modified:** `benchmarking_pipeline/pipeline/evaluator.py`

**Changes:**
- Added `task_type` extraction from config: `self.task_type = self.config.get("task", {}).get("type")`
- Updated metric calls to include `task_type` in kwargs only if available in config
- If `task_type` is not in config, metrics use their own defaults
- Maintains backward compatibility by allowing metrics to use their defaults

### 4. Updated Documentation

**File Modified:** `benchmarking_pipeline/metrics/__init__.py`

**Changes:**
- Updated package documentation to reflect optional task type parameter
- Added clear mapping of which metrics work with which task types
- Updated usage examples to show optional `task_type` parameter and defaults

## Validation Logic

Each metric now performs the following validation:

```python
task_type = kwargs.get('task_type', 'default_type')  # Default to compatible type

if task_type not in ['deterministic', 'stochastic']:
    raise ValueError(f"Invalid task_type '{task_type}'. Must be 'deterministic' or 'stochastic'.")

# Metric-specific validation
if task_type != 'expected_type':
    raise ValueError(f"{METRIC_NAME} can only be used with '{expected_type}' task_type, got '{task_type}'.")
```

**Default Task Types:**
- **MAE & RMSE**: Default to `'deterministic'` (compatible with both)
- **MAPE & MASE**: Default to `'deterministic'` (only compatible with deterministic)
- **CRPS, QuantileLoss, IntervalScore**: Default to `'stochastic'` (only compatible with stochastic)

## Backward Compatibility

- **Evaluator**: Only passes `task_type` to metrics if available in config, otherwise metrics use their defaults
- **Config Structure**: Existing configs with `task.type` field will work without changes
- **Metric Calls**: Metric calls include `task_type` from evaluator if available, otherwise use metric defaults
- **Direct Metric Usage**: Metrics can be called without `task_type` parameter and will use appropriate defaults

## Testing

Created and ran comprehensive tests to verify:
- ✅ Metrics work without task_type parameter (using defaults)
- ✅ Metrics work with explicit task_type parameter
- ✅ Metrics reject incorrect task types with appropriate error messages
- ✅ Metrics reject invalid task_type values

## Configuration Example

The existing config structure already supports this:

```yaml
task:
  type: deterministic  # or stochastic
  forecast_horizon: 10
  context_window: 10
  dataset:
    name: "univariate/*"
    normalize: false
    handle_missing: interpolate
  evaluation:
    metrics: [mae, mase, rmse, mape]  # Must match task type
    tensorboard: True
```

## Benefits

1. **Type Safety**: Prevents using incompatible metrics with wrong task types
2. **Clear Documentation**: Makes it explicit which metrics work with which task types
3. **Early Error Detection**: Fails fast with clear error messages
4. **Consistency**: Aligns with config validator rules
5. **Maintainability**: Makes the relationship between metrics and task types explicit

## Migration Notes

- **No breaking changes** for existing code that uses the Evaluator class
- **Direct metric usage** now works without `task_type` parameter (uses defaults)
- **Config files** with proper `task.type` field will work without changes
- **New configs** can optionally specify `task.type` for explicit validation

## Error Messages

The new validation provides clear, actionable error messages:

- `"Invalid task_type '{task_type}'. Must be 'deterministic' or 'stochastic'."`
- `"{METRIC_NAME} can only be used with '{expected_type}' task_type, got '{task_type}'."`
