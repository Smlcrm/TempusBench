# Metrics Usage Analysis Report

## Executive Summary

This report analyzes how metrics are used across the benchmarking pipeline codebase, identifying duplications, hard-coded implementations, and all pipeline stages that require metrics.

---

## 1. Metrics Module Overview

### Location
`benchmarking_pipeline/metrics/`

### Available Metrics
The centralized metrics module contains the following implementations:

1. **MAE** (`mae.py`) - Mean Absolute Error
2. **RMSE** (`rmse.py`) - Root Mean Squared Error
3. **MAPE** (`mape.py`) - Mean Absolute Percentage Error
4. **MASE** (`mase.py`) - Mean Absolute Scaled Error
5. **CRPS** (`crps.py`) - Continuous Ranked Probability Score
6. **QuantileLoss** (`quantile_loss.py`) - Pinball Loss for quantiles
7. **IntervalScore** (`interval_score.py`) - Winkler Score for prediction intervals

### Common Interface
All metrics implement a `__call__` method with signature:
```python
def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]
```

---

## 2. Pipeline Stages Using Metrics

### 2.1 Evaluator (`pipeline/evaluator.py`)
**Purpose:** Central evaluation component that consumes metrics from the metrics module.

**How it uses metrics:**
- Initializes metric instances in `metric_registry` dictionary (lines 34-42)
- Reads metric names from config: `evaluation.metrics` (lines 25-27)
- Default metrics: `["mae", "rmse", "mase", "mape"]`
- Applies metrics in `evaluate()` method (lines 82-136)
- Handles special cases:
  - MASE requires `y_train` parameter (lines 90-99)
  - CRPS requires `y_pred_dist_samples` kwarg (lines 100-105)
  - QuantileLoss requires `y_pred_quantiles` and `quantiles_q_values` (lines 106-114)
  - IntervalScore requires `y_pred_lower_bound` and `y_pred_upper_bound` (lines 115-123)

**Configuration dependency:**
```yaml
evaluation:
  type: deterministic  # or probabilistic
  metrics: [mae, mase, rmse, mape]
```

### 2.2 BaseModel (`models/base_model.py`)
**Purpose:** Abstract base class for all forecasting models.

**How it uses metrics:**
- Stores `training_loss` parameter (line 51) - used to select primary optimization metric
- Creates `Evaluator` instance during initialization (line 60)
- `compute_loss()` method (lines 137-190):
  - Converts inputs to numpy arrays
  - Handles shape mismatches between predictions and truth
  - Delegates to `evaluator.evaluate()` to compute all configured metrics
  - Returns dictionary of metric values

**Key insight:** Models don't calculate metrics directly; they delegate to the Evaluator.

### 2.3 HyperparameterTuner (`trainer/hyperparameter_tuning.py`)
**Purpose:** Grid search for hyperparameter optimization.

**How it uses metrics:**
- Calls `model.compute_loss()` during validation (line 125)
- Uses `model.training_loss` to select which metric to optimize (line 132)
- Stores best hyperparameters based on minimum training loss (lines 135-140)
- Final evaluation also uses `compute_loss()` (line 216)

**Important:** The `training_loss` parameter determines which metric drives hyperparameter selection.

### 2.4 ModelExecutor (`model_executor.py`)
**Purpose:** Runs individual models in isolated conda environments.

**How it uses metrics:**
- Receives final metrics from hyperparameter tuner (line 395)
- Logs metrics to TensorBoard (lines 114-137)
- Writes metrics to CSV files for persistence (lines 411-431)
- CSV structure: `metrics/{univariate|multivariate}/{dataset_name}/{model_name}/metrics.csv`

### 2.5 Logger (`pipeline/logger.py`)
**Purpose:** TensorBoard logging infrastructure.

**How it uses metrics:**
- `log_metrics()` method (lines 87-120) handles various metric types:
  - Scalar values (float, int)
  - Nested dictionaries (e.g., quantile losses)
  - Arrays (logs per-element and mean)
- `log_hparams()` method (lines 179-201) logs hyperparameters with metrics for comparison
- Sanitizes metric values (filters NaN, None)

### 2.6 Trainer (`pipeline/trainer.py`)
**Purpose:** High-level training and evaluation orchestration.

**How it uses metrics:**
- Calls `model.evaluate()` which internally uses metrics (line 70)
- Logs metrics to TensorBoard via Logger (line 74)
- Creates visualization plots of forecasts (lines 80-97)

### 2.7 BenchmarkRunner (`run_benchmark.py`)
**Purpose:** Main orchestration for running benchmarks across models and datasets.

**How it uses metrics:**
- Reads metrics from subprocess results JSON (line 483)
- Logs aggregated metrics to host TensorBoard (lines 492-493)
- Passes evaluation config to model executors (line 289)

---

## 3. Configuration-Driven Metric Selection

### Config Structure
Metrics are specified in YAML configuration files:

```yaml
model:
  arima:
    p: [1, 2]
    d: [1]
    q: [1, 2]
    training_loss: ["mae"]  # Metric used for optimization
    
evaluation:
  type: deterministic  # or probabilistic
  metrics: [mae, mase, rmse, mape]  # Metrics to compute
```

### Two-Level Metric Usage

1. **Training Loss** (`model.*.training_loss`)
   - Single metric used for hyperparameter optimization
   - Drives model selection during grid search
   - Allowed values: `['mae', 'mse', 'rmse', 'mape', 'smape', 'huber', 'log_cosh', 'poisson']`
   - Validated by ConfigValidator (lines 255-257)

2. **Evaluation Metrics** (`evaluation.metrics`)
   - Multiple metrics computed for comprehensive evaluation
   - All results logged to TensorBoard and CSV
   - Allowed values depend on evaluation type:
     - Deterministic: `['mae', 'rmse', 'mape', 'smape', 'mase']`
     - Probabilistic: `['crps', 'quantile_loss', 'interval_score', 'mae', 'rmse']`
   - Validated by ConfigValidator (lines 260-263, 543-556)

---

## 4. Duplications and Issues

### 4.1 Missing MSE Implementation
**Location:** `unify_metrics.py` (lines 90-95)

**Issue:** The script expects an 'mse' column in metrics CSV files, but there is NO MSE metric implementation in the metrics module.

**Impact:**
- `unify_metrics.py` looks for MSE but will always find None
- MSE is listed as an allowed training loss but not implemented as a metric
- Line 145: Script tries to include "MAPE" column but doesn't extract it (lines 78-99)

**Recommendation:** Add MSE metric class or remove references.

### 4.2 Hard-Coded Metric Implementations in External Notebooks

**Location:** `benchmarking_pipeline/models/anyvariate/toto/toto/evaluation/gift_eval/toto.ipynb` (lines 164-184)

**Issue:** Defines separate metric implementations:
```python
METRIC_CONFIGS = {
    "MAE": (lambda: MAE(), "MAE[0.5]"),
    "MSE": (lambda: MSE(forecast_type=0.5), "MSE[0.5]"),
    "MASE": (lambda: MASE(), "MASE[0.5]"),
    "MAPE": (lambda: MAPE(), "MAPE[0.5]"),
    "SMAPE": (lambda: SMAPE(), "sMAPE[0.5]"),
    "RMSE": (lambda: RMSE(forecast_type=0.5), "RMSE[0.5]"),
    # ... more metrics
}
```

**Problem:** 
- These appear to be from GluonTS or similar library (different API)
- NOT using the centralized metrics module
- Potential inconsistency in metric calculations
- Found in multiple TOTO/BOOM notebooks

**Locations with duplicate metric configs:**
- `benchmarking_pipeline/models/anyvariate/toto/boom/notebooks/*.ipynb` (multiple files)
- `benchmarking_pipeline/models/anyvariate/toto/boom/utils/leaderboard.py` (lines 6-13, references MASE and mean_weighted_sum_quantile_loss)

### 4.3 Hard-Coded Metric Names in Leaderboard

**Location:** `benchmarking_pipeline/models/anyvariate/toto/boom/utils/leaderboard.py`

**Issue:** Hard-coded metric column names that don't match the centralized naming:
```python
NON_ZERO_METRICS = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]
```

**Problem:**
- Assumes specific metric naming convention from external evaluation
- Not aligned with centralized metrics module
- metric renaming logic (lines 177-185) maps to different names

### 4.4 Inconsistent Metric Naming

**Current naming conventions:**
- Metrics module: lowercase (e.g., `mae`, `rmse`, `mase`)
- Config files: lowercase (e.g., `metrics: [mae, mase, rmse]`)
- CSV files: lowercase column names
- External notebooks: Mixed case and brackets (e.g., `MAE[0.5]`, `MASE[0.5]`)
- Leaderboard: Mixed case with paths (e.g., `eval_metrics/MASE[0.5]`)

**Recommendation:** Standardize on lowercase names throughout.

### 4.5 Training Loss vs Evaluation Metrics Confusion

**Location:** Multiple config files

**Issue:** `training_loss` parameter can contain metrics not in `evaluation.metrics`:
- Example: `training_loss: ["mae"]` but `evaluation.metrics: [mase, rmse, mape]`
- Training loss MUST be calculated during hyperparameter tuning but might not be in final evaluation metrics

**Current behavior:** Works because Evaluator computes all metrics in `evaluation.metrics`, and HyperparameterTuner accesses `model.training_loss` from that result.

**Risk:** If `training_loss` is not in `evaluation.metrics`, hyperparameter tuning will fail.

---

## 5. Probabilistic Metrics Usage

### Models Supporting Probabilistic Forecasts
Several foundation models generate distributional outputs:

1. **Toto** (`toto_model.py` lines 103-111):
   - Generates `num_samples` forecast samples
   - Returns mean of samples as point forecast
   - Could provide quantiles or intervals

2. **LagLlama** (`lagllama_model.py` lines 347-404):
   - Has `predict_quantiles()` method
   - Returns samples for CRPS calculation
   - Supports interval forecasts

3. **Chronos, Moirai, TimesFM, Moment:** Similar probabilistic capabilities

### Current Limitation
**Issue:** Models generate samples but the pipeline primarily uses point forecasts.

**CRPS/Interval Score Support:**
- Metrics implemented: ✅
- Models generate samples: ✅
- Pipeline passes samples to evaluator: ❌ (not implemented)

**Missing integration:** Models need to return additional outputs:
- `y_pred_dist_samples` for CRPS
- `y_pred_quantiles` + `quantiles_q_values` for QuantileLoss
- `y_pred_lower_bound` + `y_pred_upper_bound` for IntervalScore

---

## 6. Metric Persistence and Aggregation

### CSV Output
**Location:** `model_executor.py` (lines 411-431)

**Structure:**
```
metrics/
  ├── univariate/
  │   └── {dataset_name}/
  │       └── {model_name}/
  │           └── metrics.csv
  └── multivariate/
      └── {dataset_name}/
          └── {model_name}/
              └── metrics.csv
```

**Content:** One row with all metric columns (mae, rmse, mase, mape, etc.)

### Metric Unification
**Location:** `unify_metrics.py`

**Purpose:** Aggregates all metrics CSVs into single file.

**Issues:**
1. Expects "MSE" column that doesn't exist (lines 81-95)
2. Line 145: References "MAPE" in column order but doesn't extract it in lines 78-99
3. Only extracts MASE, MSE, RMSE despite more metrics being available

**Recommendation:** Update to extract all available metrics, not just hard-coded subset.

---

## 7. Visualization and Logging

### TensorBoard Integration

**Metrics logged to TensorBoard:**
1. **During hyperparameter search** (via `model_executor.py` lines 88-112):
   - Best validation score
   - Individual hyperparameter values

2. **Final evaluation** (via `model_executor.py` lines 114-137):
   - All computed metrics as scalars
   - Model configuration as text

3. **Host-level aggregation** (via `run_benchmark.py` lines 485-515):
   - Hyperparameters and metrics for HParams comparison
   - Forecast plots as images

**Visualization:**
- `visualizer.py` creates forecast plots (lines 27-95)
- Residual analysis plots (lines 97-143)
- Not directly tied to metrics, but uses predictions that metrics evaluate

---

## 8. Recommendations

### 8.1 High Priority

1. **Add MSE Metric**
   - Create `benchmarking_pipeline/metrics/mse.py`
   - Add to Evaluator registry
   - Or remove all MSE references

2. **Fix unify_metrics.py**
   - Line 145: Add MAPE extraction logic
   - Make metric extraction dynamic instead of hard-coded
   - Extract all available metrics from CSV, not just MASE/MSE/RMSE

3. **Validate training_loss in evaluation.metrics**
   - Add validation that `training_loss` metric is included in `evaluation.metrics`
   - Or automatically add training_loss to evaluation.metrics if missing

4. **Standardize Metric Naming**
   - Enforce lowercase names throughout
   - Remove hard-coded metric lists where possible
   - Use metric_registry.keys() for available metrics

### 8.2 Medium Priority

5. **Integrate Probabilistic Metrics**
   - Extend model `predict()` to optionally return samples/quantiles
   - Pass probabilistic outputs to evaluator when available
   - Enable CRPS, interval_score, quantile_loss evaluation for foundation models

6. **Consolidate External Metric Implementations**
   - Review TOTO/BOOM notebooks for alignment with centralized metrics
   - Document any intentional differences (e.g., GluonTS compatibility)
   - Consider extracting shared metric code

7. **Improve ConfigValidator**
   - Cross-validate training_loss against evaluation.metrics
   - Warn if probabilistic metrics requested but model doesn't support them

### 8.3 Low Priority

8. **Add Metric Documentation**
   - Expand docstrings in metric classes
   - Document when each metric is appropriate
   - Add examples to metrics/__init__.py

9. **Create Metric Test Suite**
   - Unit tests for each metric implementation
   - Test edge cases (NaN, inf, zero variance)
   - Validate against known reference implementations

---

## 9. Summary Table

| Pipeline Stage | Uses Metrics? | How? | Configuration |
|---|---|---|---|
| Evaluator | ✅ Direct | Instantiates and calls metric classes | `evaluation.metrics` |
| BaseModel | ✅ Indirect | Delegates to Evaluator | `model.*.training_loss` + `evaluation.metrics` |
| HyperparameterTuner | ✅ Indirect | Calls model.compute_loss() | Uses `model.training_loss` |
| ModelExecutor | ✅ Indirect | Receives computed metrics, logs them | None (consumer) |
| Logger | ✅ Indirect | Formats and logs metric values | None (consumer) |
| Trainer | ✅ Indirect | Calls model.evaluate() | None (orchestrator) |
| BenchmarkRunner | ✅ Indirect | Aggregates metrics across runs | None (orchestrator) |
| unify_metrics.py | ⚠️ Partial | Reads CSV files with hard-coded columns | None (expects fixed columns) |
| Visualizer | ❌ | Creates plots, no metric calculations | None |

---

## 10. Key Findings

### ✅ Good Practices
1. **Centralized metrics module** with consistent interface
2. **Configuration-driven** metric selection
3. **Separation of concerns**: models delegate to Evaluator
4. **Flexible metric registry** supporting easy additions
5. **Comprehensive logging** to TensorBoard and CSV

### ⚠️ Issues Found
1. **Missing MSE implementation** despite references
2. **Hard-coded metric implementations** in external notebooks
3. **Inconsistent metric naming** across components
4. **Limited probabilistic metric integration** in pipeline
5. **Hard-coded metric extraction** in unify_metrics.py
6. **No validation** that training_loss is in evaluation.metrics
7. **MAPE extraction missing** in unify_metrics.py (line 145 bug)

### 📊 Metric Coverage
- **Implemented and integrated:** MAE, RMSE, MAPE, MASE
- **Implemented but underutilized:** CRPS, QuantileLoss, IntervalScore
- **Referenced but missing:** MSE
- **External duplicates:** Multiple metrics in TOTO/BOOM evaluation code

---

## Conclusion

The benchmarking pipeline has a well-designed centralized metrics system, but there are several areas where hard-coded values, missing implementations, and incomplete integrations create maintenance risks. The most critical issues are:

1. The missing MSE metric
2. The inconsistent metric extraction in unify_metrics.py
3. The underutilized probabilistic metrics capabilities

Addressing these issues will improve consistency, maintainability, and enable fuller utilization of foundation model capabilities.

