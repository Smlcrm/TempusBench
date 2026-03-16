# Covariate Testing Guide

Manual testing strategies to verify covariate support across TempusBench models.

**See also:** [Foundation Models Covariate Support](foundation_models_covariate_support.md) — detailed reference for each model's covariate type, parameters, and translation of `x_context`/`x_target`.

## Automated Unit Tests

Run covariate tests for all foundation models:

```bash
pytest tests/unit/test_covariate_support.py -v
```

Tests verify:
- `test_foundation_model_accepts_covariates`: train/predict with `x_context` succeed and produce correct output shape
- `test_foundation_model_runs_without_covariates`: train/predict with `x_context=None` (backward compat)

Models requiring external deps (accelerate, Chronarium, etc.) are skipped when unavailable.

## Benchmark Run with Covariates

Run all covariate benchmarks using the unified config:

```bash
python scripts/run_covariate_benchmarks.py
```

To run specific models only:

```bash
python scripts/run_covariate_benchmarks.py --models chronos_tiny,tabpfn,chronos2
```

Logs: `logs/covariate_benchmarks.log` and `logs/covariate_benchmarks_summary.log`.

For manual testing with a minimal config, use `tempus_bench/config/test_covariate.yaml` or `tests/test_covariate_foundation.yaml`.

## 1. Sensitivity Test (Primary Check)

**Principle:** If covariates are used, changing `x_context` while keeping `y_context` fixed should change the forecasts.

**Steps:**
1. Create fixed `y_context` (e.g. `(100, 1)` or `(100, 2)`).
2. Run `predict` with `x_context=None`.
3. Run `predict` with `x_context` = some values (e.g. random or `np.arange(...).reshape(-1, 1)`).
4. Run `predict` with a *different* `x_context` (e.g. shifted or scaled).

**Interpretation:**
- **Deterministic models** (MOMENT, TabPFN, Time-MoE, Sundial): Forecasts should differ between runs 2 vs 3 and 3 vs 4.
- **Stochastic models:** Set `np.random.seed(...)` before each run (if the model respects it) and compare mean forecasts, or run multiple times and compare distributions.

---

## 2. Informative Covariate Test

**Principle:** If covariates are used, forecasts should improve when the covariate is informative.

**Setup:**
```python
# y is a linear function of x plus noise
n = 200
x = np.linspace(0, 10, n).reshape(-1, 1)
y = 2 * x.squeeze() + np.random.randn(n) * 0.5  # y ≈ 2*x
y_context = y[-100:].reshape(-1, 1)
x_context = x[-100:]
```

**Steps:**
1. Forecast with `x_context` provided.
2. Forecast with `x_context=None` (or zeros).
3. Compare error vs held-out true future.

**Interpretation:** If covariates are used, forecasts with `x_context` should be closer to the true future than without.

---

## 3. Shape & Runability Test

**Principle:** Ensure the model runs without error when covariates are passed.

**Minimal call pattern:**
```python
# Minimal shapes matching base_model validation
context_len = 64
forecast_horizon = 12
num_targets = 1
num_covariates = 2

y_context = np.random.randn(context_len, num_targets).astype(np.float64)
x_context = np.random.randn(context_len, num_covariates).astype(np.float64)
timestamps_context = np.arange(context_len, dtype=np.float64)
timestamps_target = np.arange(forecast_horizon, dtype=np.float64)

model.train(y_context, y_target, timestamps_context, timestamps_target, 
            x_context=x_context, x_target=None, num_samples=10, freq="D")
pred = model.predict(y_context, timestamps_context, timestamps_target,
                     x_context=x_context, x_target=None, num_samples=10, freq="D")
```

**Check:** No exceptions; `pred` has expected shape (e.g. `(num_samples, forecast_horizon, num_targets)` for stochastic).

---

## 4. Model-Specific Notes

| Model | Notes |
|-------|-------|
| **Chronos, Chronos-Bolt** | Stochastic; fix seed if possible; compare mean forecasts across runs. |
| **MOMENT** | Deterministic; `train` must receive `x_context` so `n_channels` includes covariates. |
| **TabPFN** | Deterministic; covariates are extra features; sensitivity test is straightforward. |
| **Kairos, PatchTST-FM, TimesFM 2.5** | Stochastic; compare quantile/mean forecasts. |
| **Time-MoE, Sundial** | Univariate per target; iterate over targets; test with `num_targets=1` first. |
| **Granite FlowState** | Requires `freq` in kwargs; `scale_factor` depends on `freq`. |
| **Lag-Llama** | Requires `freq`; past-only via iteration over variates. |
| **LAFN** | Fixed config from Chronarium; covariates not wired yet. |

---

## 5. Quick Script Template

```python
import numpy as np

def test_covariate_sensitivity(model_class, params, settings, seed=42):
    np.random.seed(seed)
    ctx_len, horizon, n_targets, n_cov = 64, 12, 1, 2
    
    y_context = np.random.randn(ctx_len, n_targets).astype(np.float64)
    y_target = np.random.randn(horizon, n_targets).astype(np.float64)
    ts_ctx = np.arange(ctx_len, dtype=np.float64)
    ts_tgt = np.arange(horizon, dtype=np.float64)
    
    kwargs = dict(num_samples=10, freq="D")
    
    model = model_class(params=params, settings=settings)
    model.train(y_context, y_target, ts_ctx, ts_tgt, x_context=None, x_target=None, **kwargs)
    
    # Run 1: no covariates
    pred_none = model.predict(y_context, ts_ctx, ts_tgt, x_context=None, x_target=None, **kwargs)
    
    # Run 2: with covariates A
    x_a = np.random.randn(ctx_len, n_cov).astype(np.float64)
    pred_a = model.predict(y_context, ts_ctx, ts_tgt, x_context=x_a, x_target=None, **kwargs)
    
    # Run 3: with different covariates B
    x_b = np.random.randn(ctx_len, n_cov).astype(np.float64) * 10  # different scale
    pred_b = model.predict(y_context, ts_ctx, ts_tgt, x_context=x_b, x_target=None, **kwargs)
    
    # For deterministic: pred_none != pred_a and pred_a != pred_b
    # For stochastic: compare mean of samples
    mean_none = np.mean(pred_none, axis=0)
    mean_a = np.mean(pred_a, axis=0)
    mean_b = np.mean(pred_b, axis=0)
    
    diff_a = np.abs(mean_none - mean_a).max()
    diff_b = np.abs(mean_a - mean_b).max()
    
    print(f"Max diff (none vs A): {diff_a:.6f}")
    print(f"Max diff (A vs B): {diff_b:.6f}")
    return diff_a > 1e-6 or diff_b > 1e-6  # True if covariates seem to affect forecasts
```

---

## 6. Using the Benchmark Pipeline

To test with real datasets that have covariates:

1. Use a dataset with a `covariate` column (see `model_executor.py` lines 371–382).
2. Run the benchmark for a single model and dataset.
3. Compare metrics when the dataset has covariates vs when they are removed or zeroed.

---

## Summary

The most direct check is the **sensitivity test**: same `y_context`, different `x_context`, and verify forecasts change. For deterministic models this is straightforward; for stochastic ones, compare mean forecasts or distributions.
