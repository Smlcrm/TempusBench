# Chronos-2 Covariate Data Flow

## What the model receives

For each rolling window, Chronos-2 receives:

| Input | Shape | Description |
|-------|-------|-------------|
| `target` | (n_targets, history_length) | Target series for context+train period |
| `past_covariates` | dict of 1D arrays, each (history_length,) | Covariate values over the context period |
| `future_covariates` | dict of 1D arrays, each (prediction_length,) | Covariate values over the forecast horizon |

**Chronos-2 API requirement:** All keys in `future_covariates` must be a subset of keys in `past_covariates`.

## What the model returns

- **Raw output:** Quantiles `(n_variates, n_quantiles, prediction_length)` from Chronos-2
- **Converted to samples:** `(num_samples, forecast_horizon, num_targets)` for metric computation
- **Scale:** Same as targets (normalized if dataset uses `normalize: true`)

## Covariate sensitivity check

When covariates are present, the benchmark runs an ablation on the first window:

1. Predict **with** covariates (normal run)
2. Predict **without** covariates (`use_covariates=False`)

If `max_diff < 1e-6`, a warning is printed: covariates may not be used.  
If predictions differ, stderr shows: `[Covariate check] OK: Predictions differ with vs without covariates (max_diff=X)`.

## Verification

Run the benchmark and check stderr:

```bash
python -m tempus_bench.run_benchmark --config tempus_bench/config/test_covariate.yaml 2>&1 | grep "Covariate check"
```

Expected: `[Covariate check] OK: Predictions differ with vs without covariates (max_diff=...)`
