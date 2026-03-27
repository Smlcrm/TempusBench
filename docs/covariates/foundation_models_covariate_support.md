# Foundation Models: Covariate Support Reference

This document describes covariate support for all foundation models in Tempus Bench: what type of support each model has, which model parameters are used, and how `x_context` and `x_target` are translated into model-specific inputs.

**Pipeline configuration:** Covariate wiring is declared per model in `tempus_bench/models/<name>/settings.yaml` under `capabilities.covariates` (see [models_capabilities.md](../models_capabilities.md)). `tempus_bench/utils/model_settings.py` loads and validates that file (capabilities block and covariate-mode helpers). Unit tests in `tests/unit/test_covariate_support.py` verify covariate acceptance and rejection behavior.

## Input Conventions

Tempus Bench uses a unified covariate interface:

| Parameter   | Shape                                 | Description                                               |
| ----------- | ------------------------------------- | --------------------------------------------------------- |
| `x_context` | `(num_steps_context, num_covariates)` | Past covariate values aligned with the context window     |
| `x_target`  | `(num_steps_target, num_covariates)`  | Future covariate values aligned with the forecast horizon |

The pipeline passes these based on model category:

- **Past-only models**: Receive `x_context` only; `x_target` is `None` when dataset has covariates
- **Both models**: Receive both `x_context` and `x_target` when the dataset has covariates
- **No-covariate models**: Receive `x_context=None`, `x_target=None` regardless of dataset

---

## Translation Strategy Types

Tempus Bench uses three standardized translation strategies:

| Strategy                    | Description                                                                                                                                                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Channel concatenation**   | Covariates appended as extra channels to the target series. Input shape `(n_channels, context_length)` with `n_channels = num_targets + num_covariates`. Model forecasts all channels; only target-channel outputs are kept. |
| **Native**                  | Model has a built-in covariate API. Covariates passed via dedicated parameters (e.g. `past_covariates`, `feat_dynamic_real`, sktime `X`), not as fake targets.                                                               |
| **Iteration over variates** | Model is univariate. Targets and covariates stacked; one univariate call per variate. Only first `num_targets` forecasts kept; covariate forecasts discarded.                                                                |

---

## Models with Past-Only Covariate Support

These models use **only** `x_context`. Future covariates (`x_target`) are not supported.

### Chronos (tiny, mini, small, base, large)

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)` — covariates appended as extra channels
- Input shape: `(num_targets + num_covariates, context_length)` transposed for Chronos
- Model API: `ChronosPipeline.predict(context=context_tensor, prediction_length=forecast_horizon)`

**Parameters:**

- `context`: 2D tensor `(n_channels, context_length)` where channels = targets + covariates
- Output channels beyond `num_targets` are discarded; only target-channel forecasts are kept

**Notes:** Chronos has no native covariate API. Covariates are treated as additional target channels; the model forecasts them but we discard those forecasts.

---

### Chronos-Bolt (tiny, mini, small, base)

**Type:** Past-only (channel concatenation)

**Translation:**

- Same as Chronos: `y_input = np.concatenate([y_context, x_context], axis=1)`
- Input: `(n_channels, context_length)` with channels = targets + covariates

**Parameters:**

- `inputs`: Tensor from `y_input.T` (transposed)
- `prediction_length`: Forecast horizon

---

### Kairos (10m, 23m, 50m)

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)`
- Trimmed/padded to `context_length` before passing to model

**Parameters:**

- `past_target`: Per-channel 1D tensors `(1, context_length)` — iterates over channels
- Covariate channels are forecast along with targets; only first `num_targets` channel forecasts are kept

---

### MOMENT (small, base, large)

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)`
- `n_channels = num_targets + num_covariates` passed to `MOMENTPipeline.from_pretrained(model_kwargs={"n_channels": n_channels, ...})`

**Parameters:**

- `model_kwargs.n_channels`: Set at train time based on presence of covariates
- `x_enc`: Tensor `(batch, n_channels, context_window)` — `y_input` scaled, trimmed, transposed
- `input_mask`: Ones mask of same length
- Output: `forecast[:, :num_targets]` — covariate channel forecasts discarded

---

### Time-MoE (50m, 200m)

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)`
- Uses last `max_context = context_length - forecast_horizon` steps

**Parameters:**

- `generate(normed_seqs, max_new_tokens=forecast_horizon)` — autoregressive generation
- Input: per-channel normalized series; covariates as extra channels
- Only first `num_targets` channel outputs are kept

---

### PatchTST-FM / PatchTST Granite

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)`
- Converted to list of 1D tensors (one per channel) for channel-independent processing

**Parameters:**

- `inputs_list`: `[torch.tensor(y_input[:, i]) for i in range(n_channels)]`
- `prediction_length`: Forecast horizon
- `return_loss=False`
- Output `quantile_predictions`: `(n_channels, quantiles, forecast_len)` — keep first `num_targets` channels

**PatchTST Granite:** Inherits PatchTST-FM logic; overrides `train()` to build `PatchTSTFMConfig` with `n_head` from `num_attention_heads` for Granite backbone compatibility.

---

### Sundial

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_ctx, x_ctx], axis=1)` with optional `lookback_length` truncation
- Each channel (targets + covariates) processed independently with z-score normalization

**Parameters:**

- `model.generate(seqs, max_new_tokens=forecast_horizon, num_samples=num_samples)` — diffusion-based generation
- `seqs`: `(1, context_length)` per channel, normalized
- Output: `(1, num_samples, context_length + forecast_horizon)` — last `forecast_horizon` steps kept; only first `num_targets` channels retained

---

### TimesFM 2.5 (timesfm2)

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)`
- Trimmed/padded to `context_length`

**Parameters:**

- `past_values`: List of 1D tensors, one per channel (targets + covariates)
- `TimesFm2_5ModelForPrediction(past_values=past_values, return_dict=True)`
- Output: `full_predictions` `(n_channels, horizon, quantiles)` — keep first `num_targets` channels

---

### TimesFM 500M (timesfm_500m)

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)` — covariates appended as extra channels
- Input: list of 1D series, one per channel (targets + covariates)
- Output: `forecast` returns list of `(horizon,)` per channel; keep first `num_targets` channels

**Parameters:**

- `inputs = [y_input[:, i].tolist() for i in range(n_channels)]` — targets + covariates
- `timesfm.TimesFm.forecast(inputs, freq=[0] * n_channels)` — no native covariate API
- Covariate channel forecasts discarded; only first `num_targets` channel outputs kept

---

### TabPFN

**Type:** Past-only (native)

**Translation:**

- TabPFN uses a tabular regressor over time features. Covariates are concatenated as extra columns:
  - `X_hist = make_time_features(len(y_hist)).values` — base time features
  - `X_hist = np.concatenate([X_hist, x_hist], axis=1)` when `x_context` is provided
  - For future steps: `X_future = make_time_features(...).values[-step:]`; covariates padded with `last_cov = x_hist[-1:]` (last observed value)

**Parameters:**

- `TabPFNRegressor.fit(X_hist, y_hist)` — X includes time features + covariates
- `regressor.predict(X_future)` — X_future has time features + repeated last covariate row
- Rollout: autoregressive; each step extends `y_hist` with predicted value

**Notes:** TabPFN natively accepts exogenous features in its tabular API (`fit(X, y)`, `predict(X)`). Covariates are passed as columns in X alongside time features.

---

### Lag-Llama

**Type:** Past-only (iteration over variates)

**Translation:**

- Lag-Llama uses `input_size=1` (univariate) only. Same pattern as multivariate targets:
  - `variates = np.concatenate([y_context, x_context], axis=1)` when `x_context` is provided
  - One univariate call per variate (M targets + N covariates)
  - Only first `num_targets` predictions kept; covariate forecasts discarded

**Parameters:**

- `past_target`: Single variate `(context_length, 1)` per call
- `input_size=1` always; predictor created once for univariate
- Output: `preds[:num_targets]` — covariate predictions discarded

**Notes:** Covariates do not condition the target forecast; each variate is forecast independently. Compatible with the original multivariate-target structure (iterate over columns). No multivariate logic in the module.

---

### Granite FlowState

**Type:** Past-only (iteration over variates)

**Translation:**

- FlowState accepts only one channel per call. Same iteration pattern as Lag-Llama:
  - `variates = np.concatenate([y_context, x_context], axis=1)` when `x_context` is provided
  - One univariate call per variate (targets + covariates)
  - Only first `num_targets` predictions kept; covariate forecasts discarded

**Parameters:**

- `context_tensor`: `(context_length, batch_size, 1)` — one channel per call
- `scale_factor`, `prediction_length`, `batch_first=False`
- Output: `preds[:num_targets]` — covariate predictions discarded

**Notes:** Covariates do not condition the target forecast; each variate is forecast independently. Provides pipeline compatibility for covariate datasets.

---

### Moirai 2.0 (moirai2)

**Type:** Past-only (native)

**Translation:**

- Moirai2 supports `past_feat_dynamic_real` and `feat_dynamic_real`, but `feat_dynamic_real` (future covariates) has known bugs
- Only `past_feat_dynamic_real` is used: `past_feat_dynamic_real = torch.tensor(x_context).unsqueeze(0)`
- `x_target` is ignored

**Parameters:**

- `Moirai2Forecast(..., past_feat_dynamic_real_dim=past_feat_dim, feat_dynamic_real_dim=0)`
- `past_feat_dynamic_real`: `(batch, context_length, num_covariates)`
- `past_observed_feat_dynamic_real`: Ones mask
- `feat_dynamic_real=None`, `observed_feat_dynamic_real=None`

**Notes:** Uses GluonTS/uni2ts native API for past covariates. Future covariates (`feat_dynamic_real`) are not used due to known bugs.

---

### TiRex / TiRex 1.1 GIFT-Eval

**Type:** Past-only (channel concatenation)

**Translation:**

- `y_input = np.concatenate([y_context, x_context], axis=1)` — covariates appended as extra channels
- TiRex `forecast(context=..., prediction_length=...)` accepts multi-channel input `(n_channels, context_length)`
- Input shape: `(num_targets + num_covariates, context_length)` transposed from `y_input.T`
- Output: `quantiles[:num_targets]` — covariate channel forecasts discarded

**Parameters:**

- `context`: 2D tensor `(n_channels, context_length)` where channels = targets + covariates
- `prediction_length`: Forecast horizon
- Quantiles expanded to pseudo-samples; only first `num_targets` channel forecasts kept

---

## Models with Both Past and Future Covariate Support

These models use both `x_context` and `x_target`.

### Chronos-2

**Type:** Both (native)

**Translation:**

- `past_covariates`: Dict `{f"cov_{i}": x_context_trimmed[:, i]}` for each covariate column
- `future_covariates`: Dict `{f"cov_{i}": x_target[:forecast_horizon, i]}` for each covariate column
- Chronos-2 requires future covariate keys to be a subset of past covariate keys (same names)

**Parameters:**

- `input_dict`: `{"target": target, "past_covariates": {...}, "future_covariates": {...}}`
- `target`: `(n_variates, history_length)` from `y_context.T`
- `Chronos2Pipeline.predict([input_dict], prediction_length=forecast_horizon, batch_size=1)`

**Output:** Quantiles converted to pseudo-samples for stochastic metrics.

---

### TimesFM 200M (timesfm_200m)

**Type:** Both (native)

**Translation:**

- TimesFM requires full (context + horizon) covariate coverage
- **Both:** `full_covariates = np.concatenate([x_context, x_target[:forecast_horizon]], axis=0)`
- **Past-only:** Pad future with last row: `last_row = np.tile(x_context[-1:], (forecast_horizon, 1))`
- **Future-only:** Pad past with zeros: `past_pad = np.zeros((ctx_len, num_covariates))`

**Parameters:**

- `forecast_with_covariates(inputs=inputs, dynamic_numerical_covariates=dynamic_numerical_covariates, freq=[0]*num_targets)`
- `inputs`: `[y_context[:, i].tolist() for i in range(num_targets)]` — target series only
- `dynamic_numerical_covariates`: `{f"cov_{i}": [full_covariates[:, i].tolist()] * num_targets}` — each covariate replicated per target

---

### Tiny Time Mixer (r1, r2, r2_1)

**Type:** Both (native)

**Translation:**

- sktime `TinyTimeMixerForecaster.fit(df, X=X_fit, fh=fh)` and `predict(X=X_pred)`
- **Both:** `X_fit = pd.DataFrame(x_context, ...)`, `X_pred = pd.DataFrame(x_target[:forecast_horizon], ...)`
- **Past-only:** `X_fit` from `x_context`; `X_pred` = last row of `x_context` tiled over forecast horizon
- **Future-only:** `X_fit` = zeros `(context_len, num_covariates)`; `X_pred` from `x_target`

**Parameters:**

- `X_fit`: Exogenous data for fit, aligned with `timestamps_context`
- `X_pred`: Exogenous data for predict, aligned with `timestamps_target`
- `fh`: Forecast horizon as `list(range(1, forecast_horizon + 1))`

---

## Models with No Covariate Support

These models are in `NO_COVARIATE_MODELS`. The pipeline passes `x_context=None`, `x_target=None` when the dataset has covariates.

| Model | Reason           |
| ----- | ---------------- |
| LAFN  | No covariate API |

---

## Summary Table

| Model                      | Support   | Translation Strategy    | Model Parameters                                           |
| -------------------------- | --------- | ----------------------- | ---------------------------------------------------------- |
| Chronos, Chronos-Bolt      | Past-only | Channel concatenation   | `context` = targets + covariates                           |
| Kairos                     | Past-only | Channel concatenation   | `past_target` per channel                                  |
| MOMENT                     | Past-only | Channel concatenation   | `n_channels`, `x_enc`                                      |
| Time-MoE                   | Past-only | Channel concatenation   | `generate()` over channels                                 |
| PatchTST-FM, Granite       | Past-only | Channel concatenation   | `inputs_list`                                              |
| Sundial                    | Past-only | Channel concatenation   | `generate()` per channel                                   |
| TimesFM 2.5                | Past-only | Channel concatenation   | `past_values`                                              |
| TimesFM 500M               | Past-only | Channel concatenation   | `forecast(inputs)`; discard covariate channels             |
| TabPFN                     | Past-only | Native                  | `X_hist`, `X_future` with covariates                       |
| Moirai 2.0                 | Past-only | Native                  | `past_feat_dynamic_real`                                   |
| TiRex, TiRex 1.1 GIFT-Eval | Past-only | Channel concatenation   | `context` = targets + covariates                           |
| Chronos-2                  | Both      | Native                  | `past_covariates`, `future_covariates`                     |
| TimesFM 200M               | Both      | Native                  | `forecast_with_covariates`, `dynamic_numerical_covariates` |
| Tiny Time Mixer            | Both      | Native                  | sktime `X_fit`, `X_pred`                                   |
| Lag-Llama                  | Past-only | Iteration over variates | One call per variate; discard covariate preds              |
| Granite FlowState          | Past-only | Iteration over variates | One call per variate; discard covariate preds              |
| LAFN                       | None      | N/A                     | N/A                                                        |
