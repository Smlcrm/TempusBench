# Models Served in Chronax

This document lists all forecasting models available in the [Chronax](https://github.com/Smlcrm/Chronax) repository. Chronax is a JAX-accelerated time-series forecasting library with a unified `fit()` → `predict()` interface. All models are importable from `chronax.models`.

## Summary

| Model | Category | Description |
|-------|-----------|-------------|
| AutoARIMA | Automatic | Auto model selection for ARIMA |
| AutoETS | Automatic | Auto model selection for ETS |
| AutoTheta | Automatic | Auto model selection for Theta |
| AutoTBATS | Automatic | Auto model selection for TBATS |
| AutoMFLES | Automatic | Auto model selection for MFLES |
| AutoCES | Automatic | Auto model selection for CES |
| ARIMA | ARIMA | Autoregressive integrated moving average |
| ETS | Exponential Smoothing | Error-Trend-Seasonal |
| Theta | Theta | Theta-method decomposition |
| TBATS | Multi-Frequency | Trigonometric seasonality |
| MFLES | Multi-Frequency | Multiple frequency linear exponential smoothing |
| Holt | Exponential Smoothing | Holt's linear trend |
| HoltWinters | Exponential Smoothing | Holt-Winters (trend + seasonality) |
| SimpleExponentialSmoothing | Exponential Smoothing | Single exponential smoothing |
| SeasonalExponentialSmoothing | Exponential Smoothing | Seasonal exponential smoothing |
| MSTL | Decomposition | Multiple seasonal-trend decomposition |
| STL | Decomposition | Seasonal-trend decomposition (LOESS) |
| GARCH | Volatility | Generalized autoregressive conditional heteroskedasticity |
| Naive | Baseline | Last observation |
| SeasonalNaive | Baseline | Same period last season |
| HistoricAverage | Baseline | Mean of all observations |
| WindowAverage | Baseline | Mean over recent window |
| SeasonalWindowAverage | Baseline | Seasonal window average |
| RandomWalkWithDrift | Baseline | Random walk with drift |
| CrostonClassic | Intermittent | Croston method for sparse demand |
| TSB | Intermittent | Teunter-Syntetos-Babai |
| ADIDA | Intermittent | Aggregate-disaggregate intermittent demand |
| IMAPA | Intermittent | Intermittent multiple aggregation |
| BatchedForecaster | Multi-Series | Wrapper for batched multi-series forecasting |

## By Category

### Automatic Forecasting (6)

Model-selection wrappers that search over candidate configurations.

- **AutoARIMA** — Auto ARIMA with exogenous regressors
- **AutoETS** — Auto ETS (Error-Trend-Seasonal)
- **AutoTheta** — Auto Theta method
- **AutoTBATS** — Auto TBATS (trigonometric seasonality)
- **AutoMFLES** — Auto MFLES with exogenous regressors
- **AutoCES** — Auto Complex Exponential Smoothing

### ARIMA Family (1)

- **ARIMA** — Autoregressive integrated moving average (with exogenous regressors)

### Theta Family (1)

- **Theta** — Theta-method for trend and seasonality decomposition

### Exponential Smoothing (5)

- **ETS** — Error-Trend-Seasonal (full ETS family)
- **Holt** — Holt's linear trend
- **HoltWinters** — Holt-Winters (additive/multiplicative)
- **SimpleExponentialSmoothing** — Single exponential smoothing
- **SeasonalExponentialSmoothing** — Seasonal exponential smoothing

### Multi-Frequency & Decomposition (4)

- **TBATS** — Trigonometric Box-Cox ARIMA trend seasonal
- **MFLES** — Multiple frequency linear exponential smoothing
- **MSTL** — Multiple seasonal-trend decomposition
- **STL** — Seasonal-trend decomposition (LOESS)

### Volatility (1)

- **GARCH** — GARCH for time-varying variance

### Baseline Models (6)

- **Naive** — Last observation
- **SeasonalNaive** — Same period last season
- **HistoricAverage** — Mean of all observations
- **WindowAverage** — Mean over recent window
- **SeasonalWindowAverage** — Seasonal window average
- **RandomWalkWithDrift** — Random walk with drift

### Intermittent Demand (4)

- **CrostonClassic** — Croston method for sparse demand
- **TSB** — Teunter-Syntetos-Babai
- **ADIDA** — Aggregate-disaggregate intermittent demand
- **IMAPA** — Intermittent multiple aggregation

### Multi-Series (1)

- **BatchedForecaster** — Wrapper for running models on multiple series in batch

## Features

- **JAX-accelerated** — JIT-compiled fitting and forecasting on CPU, GPU, or TPU
- **Unified API** — Every model follows `fit()` → `predict()` (or `forecast()`)
- **Prediction intervals** — Native and conformal interval support
- **NumPy compatible** — Accepts and returns standard array types

## Source

Models are defined in `/chronax/models/` and exported from `chronax.models.__init__.py`. See [Chronax on GitHub](https://github.com/Smlcrm/Chronax).
