# Models Served in StatsForecast

This document lists all forecasting models available in the [StatsForecast](https://github.com/Nixtla/statsforecast) repository by Nixtla. StatsForecast is a high-performance library for statistical and econometric time series forecasting. All models are importable from `statsforecast.models` and follow a unified `fit()` → `predict()` interface.

## Summary

| Model | Category | Description |
|-------|----------|-------------|
| AutoARIMA | Automatic | Auto ARIMA with exogenous regressors |
| AutoETS | Automatic | Auto Error-Trend-Seasonal |
| AutoCES | Automatic | Auto Complex Exponential Smoothing |
| AutoTheta | Automatic | Auto Theta method |
| AutoMFLES | Automatic | Auto MFLES with exogenous regressors |
| AutoTBATS | Automatic | Auto TBATS (trigonometric seasonality) |
| ARIMA | ARIMA | Autoregressive integrated moving average |
| AutoRegressive | ARIMA | AR without differencing |
| SimpleExponentialSmoothing | Exponential Smoothing | Single exponential smoothing |
| SimpleExponentialSmoothingOptimized | Exponential Smoothing | Optimized SES |
| SeasonalExponentialSmoothing | Exponential Smoothing | Seasonal exponential smoothing |
| SeasonalExponentialSmoothingOptimized | Exponential Smoothing | Optimized seasonal SES |
| Holt | Exponential Smoothing | Holt's linear trend |
| HoltWinters | Exponential Smoothing | Holt-Winters (trend + seasonality) |
| HistoricAverage | Baseline | Mean of all observations |
| Naive | Baseline | Last observation |
| RandomWalkWithDrift | Baseline | Random walk with drift |
| SeasonalNaive | Baseline | Same period last season |
| WindowAverage | Baseline | Mean over recent window |
| SeasonalWindowAverage | Baseline | Seasonal window average |
| ADIDA | Intermittent | Aggregate-disaggregate intermittent demand |
| CrostonClassic | Intermittent | Croston method for sparse demand |
| CrostonOptimized | Intermittent | Optimized Croston |
| CrostonSBA | Intermittent | Croston with Syntetos-Boylan approximation |
| IMAPA | Intermittent | Intermittent multiple aggregation |
| TSB | Intermittent | Teunter-Syntetos-Babai |
| MSTL | Multiple Seasonalities | Multiple seasonal-trend decomposition |
| MFLES | Multiple Seasonalities | Multiple frequency linear exponential smoothing |
| TBATS | Multiple Seasonalities | Trigonometric Box-Cox ARIMA trend seasonal |
| Theta | Theta | Standard Theta method |
| OptimizedTheta | Theta | Optimized Theta |
| DynamicTheta | Theta | Dynamic Theta |
| DynamicOptimizedTheta | Theta | Dynamic optimized Theta |
| GARCH | Volatility | GARCH for time-varying variance |
| ARCH | Volatility | ARCH (GARCH with q=0) |
| UCM | Structural | Unobserved Components Model |
| SklearnModel | ML | Wrapper for scikit-learn models |
| ConstantModel | Fallback | Constant forecast (fallback) |
| ZeroModel | Fallback | Zero forecast (fallback) |
| NaNModel | Fallback | NaN forecast (fallback) |

## By Category

### Automatic Forecasting (6)

Model-selection wrappers that search for the best parameters.

- **AutoARIMA** — Auto ARIMA with exogenous regressors
- **AutoETS** — Auto Error-Trend-Seasonal
- **AutoCES** — Auto Complex Exponential Smoothing
- **AutoTheta** — Auto Theta method
- **AutoMFLES** — Auto MFLES with exogenous regressors
- **AutoTBATS** — Auto TBATS (trigonometric seasonality)

### ARIMA Family (2)

- **ARIMA** — Autoregressive integrated moving average
- **AutoRegressive** — AR model without differencing

### Exponential Smoothing (6)

- **SimpleExponentialSmoothing** — Single exponential smoothing
- **SimpleExponentialSmoothingOptimized** — Optimized single exponential smoothing
- **SeasonalExponentialSmoothing** — Seasonal exponential smoothing
- **SeasonalExponentialSmoothingOptimized** — Optimized seasonal exponential smoothing
- **Holt** — Holt's linear trend (subclass of AutoETS)
- **HoltWinters** — Holt-Winters (subclass of AutoETS)

### Baseline Models (6)

- **Naive** — Last observation
- **SeasonalNaive** — Same period last season
- **HistoricAverage** — Mean of all observations
- **WindowAverage** — Mean over recent window
- **SeasonalWindowAverage** — Seasonal window average
- **RandomWalkWithDrift** — Random walk with drift

### Intermittent Demand (6)

- **CrostonClassic** — Croston method for sparse demand
- **CrostonOptimized** — Optimized Croston
- **CrostonSBA** — Croston with Syntetos-Boylan approximation
- **ADIDA** — Aggregate-disaggregate intermittent demand
- **IMAPA** — Intermittent multiple aggregation
- **TSB** — Teunter-Syntetos-Babai

### Multiple Seasonalities (3)

- **MSTL** — Multiple seasonal-trend decomposition (LOESS)
- **MFLES** — Multiple frequency linear exponential smoothing
- **TBATS** — Trigonometric Box-Cox ARIMA trend seasonal

### Theta Family (4)

- **Theta** — Standard Theta method
- **OptimizedTheta** — Optimized Theta
- **DynamicTheta** — Dynamic Theta
- **DynamicOptimizedTheta** — Dynamic optimized Theta

### ARCH/GARCH Family (2)

- **GARCH** — Generalized autoregressive conditional heteroskedasticity
- **ARCH** — ARCH (GARCH with q=0)

### Structural Models (1)

- **UCM** — Unobserved Components Model (level, trend, seasonal, cycle, irregular)

### Machine Learning (1)

- **SklearnModel** — Wrapper for scikit-learn regressors (RandomForest, Ridge, etc.)

### Fallback Models (3)

Used when other models fail during forecasting.

- **ConstantModel** — Constant forecast
- **ZeroModel** — Zero forecast
- **NaNModel** — NaN forecast

## Adapters

- **AutoARIMAProphet** — Prophet-compatible interface using AutoARIMA backend (`statsforecast.adapters.prophet`). Drop-in replacement for Prophet with improved speed and accuracy.

## Features

- **High performance** — Optimized implementations (e.g., 20x faster than pmdarima, 4x faster than statsmodels)
- **Unified API** — All models follow `fit()` → `predict()` (sklearn-style)
- **Probabilistic forecasting** — Native and conformal prediction intervals
- **Exogenous variables** — Supported by AutoARIMA, AutoMFLES, ARIMA
- **Distributed** — Compatible with Spark, Dask, and Ray
- **Multi-series** — Efficient forecasting of millions of time series via `StatsForecast`

## Source

Models are defined in `python/statsforecast/models.py` and exported from `statsforecast.models`. See [StatsForecast on GitHub](https://github.com/Nixtla/statsforecast).
