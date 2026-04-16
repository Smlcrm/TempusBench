# Model Set Comparison: (Tempus Bench ∪ Chronax) vs StatsForecast

**Definition of "we have"**: A model is considered to be in our set if it exists in **Chronax** OR **Tempus Bench**. The union of both repositories defines what we have.

This document compares our models against **StatsForecast**, using equivalent model families where naming differs (e.g., `arima` ↔ `ARIMA`, `seasonal_naive` ↔ `SeasonalNaive`).

## Equivalence Mapping

| Tempus Bench          | Chronax                                        | StatsForecast                             |
| --------------------- | ---------------------------------------------- | ----------------------------------------- |
| arima                 | ARIMA                                          | ARIMA                                     |
| seasonal_naive        | SeasonalNaive                                  | SeasonalNaive                             |
| croston_classic       | CrostonClassic                                 | CrostonClassic                            |
| exponential_smoothing | HoltWinters, ETS, SeasonalExponentialSmoothing | HoltWinters, SeasonalExponentialSmoothing |
| theta                 | Theta                                          | Theta                                     |

---

## Models WE Have That StatsForecast Does NOT Have

_(Models in Chronax or Tempus Bench with no equivalent in StatsForecast)_

### Foundation / Deep Learning Models — Tempus Bench

- **chronos** — Amazon's time series foundation model
- **lagllama** — Large language model for time series
- **moirai** — Microsoft's foundation model for forecasting
- **moirai_moe** — Moirai Mixture-of-Experts
- **moment** — Foundation model (statistical moments)
- **timesfm** — Google's time series foundation model
- **tiny_time_mixer** — IBM's lightweight foundation model
- **toto** — Multi-modal foundation model (Datadog)

### Classical ML / Statistical — Tempus Bench

- **prophet** — Meta's additive decomposition model
- **random_forest** — Ensemble of decision trees
- **svr** — Support Vector Regression
- **xgboost** — Extreme Gradient Boosting
- **lstm** — Long Short-Term Memory RNN
- **varmax** — Vector autoregression with exogenous variables

### Other — Tempus Bench

- **tabpfn** — Pre-trained Transformer for tabular data
- **lafn** — Large Adaptive Forecasting Network (Chronarium-backed, hybrid)

### Chronax

- **STL** — Seasonal-trend decomposition (LOESS) — StatsForecast has MSTL but not standalone STL
- **BatchedForecaster** — Multi-series wrapper (StatsForecast has different architecture via `StatsForecast` class)

**Total: 20 models** we have that StatsForecast does not.

---

## Models StatsForecast Has That WE Do NOT Have

_(Models in StatsForecast with no equivalent in Chronax or Tempus Bench)_

### Optimized Variants

- **SimpleExponentialSmoothingOptimized** — Optimized SES
- **SeasonalExponentialSmoothingOptimized** — Optimized seasonal SES

### ARIMA Family

- **AutoRegressive** — AR without differencing

### Intermittent Demand (extra Croston variants)

- **CrostonOptimized** — Optimized Croston
- **CrostonSBA** — Croston with Syntetos-Boylan approximation

### Theta Family (extra variants)

- **OptimizedTheta**
- **DynamicTheta**
- **DynamicOptimizedTheta**

### Volatility

- **ARCH** — ARCH (GARCH with q=0) _(Chronax has GARCH)_

### Structural

- **UCM** — Unobserved Components Model (level, trend, seasonal, cycle, irregular)

### ML / Fallbacks

- **SklearnModel** — Generic wrapper for any scikit-learn regressor
- **ConstantModel** — Fallback constant forecast
- **ZeroModel** — Fallback zero forecast
- **NaNModel** — Fallback NaN forecast

### Adapters

- **AutoARIMAProphet** — Prophet-compatible interface using AutoARIMA backend

**Total: 15 models** StatsForecast has that we don't.

_We have AutoARIMA, AutoETS, AutoCES, AutoTheta, AutoMFLES, AutoTBATS, ETS, Holt, HoltWinters, SimpleExponentialSmoothing, SeasonalExponentialSmoothing, MSTL, GARCH, and the baselines/intermittent models via Chronax._

---

## Summary

| Direction                          | Count |
| ---------------------------------- | ----- |
| **We have, StatsForecast doesn't** | 20    |
| **StatsForecast has, we don't**    | 15    |

### Key Takeaways

1. **Our unique strength**: Foundation models (Chronos, LagLlama, Moirai, TimesFM, TinyTimeMixer, Toto), Prophet, classical ML (LSTM, XGBoost, Random Forest, SVR), TabPFN, LAFN, and VARMAX.

2. **StatsForecast's unique strength**: Optimized SES variants, extra Theta variants (Optimized/Dynamic), CrostonOptimized/CrostonSBA, AutoRegressive, ARCH, UCM, SklearnModel wrapper, and fallback models.

3. **Overlap**: Both ecosystems cover the same classical statistical core (ARIMA, Theta, ETS, Holt-Winters, baselines, intermittent demand). Chronax + Tempus Bench together cover most of StatsForecast's classical models via Chronax.
