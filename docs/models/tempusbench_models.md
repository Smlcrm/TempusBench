# Models Served in Tempus Bench

This document lists all forecasting models available in the Tempus Bench repository. Models are discovered automatically from `tempus_bench/models/` and must have both a `{model_name}_model.py` file and a `settings.yaml` file.

## Table Convention

- **Output Modality**: Deterministic / Stochastic / Hybrid
- **Multivariate Strategy**: Channel Independent (iterate) / Native multivariate
- **Covariate Integration**: None / Past + Future / Future only / Past only
- **Autoregressive Generation**: Yes / No

## Summary (60 models)

### Statistical (7)

| Model                 | Output        | Description                                    |
| --------------------- | ------------- | ---------------------------------------------- |
| arima                 | Deterministic | Autoregressive Integrated Moving Average       |
| croston_classic       | Deterministic | Intermittent demand forecasting                |
| exponential_smoothing | Deterministic | Holt-Winters (trend + seasonality)             |
| prophet               | Deterministic | Meta's additive decomposition model            |
| seasonal_naive        | Deterministic | Seasonal baseline                              |
| theta                 | Deterministic | Theta method (theta lines decomposition)       |
| varmax                | Deterministic | Vector autoregression with exogenous variables |

### ML (11)

| Model         | Output        | Description                                         |
| ------------- | ------------- | --------------------------------------------------- |
| deepar        | Deterministic | Amazon's autoregressive RNN with future covariates  |
| itransformer  | Deterministic | ICLR 2024 multivariate inverted Transformer         |
| lstm          | Deterministic | Long Short-Term Memory RNN                          |
| nbeats        | Deterministic | N-BEATS neural basis expansion (ICLR 2020)          |
| nhits         | Deterministic | N-HiTS hierarchical interpolation with covariates   |
| patchtsmixer  | Deterministic | IBM MLP-Mixer with native channel-mixing (KDD 2023) |
| random_forest | Deterministic | Ensemble of decision trees                          |
| svr           | Deterministic | Support Vector Regression                           |
| tft           | Deterministic | Temporal Fusion Transformer with covariates         |
| timesnet      | Deterministic | TimesNet CNN with future covariates (ICLR 2023)     |
| xgboost       | Deterministic | Extreme Gradient Boosting                           |

### Foundation (42)

| Model                | Output        | Params | HF Model                                | Description                                          |
| -------------------- | ------------- | ------ | --------------------------------------- | ---------------------------------------------------- |
| chronos_tiny         | Stochastic    | 8M     | amazon/chronos-t5-tiny                  | Chronos T5 (tiny)                                    |
| chronos_mini         | Stochastic    | 20M    | amazon/chronos-t5-mini                  | Chronos T5 (mini)                                    |
| chronos_small        | Stochastic    | 46M    | amazon/chronos-t5-small                 | Chronos T5 (small)                                   |
| chronos_base         | Stochastic    | 200M   | amazon/chronos-t5-base                  | Chronos T5 (base)                                    |
| chronos_large        | Stochastic    | 710M   | amazon/chronos-t5-large                 | Chronos T5 (large)                                   |
| chronos_bolt_tiny    | Stochastic    | 9M     | amazon/chronos-bolt-tiny                | Chronos-Bolt encoder (tiny)                          |
| chronos_bolt_mini    | Stochastic    | 21M    | amazon/chronos-bolt-mini                | Chronos-Bolt encoder (mini)                          |
| chronos_bolt_small   | Stochastic    | 48M    | amazon/chronos-bolt-small               | Chronos-Bolt encoder (small)                         |
| chronos_bolt_base    | Stochastic    | 205M   | amazon/chronos-bolt-base                | Chronos-Bolt encoder (base)                          |
| chronos2             | Stochastic    | 120M   | amazon/chronos-2                        | Chronos-2 with native multivariate + covariates      |
| chronos2_small       | Stochastic    | 40M    | autogluon/chronos-2-small               | Chronos-2 small                                      |
| granite_flowstate    | Stochastic    | 9.1M   | ibm-granite/granite-tsfm                | IBM FlowState quantile model                         |
| kairos_10m           | Stochastic    | 10M    | mldi-lab/Kairos_10m                     | Kairos adaptive TSFM (10M)                           |
| kairos_23m           | Stochastic    | 23M    | mldi-lab/Kairos_23m                     | Kairos adaptive TSFM (23M)                           |
| kairos_50m           | Stochastic    | 50M    | mldi-lab/Kairos_50m                     | Kairos adaptive TSFM (50M)                           |
| lafn                 | Hybrid        | —      | —                                       | Chronarium-backed Large Adaptive Forecasting Network |
| lagllama             | Stochastic    | 10M    | time-series-foundation-models/lag-llama | Lag-Llama LLM for time series                        |
| moirai_small         | Stochastic    | 14M    | Salesforce/moirai-1.1-R-small           | Moirai 1.1 (small)                                   |
| moirai_base          | Stochastic    | 91M    | Salesforce/moirai-1.1-R-base            | Moirai 1.1 (base)                                    |
| moirai_large         | Stochastic    | 311M   | Salesforce/moirai-1.1-R-large           | Moirai 1.1 (large)                                   |
| moirai_moe           | Stochastic    | 86M    | Salesforce/moirai-moe-1.0-R-base        | Moirai Mixture-of-Experts                            |
| moirai2              | Stochastic    | 12M    | Salesforce/moirai-2                     | Moirai 2.0 decoder-only                              |
| moment_small         | Deterministic | 40M    | AutonLab/MOMENT-1-small                 | MOMENT (small)                                       |
| moment_base          | Deterministic | 125M   | AutonLab/MOMENT-1-base                  | MOMENT (base)                                        |
| moment_large         | Deterministic | 385M   | AutonLab/MOMENT-1-large                 | MOMENT (large)                                       |
| patchtst_fm          | Stochastic    | 260M   | ibm-research/patchtst-fm-r1             | PatchTST-FM pretrained zero-shot                     |
| patchtst_granite     | Stochastic    | 40M    | ibm-granite/granite-timeseries-patchtst | PatchTST fine-tuned on ETTh1                         |
| sundial              | Stochastic    | 128M   | —                                       | Tsinghua diffusion-based model                       |
| tabpfn               | Deterministic | 1M     | —                                       | TabPFN for tabular data                              |
| tabpfn_ts            | Stochastic    | 1M     | —                                       | TabPFN-TS zero-shot with covariates                  |
| time_moe_50m         | Deterministic | 50M    | Maple728/TimeMoE-50M                    | Time-MoE (50M)                                       |
| time_moe_200m        | Deterministic | 200M   | Maple728/TimeMoE-200M                   | Time-MoE (200M)                                      |
| timesfm_200m         | Deterministic | 200M   | google/timesfm-1.0-200m-pytorch         | TimesFM 1.0 (200M)                                   |
| timesfm2             | Stochastic    | 200M   | google/timesfm-2.5-200m-transformers    | TimesFM 2.5 quantile forecasts                       |
| timesfm_500m         | Stochastic    | 500M   | google/timesfm-2.0-500m-transformers    | TimesFM 2.0 (500M)                                   |
| tiny_time_mixer_r1   | Deterministic | 1M     | ibm/TTM                                 | Tiny Time Mixer R1                                   |
| tiny_time_mixer_r2   | Deterministic | 1.7M   | ibm-granite/granite-timeseries-ttm-r2   | TTM R2                                               |
| tiny_time_mixer_r2_1 | Deterministic | 1.7M   | ibm-granite/granite-timeseries-ttm-r2.1 | TTM R2.1                                             |
| tirex                | Stochastic    | 300M   | NX-AI/TiRex                             | TiRex 1.0 with covariate support                     |
| tirex_1_1_gifteval   | Stochastic    | 300M   | NX-AI/TiRex-1.1-gifteval                | TiRex 1.1 (GIFT-Eval trained)                        |
| toto                 | Stochastic    | 151M   | Datadog/Toto-Open-Base-1.0              | Datadog multi-modal model                            |

## Covariate Support

Models that natively use covariates when provided:

| Model                                     | Past covariates | Future covariates |
| ----------------------------------------- | :-------------: | :---------------: |
| chronos2 / chronos2_small                 |       Yes       |        Yes        |
| nhits                                     |       Yes       |        Yes        |
| tft                                       |       Yes       |        Yes        |
| tirex / tirex_1_1_gifteval                |       Yes       |        Yes        |
| timesfm_200m                              |       Yes       |        Yes        |
| moirai_small / moirai_base / moirai_large |       Yes       |        Yes        |
| moirai_moe                                |       Yes       |        Yes        |
| moirai2                                   |       Yes       |        Yes        |
| prophet                                   |       Yes       |        Yes        |
| arima                                     |       Yes       |        Yes        |
| toto                                      |       Yes       |        Yes        |
| tiny_time_mixer_r1 / r2 / r2_1            |       Yes       |        Yes        |
| tabpfn_ts                                 |       Yes       |        Yes        |
| deepar                                    |        —        |        Yes        |
| timesnet                                  |        —        |        Yes        |
| varmax                                    |       Yes       |         —         |

## Configuration

Models are configured in `tempus_bench/config/benchmark.yaml` under the `model:` section. Hyperparameters and tuning grids are defined there. Note: `xgboost` is commented out in the default config but the model implementation is available.
