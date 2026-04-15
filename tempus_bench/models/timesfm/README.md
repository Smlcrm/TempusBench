---
license: apache-2.0
library_name: timesfm
pipeline_tag: time-series-forecasting
---

# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting.

**Resources and Technical Documentation**:

* Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688), ICML 2024.
* [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
* [GitHub repo](https://github.com/google-research/timesfm)

**Authors**: Google Research

This is not an officially supported Google product.

## Checkpoint timesfm-2.0-500m

`timesfm-2.0-500m` is the second open model checkpoint:

- It performs univariate time series forecasting for context lengths up to 2048 time points and any horizon lengths, with an optional frequency indicator. Note that it can go even beyond 2048 context even though it was trained with that as the maximum context.
- It focuses on point forecasts. We experimentally offer 10 quantile heads but they have not been calibrated after pretraining.
- It ideally requires the context to be contiguous (i.e. no "holes"), and the context and the horizon to be of the same frequency. In case there are nans we fill in the missing values with linear interpolation before calling the model.


## Installation

This HuggingFace repo hosts TimesFm checkpoints. Please visit our [GitHub repo](https://github.com/google-research/timesfm) and follow the instructions for the PAX version to install the `timesfm` library for model inference.


## Usage 

### Initialize the model and load a checkpoint.
Then the base class can be loaded as,

```python
import timesfm

# For PAX
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend=<backend>,
          per_core_batch_size=32,
          horizon_len=128,
          input_patch_len=32,
          output_patch_len=128,
          num_layers=50,
          model_dims=1280,
          use_positional_embedding=False,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
  )
```

Note that the five parameters are fixed to load the 500m model

```python
input_patch_len=32,
output_patch_len=128,
num_layers=50,
model_dims=1280,
use_positional_embedding=False,
```

1. The context_len here can be set as the max context length **of the model**. You can provide a shorter series to the `tfm.forecast()` function and the model will handle it. Currently, the model handles a max context length of 2048, which can be increased in later releases. The input time series can have **any context length**. Padding / truncation will be handled by the inference code if needed.

2. The horizon length can be set to anything. We recommend setting it to the largest horizon length you would need in the forecasting tasks for your application. We generally recommend horizon length <= context length but it is not a requirement in the function call.

### Perform inference

We provide APIs to forecast from either array inputs or `pandas` dataframe. Both forecast methods expect (1) the input time series contexts, (2) along with their frequencies. Please look at the documentation of the functions `tfm.forecast()` and `tfm.forecast_on_df()` for detailed instructions.

In particular, regarding the frequency, TimesFM expects a categorical indicator valued in {0, 1, 2}:

- **0** (default): high frequency, long horizon time series. We recommend using this for time series up to daily granularity.
- **1**: medium frequency time series. We recommend using this for weekly and monthly data.
- **2**: low frequency, short horizon time series. We recommend using this for anything beyond monthly, e.g. quarterly or yearly.

This categorical value should be directly provided with the array inputs. For dataframe inputs, we convert the conventional letter coding of frequencies to our expected categories, that

- **0**: T, MIN, H, D, B, U
- **1**: W, M
- **2**: Q, Y

Notice you do **NOT** have to strictly follow our recommendation here. Although this is our setup during model training and we expect it to offer the best forecast result, you can also view the frequency input as a free parameter and modify it per your specific use case.


Examples:

Array inputs, with the frequencies set to low, medium, and high respectively.

```python
import numpy as np
forecast_input = [
    np.sin(np.linspace(0, 20, 100))
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)
```

`pandas` dataframe, with the frequency set to "M" monthly.

```python
import pandas as pd

# e.g. input_df is
#       unique_id  ds          y
# 0     T1         1975-12-31  697458.0
# 1     T1         1976-01-31  1187650.0
# 2     T1         1976-02-29  1069690.0
# 3     T1         1976-03-31  1078430.0
# 4     T1         1976-04-30  1059910.0
# ...   ...        ...         ...
# 8175  T99        1986-01-31  602.0
# 8176  T99        1986-02-28  684.0
# 8177  T99        1986-03-31  818.0
# 8178  T99        1986-04-30  836.0
# 8179  T99        1986-05-31  878.0

forecast_df = tfm.forecast_on_df(
    inputs=input_df,
    freq="M",  # monthly
    value_name="y",
    num_jobs=-1,
)
```

## Pretraining Data

It is important to list all the data sources in order to enable fair benchmarking. The TimesFM 2.0 series contains the pretraining set of TimesFM 1.0 along with these additional datasets (a subset of the [LOTSA](https://arxiv.org/abs/2402.02592) pretraining data that many other models are pretrained on):

| Dataset                                     | Download Source          |
| :------------------------------------------ | :----------------------- |
| azure_vm_traces                             | LOTSA Pretrain         |
| residential_load_power                      | LOTSA Pretrain         |
| borg_cluster_data                           | LOTSA Pretrain         |
| residential_pv_power                        | LOTSA Pretrain         |
| q_traffic                                   | LOTSA Pretrain         |
| london_smart_meters_with_missing             | LOTSA Pretrain         |
| taxi_30min                                 | LOTSA Pretrain         |
| solar_power                                 | LOTSA Pretrain         |
| wind_power                                  | LOTSA Pretrain         |
| kdd2022                                   | LOTSA Pretrain         |
| largest                                     | LOTSA Pretrain         |
| era5                                       | LOTSA Pretrain         |
| buildings                                   | LOTSA Pretrain         |
| cmip6                                     | LOTSA Pretrain         |
| china_air_quality                         | LOTSA Pretrain         |
| beijing_air_quality                       | LOTSA Pretrain         |
| subseasonal                                | LOTSA Pretrain         |
| kaggle_web_traffic_weekly                 | LOTSA Pretrain         |
| cdc_fluview_who_nrevss                  | LOTSA Pretrain         |
| godaddy                                   | LOTSA Pretrain         |