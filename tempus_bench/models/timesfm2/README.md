# TimesFM 2.5

TimesFM 2.5 (Time Series Foundation Model) is Google Research's pretrained decoder-only foundation model for time-series forecasting, available via Hugging Face transformers.

## Features

- **Stochastic output**: Continuous quantile forecasts (10th–90th percentiles) converted to pseudo-samples for CRPS and other probabilistic metrics
- **Multivariate**: Iterates over targets (channel-independent)
- **Non-autoregressive**: Predicts full horizon in one forward pass (up to 128 steps; longer horizons padded)
- **Covariates**: Not natively supported in TimesFM 2.5; ignored when provided

## Installation

```bash
pip install "transformers>=5.3.0" torch numpy
```

## Checkpoint

- `google/timesfm-2.5-200m-transformers` (default)

## References

- [GitHub repo](https://github.com/google-research/timesfm)
- [Hugging Face model](https://huggingface.co/google/timesfm-2.5-200m-transformers)
- [Paper](https://arxiv.org/abs/2310.10688) (ICML 2024)
