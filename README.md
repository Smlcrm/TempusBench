<!-- smlcrm:begin header -->
<!-- Logo: Smlcrm/design-system assets/logo/logo-digital-1.svg @ cf980f0. Light fill #2121a5 = token product.logo-ink; dark fill #ffffff = token brand-book.brand-white. -->
<p align="center">
  <a href="https://smlcrm.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/smlcrm-logo-dark.svg">
      <img alt="Simulacrum" src=".github/assets/smlcrm-logo-light.svg" width="300">
    </picture>
  </a>
</p>

<h1 align="center">TempusBench</h1>

<p align="center">A framework that benchmarks statistical, machine-learning and foundation forecasting models on shared tasks.</p>

<p align="center"><a href="https://smlcrm.com">smlcrm.com</a></p>
<!-- smlcrm:end header -->

<!-- smlcrm:begin badges -->
<!-- Badge colours: 2121a5 = token brand-book.brand-dark-blue (arXiv, license, language); 3483fa = token brand-book.brand-bright-blue (release). -->
<p align="center">
  <a href="https://arxiv.org/abs/2604.11529"><img alt="arXiv: 2604.11529" src="https://img.shields.io/badge/arXiv-2604.11529-2121a5"></a>
  <a href="LICENSE"><img alt="License: CC BY-NC-ND 4.0" src="https://img.shields.io/github/license/Smlcrm/TempusBench?color=2121a5"></a>
  <a href="https://github.com/Smlcrm/TempusBench/releases"><img alt="Release: latest GitHub release" src="https://img.shields.io/github/v/release/Smlcrm/TempusBench?color=3483fa"></a>
  <!-- smlcrm:no-ci: the repository has no build or test workflow (web-docs-dev.yml only publishes documentation) -->
  <a href="pyproject.toml"><img alt="Language: Python 3.11 or later" src="https://img.shields.io/badge/python-%E2%89%A53.11-2121a5"></a>
</p>
<!-- smlcrm:end badges -->

<!-- smlcrm:begin overview -->
## Overview

TempusBench evaluates time-series forecasting models on a shared set of univariate, multivariate and covariate tasks. It runs each model on rolling windows of every task, tunes hyperparameters, scores point forecasts (MAE, RMSE, MAPE, MASE) and probabilistic forecasts (CRPS, quantile score, weighted interval score, energy score), and summarises models across tasks with win rates and skill scores against a baseline. Models range from statistical baselines to foundation models such as Chronos, Moirai and TimesFM, and each runs in its own conda environment so their dependencies do not conflict. Task data downloads from the public Hugging Face dataset [Smlcrm/tempus_bench_tasks](https://huggingface.co/datasets/Smlcrm/tempus_bench_tasks) on first use.

**Who it is for.** Researchers and engineers who compare forecasting models, or add a model and want it scored the same way as the others.

**What it does not do.** It does not train foundation models or ship their weights; it downloads or loads them per model. It is a local Python package and command line, not a hosted service, and it needs conda to run the models. The license (CC BY-NC-ND 4.0) does not permit commercial use or redistribution of modified versions.
<!-- smlcrm:end overview -->

<!-- smlcrm:begin quickstart -->
## Quickstart

> **Untested.** The install command was not completed when this README was written: downloads of the dependencies (TensorFlow among them) stalled on the network. The example itself was run on 2026-10-05 against the package source (metrics and aggregators identical to v1.0.0) with Python 3.11.14, numpy 2.4.4 and pandas 3.0.2, and printed the output below.

Requires Python 3.11 or later. Running whole benchmarks also needs conda; the example below does not.

```bash
pip install "tempus_bench @ https://github.com/Smlcrm/TempusBench/archive/refs/tags/v1.0.0.tar.gz"
```

Save this as `score.py` and run `python score.py`. It scores two models' forecasts on two tasks with MAE, then aggregates them across tasks the way a benchmark run does.

```python
import numpy as np
import pandas as pd
from tempus_bench.metrics.mae import MAE
from tempus_bench.aggregators import SkillScore, WinRate

# Two tasks, each with a held-out horizon and two models' point forecasts
tasks = {
    "airline": (np.array([[417.0], [391.0], [419.0], [461.0]]),
                {"seasonal_naive": np.array([[360.0], [342.0], [406.0], [396.0]]),
                 "my_model":       np.array([[410.0], [395.0], [425.0], [450.0]])}),
    "births":  (np.array([[44.0], [40.0], [38.0], [43.0]]),
                {"seasonal_naive": np.array([[37.0], [45.0], [40.0], [35.0]]),
                 "my_model":       np.array([[41.0], [42.0], [41.0], [40.0]])}),
}

mae = MAE()
scores = pd.DataFrame({
    task: {model: mae(y_true, y_pred, model_type="deterministic") for model, y_pred in preds.items()}
    for task, (y_true, preds) in tasks.items()
})  # models as rows, tasks as columns

print(scores.round(2))
print("skill vs seasonal_naive:", SkillScore(scores, baseline_model="seasonal_naive")().round(3).to_dict())
print("win rate:", WinRate(scores)().to_dict())
```

Expected output:

```text
                airline  births
seasonal_naive     46.0    5.50
my_model            7.0    2.75
skill vs seasonal_naive: {'seasonal_naive': 0.0, 'my_model': 0.724}
win rate: {'seasonal_naive': 0.0, 'my_model': 1.0}
```

To run a full benchmark from a clone, `python -m tempus_bench.run_benchmark --config tempus_bench/config/benchmark.yaml`; see [Usage](#usage).
<!-- smlcrm:end quickstart -->

This repository is the **TempusBench** Python package and its assets (tasks, models, tests, docs). **Cloud/UI, worker, GCP deploy scripts, and Dockerfiles** live in a separate private repo (for example **`inference-tempusbench-cloud`** under your org); clone that repo and place this library at **`tempusbench_open/`** next to `tempusbench_cloud/` and `deployment/` to match its documented layout.

## Supported models and features

TempusBench supports:

- **Traditional Models**: ARIMA, LSTM, XGBoost, SVR, Prophet, Random Forest, Theta, DeepAR, Exponential Smoothing, Croston Classic, Seasonal Naive, TabPFN
- **Foundation Models**: Chronos, LagLlama, Moirai, TimesFM, Tiny Time Mixer, Toto, Moment
- **Deterministic and Stochastic Forecasting**: Automatic routing based on task type
- **Univariate and Multivariate Forecasting**: All models handle both seamlessly
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Isolated Execution**: Each model runs in its own conda environment to avoid dependency conflicts

## Task data (Hugging Face)

Task CSV files are hosted on Hugging Face to keep this repository lightweight:

**[Smlcrm/tempus_bench_tasks](https://huggingface.co/datasets/Smlcrm/tempus_bench_tasks)** (public dataset)

On the first `run_benchmark` (or any call that resolves `tempus_bench/tasks/`), missing task
CSVs are downloaded automatically with a progress bar. Upload or refresh the dataset with
`python hugging_face_upload/upload_dataset.py` (see [hugging_face_upload/README.md](hugging_face_upload/README.md)).

## Documentation

- [docs/README.md](docs/README.md) — models, covariates, development notes

## Project Structure

Package root is this directory; Python code lives under `tempus_bench/`:

```
tempus_bench/
├── __init__.py                 # Package initialization
├── run_benchmark.py           # Main entry point for benchmarks
├── config/                    # Configuration system
│   ├── benchmark.yaml         # Default configuration
│   ├── settings.yaml          # System configuration
│   ├── models.py              # Model configuration handling
│   └── validator.py           # Configuration validation
├── tasks/                     # Time series datasets (gitignored; auto-downloaded from HF)
│   ├── univariate/           # 10 univariate tasks
│   ├── multivariate/         # 10 multivariate tasks
│   └── covariate/            # 10 covariate tasks
│   # See https://huggingface.co/datasets/Smlcrm/tempus_bench_tasks
├── metrics/                   # Evaluation metrics
│   ├── __init__.py
│   ├── crps.py                # Continuous Ranked Probability Score
│   ├── quantile_score.py
│   ├── weighted_interval_score.py
│   ├── mae.py
│   ├── mape.py
│   ├── mase.py
│   └── rmse.py
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── base_model.py          # Base class for all models
│   ├── arima/                 # ARIMA model
│   ├── lstm/                  # LSTM model
│   ├── xgboost/               # XGBoost model
│   ├── prophet/               # Prophet model
│   ├── chronos/               # Chronos foundation model
│   ├── lagllama/              # LagLlama foundation model
│   ├── moirai/                # Moirai foundation model
│   ├── moirai_moe/            # Moirai MoE foundation model
│   ├── toto/                  # Toto foundation model
│   ├── moment/                # Moment foundation model
│   ├── deepar/                # DeepAR model
│   └── ... (other models)
├── pipeline/                  # Core pipeline components
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── data_types.py         # Data structures and types
│   ├── preprocessor.py       # Data preprocessing
│   ├── model_executor.py     # Model execution in isolated environments
│   ├── hyperparameter_tuner.py
│   └── visualizer.py
├── aggregators/               # Performance aggregation metrics
│   ├── __init__.py
│   ├── base_aggregator.py    # Base class for aggregators
│   ├── win_rate.py           # Average win rate aggregator
│   └── skill_score.py        # Skill score aggregator
└── utils/                     # Utility functions
    ├── __init__.py
    ├── config_manager.py     # Configuration management
    ├── envs.py               # Conda environment management
    ├── log_manager.py        # Unified logging (standard and TensorBoard)
    ├── model_config.py       # Model configuration handling
    └── paths.py              # Path management
```

## Key Features

### 1. Automatic Model Discovery

The framework automatically discovers available models from the models directory. Each model has a `settings.yaml` file that specifies its type (deterministic, stochastic, or hybrid).

All models handle both univariate and multivariate datasets internally.

```python
from tempus_bench.utils import get_available_models

# Get all available models
available_models = get_available_models()
print(available_models)
# Output: {'arima', 'lstm', 'xgboost', 'prophet', 'chronos', 'lagllama', 'moirai', ...}
```

### 2. Unified Model Interface

All models implement a consistent interface through base classes:

- **`BaseModel`**: Base class for all models with standard methods
- **`BaseModel`**: Enhanced base class for stochastic models

```python
from tempus_bench.models.base_model import BaseModel

# Deterministic model implementation
class MyModel(BaseModel):
    def train(self, y_context, y_target, **kwargs):
        # Training implementation
        pass

    def predict(self, y_context, **kwargs):
        # Prediction implementation
        pass

    def compute_metrics(self, y_true, y_pred):
        # Metrics computation
        pass
```

### 3. Comprehensive Data Handling

The pipeline automatically handles:

- **Multiple task formats** (CSV with metadata)
- **Flexible windowing** (context, train, validate splits)
- **Automatic frequency detection** from data
- **Data normalization** (optional)
- **Rolling window evaluation**

```python
from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import EvaluationConfig
from tempus_bench.utils.task_yaml_loader import load_task_config_from_task_dir
from pathlib import Path

task_dir = Path("tempus_bench/tasks/univariate/univariate_climate_daily_mean_humidity_delhi")
task_config = load_task_config_from_task_dir(task_dir)

evaluation_config = EvaluationConfig(
    task_path="univariate/univariate_climate_daily_mean_humidity_delhi",
    max_windows=5,
    max_num_variates=None
)

# Initialize data loader
data_loader = DataLoader(task_config, evaluation_config)

# Generate rolling windows
steps = [
    ('context', 50),
    ('train', 25),
    ('validate', 25)
]
window_iter = data_loader.dataset.generate_dataset_split(
    steps=steps,
    stride=1,
    max_windows=evaluation_config.max_windows,
)

# Iterate over windows
for window_idx, window_splits in window_iter:
    print(f"Window {window_idx}: {window_splits}")
```

### 4. Flexible Configuration

Configuration files support:

- **Model-specific parameters** (hyperparameter grids)
- **Task configuration** (context window, forecast horizon)
- **Evaluation settings** (metrics, loss functions)
- **System configuration** (paths, logging, TensorBoard)

```yaml
# benchmark.yaml
task_path: "*" # Use all tasks

evaluation:
  tuning_loss: mae
  point_forecast_statistic: mean
  max_num_variates: 4
  max_windows: 5

model:
  # Traditional models with hyperparameter grids
  arima:
    p: [1, 2]
    d: [1]
    q: [1, 2]
    s: [2]

  xgboost:
    lookback_window: [30]
    n_estimators: [200]
    max_depth: [4]
    learning_rate: [0.05]

  # Foundation models (no hyperparameters needed)
  chronos: {}
  lagllama: {}
  moirai: {}
```

## Installation

### Prerequisites

- Python 3.11+
- Conda

**Note**: All models added to the `tempus_bench/models` directory must be compatible with Python 3.0 or later (Python 3.x series).

### Setup

1. **Clone the repository** and `cd` into the repo root (this directory):

   ```bash
   git clone https://github.com/Smlcrm/TempusBench.git
   cd TempusBench
   ```

2. **Install the package**:

   ```bash
   pip install -r requirements.txt
   ```

   That editable-installs this tree and pulls runtime deps from `pyproject.toml` (including TensorFlow and TensorBoard), plus pytest and linters from `requirements.txt`. For runtime only: `pip install -e .`. For dev extras from `pyproject.toml` only: `pip install -e ".[dev]"`.

   You can also run **`./install.sh`**, which installs this package in editable mode from the repo root.

3. **Verify installation**:
   ```bash
   python -c "from tempus_bench.utils import get_available_models; print('Installation successful!')"
   ```

## Usage

### Basic Usage via Command Line

```bash
# Activate the conda environment
conda activate sim.benchmarks

# Run benchmark with default configuration
python -m tempus_bench.run_benchmark

# Run with custom configuration
python -m tempus_bench.run_benchmark --config tempus_bench/config/benchmark.yaml

# The system will automatically:
# 1. Discover all available models (deterministic and stochastic)
# 2. Load all tasks from tempus_bench/tasks/
# 3. Perform hyperparameter tuning for each model
# 4. Evaluate models on rolling windows
# 5. Store results in runs/ directory
```

### TensorBoard forecast series (grouped by model)

Event files are written under **`runs/<run_id>/tensorboard/`** (not under a repo-root `tensorboard/` folder). **Actual vs predicted** are logged as **Scalars** only (no PNGs or image summaries):

- **Tag path (model, task, forecast origin, hyperparam trial, variate):**  
  `forecast/<model>/<task>/o<nanoseconds>/h<12-hex-or-default>/v<variate>/{actual|predicted}`  
  **Model** is first so the Scalars sidebar nests under each model. The `o…` segment is the **first validation timestamp** (start of the forecast horizon), as zero-padded pandas-int64-nanoseconds, so ordering by tag matches calendar order. A missing or invalid forecast-start time **raises** (no rolling-index fallback). The `h…` segment is a stable hash of the **hyperparameter dict** for that curve; it is `default` when params are empty (typical foundation runs). **Without** `h…`, multiple grid points that share the same forecast start would write to the same tags and TensorBoard would merge runs into one jagged line.
- **Step:** forecast **horizon index** within that window (`0 … H-1`). The chart x-axis is the position in the forecast, not the rolling-window id.

**Same plot for actual + predicted (recommended):** open the **Custom Scalars** tab. The run includes a layout summary (`custom_scalars__config__`) built like TensorBoard’s official demo: **one category per model** when hyperparameters are empty (foundation-style), or **one category per (model · hyperparam summary)** when tuning, and **one multiline chart per (task, forecast origin, variate)** inside that category. Two tag entries per chart so **actual** and **predicted** render as **two lines on one axes**. Reference: [tensorboard/plugins/custom_scalar/custom_scalar_demo.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/custom_scalar/custom_scalar_demo.py) (see the `wave trig functions` chart with `cosine` + `sine`). The layout is written with `tf.summary.experimental.write_raw_pb` at step `0`, same as that demo. If you still see **empty** charts titled with an old `w0` style, point TensorBoard at a **fresh** `runs/<run_id>/tensorboard` tree or remove stale event files—those panels are from an older layout whose tags no longer exist.

In the plain **Scalars** tab, each tag is still a separate series in the sidebar; overlay there is limited unless you manually compare selections. For hierarchical browsing only, expand `forecast/<your_model>/` and pick task / window / variate tags.

### TensorBoard HParams (hyperparameter × metrics table)

When you sweep hyperparameters (lists in the benchmark `model:` section), each **(model, task, window, param combo)** is logged for the **HParams** dashboard using TensorBoard’s recommended pattern:

1. **`hparams_config`** is written **once** on the root `tensorboard/` writer (declares columns such as `model`, `task`, `window`, `sp`, `theta_method`, `use_reduced_rank`, plus validation **metrics** from the metric registry).
2. Each trial is a **separate sub-run** under **`tensorboard/hparams_sessions/<trial_id>/`**, with **`hp.hparams`**, metric scalars whose **tags match** those metrics (e.g. `mae`, `rmse`), and **`session_end`** so trials appear as distinct rows in the HParams table without overwriting each other.

Point TensorBoard at `runs/<run_id>/tensorboard`, open the **HParams** tab, and select the runs under `hparams_sessions` (or “all”); filter and sort by **`model`** and **`task`** to compare hyperparameter choices. Example config: `tempus_bench/config/local_hparams_tb_demo.yaml`.

### Python API Usage

```python
from tempus_bench.run_benchmark import BenchmarkRunner

# Initialize and run benchmarks
config_path = "tempus_bench/config/benchmark.yaml"
runner = BenchmarkRunner(config_path=config_path)
runner.run()

# The system handles:
# - Hyperparameter optimization
# - Rolling window evaluation
# - Multiple model execution
# - Result storage
```

### Running Individual Models

```python
from tempus_bench.utils.config_manager import ConfigManager
from tempus_bench.pipeline.model_executor import ModelExecutor
from tempus_bench.utils.log_manager import LogManager

# First, create a ConfigManager to load configurations
manager = ConfigManager(
    config_path="tempus_bench/config/benchmark.yaml",
)

# Get a job config
job_config, _ = next(manager.generate_run_configs())

# Initialize executor
executor = ModelExecutor(
    job_config=job_config
)

# Execute a single model with specific hyperparameters
results = executor.execute_model(
    model_name='arima',
    hyperparameters={'p': 2, 'd': 1, 'q': 2, 's': 2},
    context_steps=50,
    train_steps=25,
    validate_steps=25,
    task_path="tempus_bench/tasks/univariate/univariate_climate_daily_mean_humidity_delhi",
    window_idx=0,
    config_path=job_config.config_path
)
```

### Scripts and Automation

Use `python -m tempus_bench.run_benchmark` with the YAML configs under `tempus_bench/config/`, or wrap that command in your own automation. Optional batch shell scripts are not part of the core package API.

### Adding New Models

**Important**: All models must be compatible with Python 3.0 or later (Python 3.x series). Ensure your model implementation and dependencies work with Python 3.x.

1. **Choose model type** (deterministic, stochastic, or hybrid):
   - **Deterministic**: Point forecasts (mean/median predictions)
   - **Stochastic**: Probabilistic forecasts (samples/quantiles)
   - **Hybrid**: Both point forecasts and samples

2. **Create model directory**:

   ```
   tempus_bench/models/my_model/
   ├── my_model_model.py
   ├── requirements.txt
   └── settings.yaml
   ```

3. **Create settings.yaml**:

   ```yaml
   # settings.yaml
   model_type: deterministic # or 'stochastic' or 'hybrid'
   python_version: "3.11.13" # Must be Python 3.0 or later (3.x series)
   ```

   **Note**: The `python_version` specified in `settings.yaml` must be Python 3.0 or later. The framework requires all models to be compatible with Python 3.x series.

4. **Implement model class**:

   ```python
   from tempus_bench.models.base_model import BaseModel

   class MyModelModel(BaseModel):
       def __init__(self, params, settings):
           super().__init__(params, settings)
           # Store model-specific params
           self.params = params

       def train(self, y_context, y_target, timestamps_context, timestamps_target, freq, **kwargs):
           # Training implementation
           # Must return self for method chaining
           return self

       def predict(self, y_context, timestamps_context, timestamps_target, freq, **kwargs):
           # Prediction implementation
           # Return numpy array of predictions
           return predictions

       def compute_metrics(self, y_true, y_pred):
           # Compute evaluation metrics
           return {'mae': mae, 'rmse': rmse}
   ```

5. **Add to configuration**:
   ```yaml
   # In tempus_bench/config/benchmark.yaml
   model:
     my_model:
       param1: [value1, value2]
       param2: [value3, value4]
   ```

The model will be automatically discovered and available for benchmarking!

## Model Categories

### Traditional Models

- **Statistical Models**: ARIMA, Theta, Seasonal Naive
- **Machine Learning**: XGBoost, Random Forest, SVR
- **Deep Learning**: LSTM, DeepAR
- **Ensemble**: TabPFN

### Foundation Models

- **Chronos**: Amazon's time series foundation model
- **LagLlama**: Large language model for time series
- **Moirai**: Microsoft's foundation model for forecasting
- **TimesFM**: Google's time series foundation model
- **Tiny Time Mixer**: Lightweight transformer model
- **Toto**: Multi-modal foundation model

## Evaluation Metrics

The framework supports various evaluation metrics:

- **Point Forecast Metrics**: MAE, RMSE, MAPE, SMAPE
- **Probabilistic Metrics**: CRPS, quantile score (QS), weighted interval score (WIS)
- **Custom Metrics**: Easy to add new evaluation functions

### Performance Aggregation

The framework includes aggregators to summarize model performance across multiple tasks:

- **Win Rate**: For a **single** metric pivot (models × tasks), computes the win rate as the fraction of pairwise comparisons (against other models, on tasks where both have scores) where the model’s error is lower (ties count as 0.5). To summarize multiple metrics with **equal weight per metric**—so dense metrics like MAE do not dominate—use the mean of per-metric win rates:

  ```python
  from tempus_bench.aggregators import WinRate, average_win_rate_across_metrics

  # One pivot per metric: models as rows, tasks as columns, scores as values
  win_rate = WinRate(pivot_table)
  win_rates = win_rate()  # Series per model for that metric only

  combined = average_win_rate_across_metrics(pivot_tables)  # mean across metrics
  ```

- **Skill Score**: Computes skill score for each model compared to a baseline model (default: `seasonal_naive`), quantifying how much a model reduces forecasting error compared to the baseline.

  ```python
  from tempus_bench.aggregators import SkillScore

  # Compute skill score compared to baseline
  skill_score = SkillScore(pivot_table, baseline_model="seasonal_naive")
  skill_scores = skill_score()  # Returns Series with skill scores for each model
  ```

Both aggregators handle missing values (NaN) gracefully and can be extended by implementing the `BaseAggregator` interface.

## Contributing

1. **Follow the project structure** for consistency
2. **Add comprehensive documentation** for new features
3. **Include tests** for new functionality
4. **Use type hints** for better code quality
5. **Follow PEP 8** style guidelines

## Testing

Run these from the **repository root**.

```bash
pytest
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
pytest --cov=tempus_bench
```

## Support

For questions and support:

- Create an issue on GitHub
- See [docs/README.md](docs/README.md)
- Review the test examples for usage patterns

## Roadmap

- [ ] Support for more foundation models
- [ ] Advanced hyperparameter optimization
- [ ] Distributed training capabilities
- [ ] Real-time forecasting pipeline
- [ ] Model interpretability tools
- [ ] Automated model selection

<!-- smlcrm:begin links -->
## Links

- Documentation: [smlcrm.com/docs/tempusbench](https://www.smlcrm.com/docs/tempusbench/) and [docs/README.md](docs/README.md)
- Paper: [arXiv:2604.11529](https://arxiv.org/abs/2604.11529)
- Task data: [Smlcrm/tempus_bench_tasks on Hugging Face](https://huggingface.co/datasets/Smlcrm/tempus_bench_tasks)
- Issues: [Smlcrm/TempusBench/issues](https://github.com/Smlcrm/TempusBench/issues)
- Releases: [Smlcrm/TempusBench/releases](https://github.com/Smlcrm/TempusBench/releases)
- Website: [smlcrm.com](https://smlcrm.com)
- Related repositories: [Smlcrm/Chronax](https://github.com/Smlcrm/Chronax), the JAX forecasting library whose models TempusBench benchmarks
<!-- smlcrm:end links -->

<!-- smlcrm:begin citation -->
## Citation

If you use TempusBench in your work, cite the paper, [arXiv:2604.11529](https://arxiv.org/abs/2604.11529), and the software you used. GitHub's "Cite this repository" button reads the same data from [`CITATION.cff`](CITATION.cff) and offers the paper first.

```bibtex
@misc{smlcrm_tempusbench_paper,
  title         = {TempusBench: An Evaluation Framework for Time-Series Forecasting},
  author        = {Goktas, Denizalp and Riaño-Briceño, Gerardo and Abdullah, Alif and Nair, Aryan and Shen, Chenkai and de Lucio, Beatriz and Magnusson, Alexandra and Mashrur, Farhan and Abdulla, Ahmed and Sen, Shawrna and Thippireddy, Mahitha and Schwartz, Gregory and Greenwald, Amy},
  year          = {2026},
  eprint        = {2604.11529},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.11529}
}
```

The software:

```bibtex
@software{smlcrm_tempusbench,
  title   = {TempusBench},
  author  = {{Simulacrum, Inc.}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Smlcrm/TempusBench}
}
```
<!-- smlcrm:end citation -->

<!-- smlcrm:begin license -->
## License and contact

Released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International license (CC BY-NC-ND 4.0). See [LICENSE](LICENSE).

Contact: [support@smlcrm.com](mailto:support@smlcrm.com) · [smlcrm.com](https://smlcrm.com) · [github.com/Smlcrm](https://github.com/Smlcrm)
<!-- smlcrm:end license -->
