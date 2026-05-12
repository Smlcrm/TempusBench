"""TempusBench adapter for Chronarium-loaded LAFN checkpoints.

Loads any LAFN model that ``chronarium.Chronarium.load_model(...,
model_init="interface")`` can return — local or GCS — and exposes the
``BaseModel.train`` / ``BaseModel.predict`` contract expected by the
benchmark pipeline. Predictions are produced in the model's native
non-autoregressive forward via ``LAFNInterface.sample`` and returned as
a stochastic ``(num_samples, forecast_horizon, num_targets)`` array. The
benchmark pipeline derives deterministic point forecasts from the sample
mean (``point_forecast_statistic="mean"``).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import polars as pl

from tempus_bench.models.base_model import (
    BaseModel,
    PydanticBaseModel,
    validate_covariate_support,
)


class LafnHyperparams(PydanticBaseModel):
    """LAFN is loaded zero-shot from a Chronarium checkpoint, no params to tune."""

    pass


# Synthetic dates fed to ``LAFNInterface``. The interface only uses
# *relative* deltas (in ``base_frequency`` units) between consecutive rows,
# so the absolute calendar value is irrelevant — we just need a strictly
# increasing datetime column with uniform spacing matching the model's
# training cadence.
_SYNTHETIC_START = datetime(2024, 1, 1)


def _resolve_credentials_path(raw_path: Optional[str]) -> Optional[Path]:
    """Resolve ``credentials_path`` against the project / model directory / cwd.

    Settings files in this repo store paths relative to the project root
    (e.g. ``tempus_bench/models/lafn/gcp_credentials.json``). When a
    benchmark is launched from a different cwd we still want to find the
    file, so we fall back to the directory holding this module.
    """

    if raw_path is None:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    here = Path(__file__).resolve().parent
    fallbacks = [
        here / Path(raw_path).name,
        here / Path(raw_path),
    ]
    for f in fallbacks:
        if f.is_file():
            return f.resolve()
    return None


def _build_polars_index(num_steps: int) -> pl.Series:
    """Sequential daily-spaced datetimes; only relative spacing matters."""

    return pl.Series(
        "ts",
        [_SYNTHETIC_START + timedelta(days=i) for i in range(num_steps)],
    )


def _build_input_dataframe(
    *,
    y_context: np.ndarray,
    forecast_horizon: int,
    target_columns: Sequence[str],
    x_context: Optional[np.ndarray],
    x_target: Optional[np.ndarray],
    covariate_columns: Sequence[str],
) -> pl.DataFrame:
    """Assemble the polars ``LAFNInterface`` expects: past targets, future nulls.

    Past rows hold real ``y_context`` values; future rows hold ``None`` for
    every target column so :meth:`LAFNInterface.build_context_dataframes`
    correctly anchors ``forecast_basetime`` to the last context step.
    """

    num_context = y_context.shape[0]
    total_rows = num_context + forecast_horizon
    data: Dict[str, Any] = {"ts": _build_polars_index(total_rows)}

    for col_idx, col in enumerate(target_columns):
        past = y_context[:, col_idx].astype(np.float64).tolist()
        future = [None] * forecast_horizon
        data[col] = past + future

    if covariate_columns:
        if x_context is None or x_target is None:
            raise ValueError(
                "Covariate columns requested but x_context/x_target are missing."
            )
        x_full = np.concatenate(
            [np.asarray(x_context, dtype=np.float64),
             np.asarray(x_target, dtype=np.float64)],
            axis=0,
        )
        for col_idx, col in enumerate(covariate_columns):
            data[col] = x_full[:, col_idx].tolist()

    return pl.DataFrame(data)


def _samples_to_array(
    df: pl.DataFrame,
    *,
    target_columns: Sequence[str],
    forecast_horizon: int,
    num_samples: int,
) -> np.ndarray:
    """Convert :meth:`LAFNInterface.sample` output to ``(S, H, T)`` ndarray."""

    required = {"sample_id", "batch_id", "step"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected sample columns {required!r}, missing {missing!r}; "
            f"got {df.columns!r}"
        )

    df = df.filter(pl.col("batch_id") == 0).sort(["sample_id", "step"])

    sample_arrays = []
    for s in range(num_samples):
        sub = df.filter(pl.col("sample_id") == s)
        if sub.height == 0:
            raise ValueError(
                f"sample_id={s} not found in LAFN sample output "
                f"(num_samples={num_samples})"
            )
        per_target = [sub[c].to_numpy().astype(np.float64) for c in target_columns]
        sample_arrays.append(np.stack(per_target, axis=-1)[:forecast_horizon])

    return np.stack(sample_arrays, axis=0)


class LafnModel(BaseModel):
    """LAFN benchmark adapter (stochastic: empirical samples).

    The Chronarium-loaded ``LAFNInterface`` is built once at ``__init__`` and
    reused across all benchmark windows; ``train`` is a no-op because the
    checkpoint is fully pre-trained.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, LafnHyperparams)

        # Lazy import: keeps benchmark collection fast even when chronarium /
        # JAX are not installed in the parent env (tempus_bench.run_benchmark
        # only imports the model in the per-model conda env).
        from chronarium import Chronarium

        credentials_path = _resolve_credentials_path(
            getattr(self, "credentials_path", None)
        )
        if credentials_path is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

        manager = Chronarium(
            model_repo_path=self.model_repo_path,
            credentials_path=credentials_path,
        )

        self._interface = manager.load_model(
            self.model_name,
            checkpoint_version=getattr(self, "checkpoint_version", None),
            checkpoint_step=getattr(self, "checkpoint_step", None),
            model_init="interface",
        )
        self._interface.eval()

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "LafnModel":
        """Pre-trained model — accept the call and mark fitted."""

        validate_covariate_support(
            x_context,
            x_target,
            supports_past_only=False,
            supports_future_only=False,
            supports_both=False,
            model_name="LAFN",
        )
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return ``(num_samples, forecast_horizon, num_targets)`` LAFN samples.

        The benchmark's ``MetricRegistry`` derives deterministic metrics
        (mae / rmse / ...) from the sample mean when ``model_type =
        stochastic`` and ``point_forecast_statistic = mean``, which matches
        LAFN's analytic mean to within sampling noise for sufficient
        ``num_samples``.
        """

        validate_covariate_support(
            x_context,
            x_target,
            supports_past_only=False,
            supports_future_only=False,
            supports_both=False,
            model_name="LAFN",
        )

        if y_context.ndim != 2:
            raise ValueError(
                f"LafnModel.predict expects 2D y_context (steps, targets); "
                f"got shape {y_context.shape}"
            )

        forecast_horizon = int(timestamps_target.shape[0])
        num_targets = int(y_context.shape[-1])
        target_columns = [f"y{i}" for i in range(num_targets)]

        df = _build_input_dataframe(
            y_context=np.asarray(y_context, dtype=np.float64),
            forecast_horizon=forecast_horizon,
            target_columns=target_columns,
            x_context=None,
            x_target=None,
            covariate_columns=[],
        )

        num_samples = int(kwargs.get("num_samples", 100))
        sample_df = self._interface.sample(
            dataframe=df,
            index_column="ts",
            target_columns=target_columns,
            covariate_columns=[],
            num_samples=num_samples,
            forecast_horizon=forecast_horizon,
        )
        samples = _samples_to_array(
            sample_df,
            target_columns=target_columns,
            forecast_horizon=forecast_horizon,
            num_samples=num_samples,
        )

        return samples
