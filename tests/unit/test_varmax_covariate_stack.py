"""VARMAX: covariates stacked as additional endogenous series."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from tempus_bench.models.varmax.varmax_model import (
    VarmaxHyperparams,
    VarmaxModel,
    build_varmax_endog,
)
from tempus_bench.utils.paths import get_tasks_dir


class TestVarmaxHyperparamsPydantic:
    """Guard against Pydantic v2 'class not fully defined' for Literal trend (conda subprocess)."""

    @pytest.mark.parametrize(
        "trend",
        ["c", "t", "ct", "n"],
    )
    def test_model_validate_accepts_trend_literals(self, trend: str) -> None:
        h = VarmaxHyperparams.model_validate({"p": 1, "q": 0, "trend": trend})
        assert h.trend == trend

    def test_model_validate_default_trend(self) -> None:
        h = VarmaxHyperparams.model_validate({"p": 1, "q": 0})
        assert h.trend == "c"


class TestBuildVarmaxEndog:
    def test_stacks_target_and_covariates(self) -> None:
        y = np.linspace(0, 1, 20).reshape(-1, 1)
        x = np.column_stack([np.linspace(1, 2, 20), np.linspace(-1, 0, 20)])
        endog, n_t = build_varmax_endog(y, x)
        assert n_t == 1
        assert endog.shape == (20, 3)
        np.testing.assert_array_equal(endog[:, 0:1], y)

    def test_no_covariates_requires_two_target_columns(self) -> None:
        y = np.linspace(0, 1, 20).reshape(-1, 1)
        with pytest.raises(ValueError, match="at least two endogenous"):
            build_varmax_endog(y, None)

    def test_two_targets_without_covariates(self) -> None:
        y = np.random.default_rng(0).standard_normal((25, 2))
        endog, n_t = build_varmax_endog(y, None)
        assert n_t == 2
        assert endog.shape == (25, 2)

    def test_row_mismatch_raises(self) -> None:
        y = np.zeros((5, 1))
        x = np.zeros((3, 1))
        with pytest.raises(ValueError, match="row count"):
            build_varmax_endog(y, x)


class TestVarmaxModelCovariateIntegration:
    def test_train_predict_single_target_with_one_covariate(self) -> None:
        rng = np.random.default_rng(42)
        n = 120
        # Correlated bivariate series so VARMAX(1,0) fits reliably
        eps = rng.standard_normal((n, 2)) * 0.15
        y1 = np.cumsum(eps[:, 0]).reshape(-1, 1)
        x1 = 0.5 * y1.squeeze() + eps[:, 1]

        params = {"p": 1, "q": 0, "trend": "n"}
        settings: dict = {"model_type": "deterministic"}
        m = VarmaxModel(params=params, settings=settings)

        ctx = 80
        h = 10
        y_context = y1[:ctx]
        y_target = y1[ctx : ctx + 5]
        x_context = x1[:ctx].reshape(-1, 1)
        x_target = x1[ctx : ctx + 5].reshape(-1, 1)
        ts_c = np.arange(ctx, dtype=np.int64)
        ts_t = np.arange(ctx, ctx + 5, dtype=np.int64)

        m.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_c,
            timestamps_target=ts_t,
            x_context=x_context,
            x_target=x_target,
        )

        y_hist = y1[: ctx + 5]
        x_hist = x1[: ctx + 5].reshape(-1, 1)
        ts_hist = np.arange(ctx + 5, dtype=np.int64)
        ts_future = np.arange(ctx + 5, ctx + 5 + h, dtype=np.int64)

        pred = m.predict(
            y_context=y_hist,
            timestamps_context=ts_hist,
            timestamps_target=ts_future,
            x_context=x_hist,
            x_target=None,
        )
        assert pred.shape == (h, 1)
        assert np.all(np.isfinite(pred))


class TestVarmaxTrendForecastRegression:
    """Regression: old code passed exog during train but not forecast(), causing ValueError
    when trend != 'n'. Current code stacks covariates into endog with exog=None everywhere."""

    @pytest.mark.parametrize("trend", ["c", "t", "ct"])
    def test_forecast_with_trend_and_no_exog(self, trend: str) -> None:
        rng = np.random.default_rng(99)
        n = 80
        y = np.cumsum(rng.standard_normal((n, 2)), axis=0)
        ctx, h = 60, 10
        m = VarmaxModel(
            params={"p": 1, "q": 0, "trend": trend},
            settings={"model_type": "deterministic"},
        )
        ts_c = np.arange(ctx, dtype=np.int64)
        ts_t = np.arange(ctx, ctx + 5, dtype=np.int64)
        m.train(
            y_context=y[:ctx],
            y_target=y[ctx : ctx + 5],
            timestamps_context=ts_c,
            timestamps_target=ts_t,
        )
        pred = m.predict(
            y_context=y[: ctx + 5],
            timestamps_context=np.arange(ctx + 5, dtype=np.int64),
            timestamps_target=np.arange(ctx + 5, ctx + 5 + h, dtype=np.int64),
        )
        assert pred.shape == (h, 2)
        assert np.all(np.isfinite(pred))

    def test_forecast_with_trend_t_and_covariate(self) -> None:
        rng = np.random.default_rng(77)
        n = 120
        eps = rng.standard_normal((n, 2)) * 0.15
        y1 = np.cumsum(eps[:, 0]).reshape(-1, 1)
        x1 = 0.3 * y1.squeeze() + eps[:, 1]
        ctx, h = 80, 10

        m = VarmaxModel(
            params={"p": 1, "q": 0, "trend": "t"},
            settings={"model_type": "deterministic"},
        )
        ts_c = np.arange(ctx, dtype=np.int64)
        ts_t = np.arange(ctx, ctx + 5, dtype=np.int64)
        m.train(
            y_context=y1[:ctx],
            y_target=y1[ctx : ctx + 5],
            timestamps_context=ts_c,
            timestamps_target=ts_t,
            x_context=x1[:ctx].reshape(-1, 1),
            x_target=x1[ctx : ctx + 5].reshape(-1, 1),
        )
        pred = m.predict(
            y_context=y1[: ctx + 5],
            timestamps_context=np.arange(ctx + 5, dtype=np.int64),
            timestamps_target=np.arange(ctx + 5, ctx + 5 + h, dtype=np.int64),
            x_context=x1[: ctx + 5].reshape(-1, 1),
        )
        assert pred.shape == (h, 1)
        assert np.all(np.isfinite(pred))


class TestVarmaxNonstationaryPanelRegression:
    """Batch regression: default VARMAX init used to raise LinAlgError on some multivariate tasks."""

    def test_batadal_software_window_fits_with_approximate_diffuse(self) -> None:
        csv_path = (
            get_tasks_dir()
            / "multivariate"
            / "batadal_software_multivariate"
            / "batadal_software_multivariate.csv"
        )
        if not csv_path.is_file():
            pytest.skip(f"task CSV not present: {csv_path}")
        df = pl.read_csv(csv_path)
        targets = df.filter(pl.col("variable_type") == "target")
        vals = [json.loads(r) for r in targets["values"].to_list()]
        y = np.asarray(vals, dtype=float).T[:, :10]
        # Shorter span than cloud tasks keeps CI fast; still hit Schur/PD failures with stationary init.
        cend, tend, h = 256, 320, 32
        y_context = y[:cend]
        y_target = y[cend:tend]
        m = VarmaxModel(
            params={"p": 1, "q": 0, "trend": "c"},
            settings={"model_type": "deterministic"},
        )
        ts_c = np.arange(y_context.shape[0], dtype=np.int64)
        ts_t = np.arange(y_target.shape[0], dtype=np.int64)
        m.train(
            y_context=y_context,
            y_target=y_target,
            timestamps_context=ts_c,
            timestamps_target=ts_t,
        )
        pred = m.predict(
            y_context=y[:tend],
            timestamps_context=np.arange(tend, dtype=np.int64),
            timestamps_target=np.arange(h, dtype=np.int64),
        )
        assert pred.shape == (h, 10)
        assert np.all(np.isfinite(pred))
