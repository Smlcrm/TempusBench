"""TT13 / TT14: horizon & context contracts; validation edges (per model).

Requires ``RUN_MODEL_MATRIX_TEST=1``. Uses the same weight / offline fixtures as the matrix.

Restrict to one package per run with ``MODEL_MATRIX_ONLY=<package_id>`` (see
``run_model_matrix_one_by_one.sh``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from tests.integration.model_matrix_support import (
    MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON,
    build_predict_kwargs,
    build_train_kwargs,
    default_params_from_settings,
    find_model_class_or_skip,
    load_settings,
    make_covariate,
    make_univariate,
    matrix_dimensions,
    model_type_from_settings,
    require_hf_weights_if_configured,
    run_train_predict_or_skip,
    skip_if_excluded,
    sorted_model_packages,
    suggested_context_cap,
    supports_univariate,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_MATRIX_TEST", "") != "1",
        reason="Set RUN_MODEL_MATRIX_TEST=1 for context/horizon edge tests.",
    ),
    pytest.mark.model_matrix,
]


@pytest.fixture(autouse=True)
def _matrix_hf_offline(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "matrix_hf_edges"
    for d in ("hf_home", "hf_hub", "transformers", "datasets", "tabpfn_cache"):
        (run / d).mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(run / "hf_home"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(run / "hf_hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(run / "transformers"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(run / "datasets"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(run / "tabpfn_cache"))


@pytest.mark.parametrize("package", sorted_model_packages())
def test_predict_rejects_empty_timestamps_target(package: str) -> None:
    skip_if_excluded(package)
    settings = load_settings(package)
    if not supports_univariate(settings):
        pytest.skip(f"{package}: univariate edge uses single-target path")
    require_hf_weights_if_configured(package, settings)

    ctx, tr, hz = matrix_dimensions()
    if hz < 2:
        pytest.skip("need hz>=2 to train then swap empty ts")
    rng = np.random.default_rng(42)
    data = make_univariate(rng, ctx_steps=ctx, train_steps=tr, forecast_horizon=hz)

    params = default_params_from_settings(settings, package=package)
    Model = find_model_class_or_skip(package)
    model = Model(params, settings)
    moir_vh = hz if package in MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON else None
    tk = build_train_kwargs(
        data, package=package, num_samples=4, moirai_validate_horizon=moir_vh
    )
    try:
        model.train(
            data["y_ctx"],
            data["y_train"],
            data["ts_ctx"],
            data["ts_train"],
            **tk,
        )
    except ImportError as exc:
        pytest.skip(f"{package}: train ImportError: {exc}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"{package}: train ModuleNotFoundError: {exc}")
    y_full = np.concatenate([data["y_ctx"], data["y_train"]], axis=0)
    ts_full = np.concatenate([data["ts_ctx"], data["ts_train"]], axis=0)
    empty_ts = np.array([], dtype=np.int64)
    pk = build_predict_kwargs(data, package=package, num_samples=4)
    if package == "lstm":
        pk.pop("num_samples", None)
    elif model_type_from_settings(settings) == "deterministic":
        pk.pop("num_samples", None)

    with pytest.raises(ValueError, match="timestamps_target cannot be empty"):
        model.predict(y_full, ts_full, empty_ts, **pk)


@pytest.mark.parametrize("package", sorted_model_packages())
def test_covariate_future_shape_mismatch_raises(package: str) -> None:
    skip_if_excluded(package)
    settings = load_settings(package)
    caps = settings.get("capabilities") or {}
    if str(caps.get("covariates", "none")) != "past_future":
        pytest.skip(f"{package}: past_future covariates required for mismatch case")
    if not supports_univariate(settings):
        pytest.skip(f"{package}: needs univariate target column")

    require_hf_weights_if_configured(package, settings)
    ctx, tr, hz = matrix_dimensions()
    rng = np.random.default_rng(43)
    data = make_covariate(rng, ctx_steps=ctx, train_steps=tr, forecast_horizon=hz, past_only=False)
    params = default_params_from_settings(settings, package=package)
    Model = find_model_class_or_skip(package)
    model = Model(params, settings)
    moir_vh = hz if package in MOIRAI_PACKAGES_WITH_VALIDATE_HORIZON else None
    tk = build_train_kwargs(
        data, package=package, num_samples=4, moirai_validate_horizon=moir_vh
    )
    try:
        model.train(
            data["y_ctx"],
            data["y_train"],
            data["ts_ctx"],
            data["ts_train"],
            **tk,
        )
    except ImportError as exc:
        pytest.skip(f"{package}: train ImportError: {exc}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"{package}: train ModuleNotFoundError: {exc}")
    y_full = np.concatenate([data["y_ctx"], data["y_train"]], axis=0)
    ts_full = np.concatenate([data["ts_ctx"], data["ts_train"]], axis=0)
    bad_x_future = data["x_target"][: max(1, hz - 1), :]
    pk = build_predict_kwargs(data, package=package, num_samples=4)
    pk["x_target"] = bad_x_future
    if package == "lstm":
        pk.pop("num_samples", None)
    elif model_type_from_settings(settings) == "deterministic":
        pk.pop("num_samples", None)

    with pytest.raises(ValueError):
        model.predict(y_full, ts_full, data["ts_target"], **pk)


@pytest.mark.parametrize("package", sorted_model_packages())
def test_oversized_context_still_matches_horizon(package: str) -> None:
    """If settings cap context, feed extraHistory; output horizon must still match request."""
    skip_if_excluded(package)
    settings = load_settings(package)
    if not supports_univariate(settings):
        pytest.skip(f"{package}: univariate only for this smoke")

    cap = suggested_context_cap(settings)
    if cap is None:
        pytest.skip(f"{package}: no context_length / lookback hint in settings")

    require_hf_weights_if_configured(package, settings)

    _, _, hz = matrix_dimensions()
    ctx = min(cap + 512, 50_000)
    tr = max(matrix_dimensions()[1], 16)
    rng = np.random.default_rng(44)
    data = make_univariate(rng, ctx_steps=ctx, train_steps=tr, forecast_horizon=hz)
    out, _, _, st = run_train_predict_or_skip(package, data)
    mt = st.get("model_type", "deterministic")
    if mt == "deterministic":
        assert out.shape[0] == hz
    else:
        assert out.shape[1] == hz
