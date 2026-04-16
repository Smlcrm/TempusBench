"""Bucket-local inference smoke tests (FUSE-style ``MODEL_WEIGHTS_PATH``).

Requires:
  - ``RUN_LOCAL_WEIGHTS_TEST=1``
  - ``MODEL_WEIGHTS_PATH`` pointing at a directory whose children include each
    model's ``hf_model_name`` path (e.g. ``.../ibm-granite/granite-timeseries-ttm-r2/``).

Uses empty Hugging Face caches per test via ``tmp_path`` so runs do not pick up
weights from a pre-warmed conda environment.

Run (example):
  RUN_DIR=$(mktemp -d) && source path/to/run_local_weights_env.sh "$RUN_DIR" \\
    && export MODEL_WEIGHTS_PATH=/path/to/rsync/root \\
    && RUN_LOCAL_WEIGHTS_TEST=1 pytest tests/integration/test_local_weights_inference.py -q
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml


def _models_dir() -> Path:
    import tempus_bench.models as m

    return Path(m.__path__[0])


def _load_settings(package: str) -> dict[str, Any]:
    path = _models_dir() / package / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_candidate_ok(name: str) -> bool:
    if not name.endswith("Model") or name == "BaseModel":
        return False
    # e.g. PydanticBaseModel — not a TempusBench forecaster
    if name.endswith("BaseModel"):
        return False
    return True


def _find_model_class(package: str):
    """Resolve ``FooModel`` from package ``__init__`` or ``{package}_model``."""
    mod = importlib.import_module(f"tempus_bench.models.{package}")
    for name in dir(mod):
        if not _model_candidate_ok(name):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == mod.__name__:
            return obj
    sub = importlib.import_module(
        f"tempus_bench.models.{package}.{package}_model"
    )
    for name in dir(sub):
        if not _model_candidate_ok(name):
            continue
        obj = getattr(sub, name)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == sub.__name__:
            return obj
    raise ValueError(f"No *Model class in tempus_bench.models.{package}")


def _weights_root() -> str:
    return os.environ.get("MODEL_WEIGHTS_PATH", "").strip()


def _require_local_weights(hf_id: str) -> Path:
    root = _weights_root()
    local = Path(root) / hf_id
    if not local.is_dir() or not any(local.iterdir()):
        pytest.skip(f"Missing non-empty weights dir for {hf_id!r}: {local}")
    return local


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_WEIGHTS_TEST", "") != "1",
    reason="Set RUN_LOCAL_WEIGHTS_TEST=1 to run bucket-local inference tests.",
)


@pytest.fixture(autouse=True)
def _isolated_hf_hub_caches(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not read/write the developer's default Hugging Face caches."""
    run = tmp_path / "isolated_run"
    (run / "hf_home").mkdir(parents=True)
    (run / "hf_hub").mkdir(parents=True)
    (run / "transformers").mkdir(parents=True)
    (run / "datasets").mkdir(parents=True)
    (run / "tabpfn_cache").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(run / "hf_home"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(run / "hf_hub"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(run / "transformers"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(run / "datasets"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(run / "tabpfn_cache"))


@pytest.mark.parametrize(
    "package",
    [
        "tiny_time_mixer_r1",
        "tiny_time_mixer_r2",
        "tiny_time_mixer_r2_1",
    ],
)
def test_tiny_time_mixer_local_weights(package: str) -> None:
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings.get("hf_model_name")
    assert isinstance(hf_id, str) and "/" in hf_id
    _require_local_weights(hf_id)

    Model = _find_model_class(package)
    model = Model({}, settings)

    ctx, horizon = 512, 16
    rng = np.random.default_rng(0)
    y = rng.standard_normal((ctx, 1)).astype(np.float64)
    day_ms = 86_400_000
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_t = (np.arange(ctx, ctx + horizon, dtype=np.int64) * day_ms).astype(np.int64)

    out = model.predict(y, ts_c, ts_t)
    assert out.shape == (horizon, 1)


def _day_ms_timestamps(ctx: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    day_ms = 86_400_000
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_t = (np.arange(ctx, ctx + horizon, dtype=np.int64) * day_ms).astype(np.int64)
    return ts_c, ts_t


@pytest.mark.parametrize("package", ["timesfm_200m"])
def test_timesfm_200m_local_weights(package: str) -> None:
    pytest.importorskip("timesfm")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    day_ms = 86_400_000
    ctx = max(200, int(settings.get("context_len", 128)))
    train_span = 16
    horizon = 24
    rng = np.random.default_rng(1)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((train_span, 1)).astype(np.float64)
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_train_target = ts_c[-1] + (
        np.arange(1, train_span + 1, dtype=np.int64) * day_ms
    )
    ts_predict = ts_train_target[-1] + (
        np.arange(1, horizon + 1, dtype=np.int64) * day_ms
    )

    x_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    x_t_train = rng.standard_normal((train_span, 1)).astype(np.float64)
    x_t_pred = rng.standard_normal((horizon, 1)).astype(np.float64)

    model.train(
        y_c,
        y_t,
        ts_c,
        ts_train_target,
        x_context=x_c,
        x_target=x_t_train,
    )
    out = model.predict(
        y_c,
        ts_c,
        ts_predict,
        x_context=x_c,
        x_target=x_t_pred,
    )
    assert out.shape[0] == horizon


@pytest.mark.parametrize("package", ["lagllama"])
def test_lagllama_local_weights(package: str) -> None:
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    ctx_len = int(settings.get("context_length", 256))
    horizon = 12
    ctx = max(ctx_len, 256)
    rng = np.random.default_rng(2)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((horizon, 1)).astype(np.float64)
    day_ms = 86_400_000
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_t = ts_c[-1] + (np.arange(1, horizon + 1, dtype=np.int64) * day_ms)

    model.train(
        y_c,
        y_t,
        ts_c,
        ts_t,
        num_samples=4,
    )
    out = model.predict(
        y_c, ts_c, ts_t, freq="h", num_samples=4,
    )
    assert out.shape[1] == horizon


@pytest.mark.parametrize("package", ["tabpfn"])
def test_tabpfn_local_weights(package: str) -> None:
    pytest.importorskip("tabpfn")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    ctx, horizon = 128, 8
    rng = np.random.default_rng(3)
    y_c = rng.standard_normal((ctx, 2)).astype(np.float64)
    y_t = rng.standard_normal((4, 2)).astype(np.float64)
    ts_c, ts_t = _day_ms_timestamps(ctx, horizon)
    x_c = rng.standard_normal((ctx, 1)).astype(np.float64)

    model.train(y_c, y_t, ts_c, ts_t, x_context=x_c)
    out = model.predict(y_c, ts_c, ts_t, x_context=x_c)
    assert out.shape == (horizon, 2)


@pytest.mark.parametrize("package", ["moirai2"])
def test_moirai2_local_weights(package: str) -> None:
    pytest.importorskip("uni2ts")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    day_ms = 86_400_000
    ctx, train_h, pred_h = 96, 24, 12
    rng = np.random.default_rng(4)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((train_h, 1)).astype(np.float64)
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_train = ts_c[-1] + (np.arange(1, train_h + 1, dtype=np.int64) * day_ms)

    model.train(y_c, y_t, ts_c, ts_train, num_samples=8)

    y_full = np.concatenate([y_c, y_t], axis=0)
    ts_full = np.concatenate([ts_c, ts_train], axis=0)
    ts_pred = ts_train[-1] + (np.arange(1, pred_h + 1, dtype=np.int64) * day_ms)
    out = model.predict(y_full, ts_full, ts_pred, num_samples=8)
    assert out.ndim == 3 and out.shape[0] == 8 and out.shape[2] == 1
    # Forward horizon can be slightly shorter than ``len(ts_pred)`` (patch / mask alignment).
    assert 1 <= out.shape[1] <= pred_h


@pytest.mark.parametrize("package", ["moirai_moe"])
def test_moirai_moe_local_weights(package: str) -> None:
    pytest.importorskip("uni2ts")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    ctx, train_h, pred_h = 64, 16, 8
    rng = np.random.default_rng(5)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((train_h, 1)).astype(np.float64)
    day_ms = 86_400_000
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_yt = (np.arange(train_h, dtype=np.int64) * day_ms).astype(np.int64) + (
        ts_c[-1] + day_ms
    )
    ts_tgt = (np.arange(pred_h, dtype=np.int64) * day_ms).astype(np.int64) + (
        ts_yt[-1] + day_ms
    )

    model.train(
        y_c,
        y_t,
        ts_c,
        ts_yt,
        num_samples=4,
    )
    full_y = np.concatenate([y_c, y_t], axis=0)
    full_ts = np.concatenate([ts_c, ts_yt], axis=0)
    out = model.predict(
        full_y,
        full_ts,
        ts_tgt,
        num_samples=4,
    )
    assert out.ndim == 3 and out.shape[0] == 4 and out.shape[2] == 1
    assert 1 <= out.shape[1] <= pred_h


@pytest.mark.parametrize("package", ["sundial"])
def test_sundial_local_weights(package: str) -> None:
    pytest.importorskip("transformers")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    lb = int(settings.get("lookback_length", 128))
    horizon = min(24, int(settings.get("forecast_length", 50)))
    ctx = max(lb, 64)
    train_h = 8
    rng = np.random.default_rng(6)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((train_h, 1)).astype(np.float64)
    day_ms = 86_400_000
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ms).astype(np.int64)
    ts_train = ts_c[-1] + (
        np.arange(1, train_h + 1, dtype=np.int64) * day_ms
    )
    ts_t = ts_train[-1] + (np.arange(1, horizon + 1, dtype=np.int64) * day_ms)
    x_c = rng.standard_normal((ctx, 1)).astype(np.float64)

    model.train(y_c, y_t, ts_c, ts_train, x_context=x_c)
    out = model.predict(
        y_c, ts_c, ts_t, x_context=x_c, num_samples=4,
    )
    assert out.shape[1] == horizon
    assert out.shape[2] == 1


@pytest.mark.parametrize("package", ["tirex", "tirex_1_1_gifteval"])
def test_tirex_local_weights(package: str) -> None:
    pytest.importorskip("tirex")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    package = str(package)
    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    ctx, horizon = 128, 8
    rng = np.random.default_rng(7)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((8, 1)).astype(np.float64)
    ts_c, ts_t = _day_ms_timestamps(ctx, horizon)
    x_c = rng.standard_normal((ctx, 1)).astype(np.float64)

    model.train(y_c, y_t, ts_c, ts_t, x_context=x_c)
    out = model.predict(
        y_c, ts_c, ts_t, x_context=x_c, num_samples=8,
    )
    assert out.shape[1] == horizon
    assert out.shape[2] == 1


@pytest.mark.parametrize("package", ["toto"])
def test_toto_local_weights(package: str) -> None:
    pytest.importorskip("transformers")
    root = _weights_root()
    if not root:
        pytest.skip("MODEL_WEIGHTS_PATH is not set")

    settings = _load_settings(package)
    hf_id = settings["hf_model_name"]
    _require_local_weights(str(hf_id))

    Model = _find_model_class(package)
    model = Model({}, settings)

    ctx, train_h, horizon = 96, 8, 12
    rng = np.random.default_rng(8)
    y_c = rng.standard_normal((ctx, 1)).astype(np.float64)
    y_t = rng.standard_normal((train_h, 1)).astype(np.float64)
    day_ns = 86_400_000_000_000
    ts_c = (np.arange(ctx, dtype=np.int64) * day_ns).astype(np.int64)
    ts_train = ts_c[-1] + (np.arange(1, train_h + 1, dtype=np.int64) * day_ns)
    ts_t = ts_train[-1] + (np.arange(1, horizon + 1, dtype=np.int64) * day_ns)

    model.train(y_c, y_t, ts_c, ts_train)
    out = model.predict(y_c, ts_c, ts_t, freq="d", num_samples=4)
    assert out.ndim == 3 and out.shape[0] == 4 and out.shape[2] == 1
    assert out.shape[1] == horizon