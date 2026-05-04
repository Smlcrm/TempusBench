import functools
import inspect
import re
import textwrap
import numpy as np
import pandas as pd

from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ttm import TinyTimeMixerForecaster

from tempus_bench.models.base_model import (
    BaseModel,
    validate_inputs,
    validate_covariate_support,
)
from tempus_bench.utils.sktime_datetime_freq import (
    infer_pandas_freq_offset_for_datetime_index,
)


def _load_ttm_context_length_from_pretrained(
    model_path: str, revision: str,
) -> int:
    """Read ``context_length`` from Hugging Face config (sktime TTM requires an exact sequence length)."""
    from sktime.libs.granite_ttm import TinyTimeMixerConfig

    config = TinyTimeMixerConfig.from_pretrained(
        model_path,
        revision=revision,
    )
    return int(config.context_length)


@functools.lru_cache(maxsize=16)
def _get_ttm_context_length(model_path: str, revision: str) -> int:
    ctx = _load_ttm_context_length_from_pretrained(model_path, revision)
    if ctx <= 0:
        raise ValueError(
            f"TinyTimeMixer config at {model_path!r} (revision={revision!r}) "
            f"has invalid context_length={ctx}"
        )
    return ctx


def _truncate_ttm_aligned_history(
    y_context: np.ndarray,
    timestamps_context: np.ndarray,
    x_context: Optional[np.ndarray],
    *,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Keep only the last ``max_len`` rows of ``y`` / timestamps / past covariates (aligned)."""
    n_rows = int(y_context.shape[0])
    ts = np.asarray(timestamps_context)
    if ts.ndim != 1:
        raise ValueError(
            f"timestamps_context must be 1D, got shape {ts.shape}"
        )
    if ts.shape[0] != n_rows:
        raise ValueError(
            f"timestamps_context length ({ts.shape[0]}) must match y_context rows ({n_rows})"
        )
    if x_context is not None:
        xc = np.asarray(x_context)
        if xc.shape[0] != n_rows:
            raise ValueError(
                f"x_context rows ({xc.shape[0]}) must match y_context rows ({n_rows})"
            )
    if n_rows <= max_len:
        return y_context, timestamps_context, x_context
    start = n_rows - max_len
    y2 = y_context[start:]
    ts2 = ts[start:]
    x2 = None if x_context is None else np.asarray(x_context)[start:]
    return y2, ts2, x2


_TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = False
_SKTIME_TTM_FH_PATCH_DONE = False
_SKTIME_TTM_FIT_MISMATCHED_KEYS_PATCH_DONE = False
_SKTIME_TTM_FREQ_TOKEN_PATCH_DONE = False

# sktime TinyTimeMixerForecaster._predict: the non-MultiIndex branch builds
# ``ForecastingHorizon(range(...))`` without ``freq`` while MultiIndex uses ``freq=self.fh``.
# ``to_absolute`` then hits ``to_offset(fh.freq)`` with ``freq is None`` (Batch / cloud logs:
# _fh.py ``_to_offset``). Patch the bound method source once per process.
_TTM_PREDICT_ELSE_FH_PATTERN = re.compile(
    r"(index = \(\s*\n)"
    r"(\s*)ForecastingHorizon\(range\(1, pred\.shape\[1\] \+ 1\)\)\s*\n"
    r"(\s*)\.to_absolute\(self\._cutoff\)\s*\n"
    r"(\s*)\._values\s*\n"
    r"(\s*\))",
    re.MULTILINE,
)


def _fix_sktime_ttm_predict_source_for_tests(src: str) -> tuple[str, int]:
    """Replace else-branch ForecastingHorizon to include ``freq=self.fh.freq``. For unit tests."""
    out, n = _TTM_PREDICT_ELSE_FH_PATTERN.subn(
        lambda m: (
            f"{m.group(1)}"
            f"{m.group(2)}ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)\n"
            f"{m.group(3)}.to_absolute(self._cutoff)\n"
            f"{m.group(4)}._values\n"
            f"{m.group(5)}"
        ),
        src,
        count=1,
    )
    return out, n


def _patch_sktime_ttm_predict_forecasting_horizon_freq() -> None:
    """Monkeypatch ``TinyTimeMixerForecaster._predict`` so the non-MultiIndex path passes ``fh.freq``."""
    global _SKTIME_TTM_FH_PATCH_DONE
    if _SKTIME_TTM_FH_PATCH_DONE:
        return
    import sktime.forecasting.ttm as ttm_mod
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    src = textwrap.dedent(
        "".join(inspect.getsourcelines(TinyTimeMixerForecaster._predict)[0])
    )
    patched, n = _fix_sktime_ttm_predict_source_for_tests(src)
    if n != 1:
        if _TTM_PREDICT_ELSE_FH_PATTERN.search(src) is None:
            # Else branch already uses freq (fixed upstream) or layout changed.
            m_alt = re.search(
                r"else:\s*\n\s*index = \(\s*\n\s*ForecastingHorizon\(range\(1, pred\.shape\[1\] \+ 1\), freq=self\.fh\.freq\)",
                src,
            )
            if m_alt is not None:
                _SKTIME_TTM_FH_PATCH_DONE = True
                return
        raise RuntimeError(
            "TempusBench could not patch sktime TinyTimeMixerForecaster._predict "
            "(non-MultiIndex ForecastingHorizon freq). Check sktime version."
        )

    local: dict[str, Any] = {}
    exec(
        compile(patched, "<TinyTimeMixerForecaster._predict tempusbench>", "exec"),
        ttm_mod.__dict__,
        local,
    )
    TinyTimeMixerForecaster._predict = local["_predict"]  # type: ignore[assignment]
    _SKTIME_TTM_FH_PATCH_DONE = True


def _patch_sktime_ttm_fit_mismatched_keys() -> None:
    """Monkeypatch ``TinyTimeMixerForecaster._fit`` so the ``mismatched_keys``
    iteration handles tuples returned by ``transformers.from_pretrained``.

    ``_fit`` iterates ``info["mismatched_keys"]`` calling ``key.split(".")``
    but transformers returns tuples ``(key_str, shape_model, shape_ckpt)`` —
    causing ``AttributeError: 'tuple' object has no attribute 'split'``.

    We wrap ``_fit`` to normalise ``mismatched_keys`` to plain strings *before*
    the original code runs.
    """
    global _SKTIME_TTM_FIT_MISMATCHED_KEYS_PATCH_DONE
    if _SKTIME_TTM_FIT_MISMATCHED_KEYS_PATCH_DONE:
        return
    from sktime.forecasting.ttm import TinyTimeMixerForecaster
    import functools

    _orig_fit = TinyTimeMixerForecaster._fit

    @functools.wraps(_orig_fit)
    def _fit_with_normalised_keys(self, y, X=None, fh=None):
        _orig_from_pt = None
        try:
            _TinyTimeMixer = type(None)
            try:
                from sktime.libs.granite_ttm import TinyTimeMixerForPrediction
                _TinyTimeMixer = TinyTimeMixerForPrediction
            except ImportError:
                from transformers import TinyTimeMixerForPrediction  # type: ignore[no-redef]
                _TinyTimeMixer = TinyTimeMixerForPrediction

            _orig_from_pt = _TinyTimeMixer.from_pretrained.__func__  # type: ignore[union-attr]

            @classmethod  # type: ignore[misc]
            @functools.wraps(_orig_from_pt)
            def _fixed_from_pt(cls, *a, **kw):
                result = _orig_from_pt(cls, *a, **kw)
                if isinstance(result, tuple) and len(result) == 2:
                    model, info = result
                    if isinstance(info, dict) and "mismatched_keys" in info:
                        info["mismatched_keys"] = [
                            k[0] if isinstance(k, tuple) else k
                            for k in info["mismatched_keys"]
                        ]
                    return model, info
                return result

            _TinyTimeMixer.from_pretrained = _fixed_from_pt  # type: ignore[assignment]
        except Exception:
            _orig_from_pt = None

        try:
            return _orig_fit(self, y, X=X, fh=fh)
        finally:
            if _orig_from_pt is not None:
                _TinyTimeMixer.from_pretrained = classmethod(_orig_from_pt)  # type: ignore[assignment]

    TinyTimeMixerForecaster._fit = _fit_with_normalised_keys
    _SKTIME_TTM_FIT_MISMATCHED_KEYS_PATCH_DONE = True


def _patch_sktime_ttm_freq_token_default() -> None:
    """Monkeypatch the TTM encoder ``forward`` to supply a default ``freq_token``
    when ``resolution_prefix_tuning`` is enabled but sktime doesn't pass one.

    Without this patch the encoder raises
    ``Exception("Expecting freq_token in forward")``.  We inject a zero tensor
    (the first vocabulary entry) so the resolution-prefix pathway runs and all
    weight shapes (``num_patches=9``) remain valid.
    """
    global _SKTIME_TTM_FREQ_TOKEN_PATCH_DONE
    if _SKTIME_TTM_FREQ_TOKEN_PATCH_DONE:
        return

    try:
        from sktime.libs.granite_ttm.modeling_tinytimemixer import (
            TinyTimeMixerEncoder,
        )
    except ImportError:
        _SKTIME_TTM_FREQ_TOKEN_PATCH_DONE = True
        return

    import functools
    import torch

    _orig_encoder_forward = TinyTimeMixerEncoder.forward

    @functools.wraps(_orig_encoder_forward)
    def _forward_with_default_freq(self, past_values, output_hidden_states=False,
                                   return_dict=None, freq_token=None):
        if freq_token is None and getattr(self, "resolution_prefix_tuning", False):
            freq_token = torch.zeros(
                past_values.shape[0], dtype=torch.long, device=past_values.device,
            )
        return _orig_encoder_forward(
            self, past_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
        )

    TinyTimeMixerEncoder.forward = _forward_with_default_freq
    _SKTIME_TTM_FREQ_TOKEN_PATCH_DONE = True


def _patch_transformers_tiny_time_mixer_tied_weights() -> None:
    """Avoid ``from_pretrained`` failure when ``mark_tied_weights_as_initialized`` looks up
    tied keys that are not registered ``nn.Parameter`` leaves (IBM / Granite TTM + recent
    ``transformers``). Batch logs: AttributeError in ``Module.__getattr__`` from
    ``get_parameter`` inside ``mark_tied_weights_as_initialized`` (line ~4613).
    """
    global _TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE
    if _TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE:
        return
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
        # Newer ``transformers`` removed this hook; TTM loading no longer hits that path.
        _TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = True
        return

    def _tied_weights_key_map(model: Any) -> dict:
        """``PreTrainedModel`` sets ``all_tied_weights_keys`` in ``__init__``; remote-code
        models (e.g. Granite TTM) may only define class-level ``_tied_weights_keys``."""
        expanded = getattr(model, "all_tied_weights_keys", None)
        if isinstance(expanded, dict):
            return expanded
        legacy = getattr(model, "_tied_weights_keys", None)
        if isinstance(legacy, dict):
            return legacy
        return {}

    def _mark_tied_weights_safe(self, loading_info):
        tied = _tied_weights_key_map(self)
        # Remote-code TTM may omit instance ``all_tied_weights_keys``; ``transformers``
        # still reads it in ``_move_missing_keys_from_meta_to_device`` after this hook.
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = dict(tied)
        for tied_param in list(tied.keys()):
            try:
                param = self.get_parameter(tied_param)
            except AttributeError:
                continue
            param._is_hf_initialized = True
        if self.is_remote_code():
            def _has_hf_init_flag(key: str) -> bool:
                try:
                    obj = self.get_parameter_or_buffer(key)
                except AttributeError:
                    return False
                return bool(getattr(obj, "_is_hf_initialized", False))

            loading_info.missing_keys = {
                key
                for key in loading_info.missing_keys
                if key in tied or not _has_hf_init_flag(key)
            }

    PreTrainedModel.mark_tied_weights_as_initialized = _mark_tied_weights_safe
    _TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = True


class TinyTimeMixerR1Hyperparams(PydanticBaseModel):
    pass


class TinyTimeMixerR1Model(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TinyTimeMixer model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, TinyTimeMixerR1Hyperparams)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "TinyTimeMixerR1Model":
        """
        Train/fine-tune the foundation model on given data.

        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        validate_covariate_support(
            x_context,
            x_target,
            supports_past_only=True,
            supports_future_only=True,
            supports_both=True,
            model_name="TinyTimeMixer",
        )
        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ):
        validate_covariate_support(
            x_context,
            x_target,
            supports_past_only=True,
            supports_future_only=True,
            supports_both=True,
            model_name="TinyTimeMixer",
        )
        model_path = getattr(self, "hf_model_name", None) or getattr(
            self, "model_path", "ibm/TTM"
        )
        revision = getattr(self, "revision", "main")
        ttm_context_len = _get_ttm_context_length(model_path, revision)
        y_context, timestamps_context, x_context = _truncate_ttm_aligned_history(
            y_context,
            timestamps_context,
            x_context,
            max_len=ttm_context_len,
        )
        num_targets = y_context.shape[1]
        forecast_horizon = timestamps_target.shape[0]
        columns = list(range(num_targets))
        timestamps_context = self._convert_to_datetimeindex(timestamps_context)
        if not isinstance(timestamps_context, pd.DatetimeIndex):
            timestamps_context = pd.DatetimeIndex(timestamps_context)
        df = pd.DataFrame(y_context, index=timestamps_context, columns=columns)
        freq_offset = infer_pandas_freq_offset_for_datetime_index(df.index)
        fh = ForecastingHorizon(
            np.arange(1, forecast_horizon + 1, dtype=np.int64),
            is_relative=True,
            freq=freq_offset,
        )
        _patch_transformers_tiny_time_mixer_tied_weights()
        _patch_sktime_ttm_predict_forecasting_horizon_freq()
        _patch_sktime_ttm_fit_mismatched_keys()
        _patch_sktime_ttm_freq_token_default()
        self._model = TinyTimeMixerForecaster(
            model_path=model_path, revision=revision
        )

        X_fit = None
        X_pred = None
        timestamps_target_dt = self._convert_to_datetimeindex(timestamps_target)
        # TTM API requires X for both fit (past) and predict (future). Partial cases
        # use padding: past-only pads future with last row; future-only pads past with zeros.
        if x_context is not None and x_target is not None:
            cov_cols = list(range(x_context.shape[1]))
            X_fit = pd.DataFrame(x_context, index=timestamps_context, columns=cov_cols)
            x_target_trimmed = x_target[:forecast_horizon]
            X_pred = pd.DataFrame(
                x_target_trimmed, index=timestamps_target_dt, columns=cov_cols
            )
        elif x_context is not None:
            # Past-only: pad future with last observed value
            cov_cols = list(range(x_context.shape[1]))
            X_fit = pd.DataFrame(x_context, index=timestamps_context, columns=cov_cols)
            last_row = np.tile(x_context[-1:], (forecast_horizon, 1))
            X_pred = pd.DataFrame(
                last_row, index=timestamps_target_dt, columns=cov_cols
            )
        elif x_target is not None:
            # Future-only: pad past with zeros (unknown in past)
            x_target_trimmed = x_target[:forecast_horizon]
            cov_cols = list(range(x_target_trimmed.shape[1]))
            ctx_len = y_context.shape[0]
            past_pad = np.zeros((ctx_len, x_target_trimmed.shape[1]), dtype=np.float64)
            X_fit = pd.DataFrame(
                past_pad, index=timestamps_context, columns=cov_cols
            )
            X_pred = pd.DataFrame(
                x_target_trimmed, index=timestamps_target_dt, columns=cov_cols
            )

        self._model.fit(df, X=X_fit, fh=fh)
        # sktime may normalize the stored FH and lose .freq after fit(); re-inject
        # to prevent TypeError in _predict → to_absolute → _to_offset.
        if hasattr(self._model, "fh") and getattr(self._model.fh, "freq", None) is None:
            self._model.fh.freq = freq_offset
        forecast = self._model.predict(X=X_pred)
        return np.asarray(forecast)  # (forecast_horizon, num_targets)

    def _convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                min_ts = np.min(timestamps)
                max_ts = np.max(timestamps)

                # Pandas datetime bounds for 64-bit ns: 1677-09-21 to 2262-04-11
                # 1677-09-21T00:12:43.145224192Z = -9223372036854775808 ns
                # 2262-04-11T23:47:16.854775807Z = 9223372036854775807 ns
                NS_LOWER = -9223372036854775808
                NS_UPPER = 9223372036854775807
                US_LOWER = NS_LOWER // 1000
                US_UPPER = NS_UPPER // 1000
                MS_LOWER = NS_LOWER // 1_000_000
                MS_UPPER = NS_UPPER // 1_000_000
                S_LOWER = NS_LOWER // 1_000_000_000
                S_UPPER = NS_UPPER // 1_000_000_000

                def in_bounds(val, lower, upper):
                    return lower <= val <= upper

                # Try to classify the likely unit and check bounds
                unit = None
                if isinstance(min_ts, (int, np.integer)):
                    # Try nanoseconds
                    if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(
                        max_ts, NS_LOWER, NS_UPPER
                    ):
                        unit = "ns"
                    # Try microseconds
                    elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(
                        max_ts, US_LOWER, US_UPPER
                    ):
                        unit = "us"
                    # Try milliseconds
                    elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(
                        max_ts, MS_LOWER, MS_UPPER
                    ):
                        unit = "ms"
                    # Try seconds
                    elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(
                        max_ts, S_LOWER, S_UPPER
                    ):
                        unit = "s"
                    else:
                        raise ValueError(
                            f"Timestamps are out of bounds for pandas datetime64[ns] (min={min_ts}, max={max_ts})."
                        )
                    timestamps = pd.to_datetime(timestamps, unit=unit)
                else:
                    timestamps = pd.to_datetime(timestamps)

        return timestamps
