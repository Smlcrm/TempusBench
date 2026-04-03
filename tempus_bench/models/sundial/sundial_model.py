"""
Sundial diffusion-based foundation model for time series forecasting.

Sundial generates multiple sample trajectories via diffusion, making it a
stochastic forecaster. It operates on univariate sequences, so multivariate
targets are handled by iterating over each target independently with
z-score normalization.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from packaging.version import Version
from pydantic import BaseModel as PydanticBaseModel
import transformers
from transformers import AutoModelForCausalLM

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support


def _patch_dynamic_cache_for_sundial():
    """Polyfill DynamicCache attrs removed in transformers >=4.48 for Sundial's trust_remote_code."""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            if hasattr(self, "get_max_cache_shape"):
                shape = self.get_max_cache_shape()
                if shape is not None and shape > 0:
                    return shape
            return None

        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length

        DynamicCache.get_usable_length = _get_usable_length


def _needs_sundial_generation_compat() -> bool:
    return Version(transformers.__version__) >= Version("4.48.0")


def _sundial_use_patch_attention_mask() -> bool:
    """Patch-length ``attention_mask`` fixes position_ids on transformers 4.48–4.x but breaks KV-cache updates on 5.x."""
    v = Version(transformers.__version__)
    return Version("4.48.0") <= v < Version("5.0.0")


def _patch_generation_mixin_for_sundial(model: Any) -> None:
    """Patch GenerationMixin API drift for Sundial remote code on newer transformers."""
    cls = model.__class__
    if hasattr(cls, "_extract_past_from_model_output"):
        return

    def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
        if hasattr(outputs, "past_key_values"):
            return outputs.past_key_values
        if hasattr(outputs, "mems"):
            return outputs.mems
        if hasattr(outputs, "past_buckets_states"):
            return outputs.past_buckets_states
        return None

    cls._extract_past_from_model_output = _extract_past_from_model_output


class SundialHyperparams(PydanticBaseModel):
    pass


class SundialModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, SundialHyperparams)

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
    ) -> "SundialModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Sundial",
        )
        _patch_dynamic_cache_for_sundial()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )
        if _needs_sundial_generation_compat():
            _patch_generation_mixin_for_sundial(self._model)
        self._model.eval()
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
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("SundialModel is not fitted. Call train() first.")

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Sundial",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 20)
        lookback = getattr(self, "lookback_length", None) or y_context.shape[0]

        # Truncate to lookback_length (dev branch compatibility)
        y_ctx = y_context[-lookback:] if y_context.shape[0] > lookback else y_context

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            x_ctx = x_context[-lookback:] if x_context.shape[0] > lookback else x_context
            y_input = np.concatenate([y_ctx, x_ctx], axis=1)
        else:
            y_input = y_ctx

        n_channels = y_input.shape[1]
        all_samples = []

        for i in range(n_channels):
            series = y_input[:, i].astype(np.float64)

            mean = series.mean()
            std = series.std()
            if std == 0:
                std = 1.0
            normed = (series - mean) / std

            seqs = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": forecast_horizon,
                    "num_samples": num_samples,
                }
                if _needs_sundial_generation_compat():
                    # SundialPatchEmbedding left-pads to a multiple of ``input_token_len``.
                    # Right-pad the raw series to the same multiple so
                    # ``input_ids.shape[1] // input_token_len`` matches the patch count from
                    # the embedding (avoids position_ids / mask shape errors on transformers
                    # ≥4.48). Use a patch-length ``attention_mask`` (not raw timesteps).
                    patch_size = int(
                        getattr(self._model.config, "input_token_len", 16)
                    )
                    L = int(seqs.shape[1])
                    pad = (patch_size - (L % patch_size)) % patch_size
                    if pad:
                        pad_val = float(seqs[0, -1].item())
                        seqs = torch.nn.functional.pad(
                            seqs, (0, pad), mode="constant", value=pad_val
                        )
                    if _sundial_use_patch_attention_mask():
                        token_len = max(1, seqs.shape[1] // patch_size)
                        generate_kwargs["attention_mask"] = torch.ones(
                            (1, token_len), dtype=torch.long, device=seqs.device
                        )
                output = self._model.generate(
                    seqs, **generate_kwargs
                )

            # output: (1, num_samples, context_length + forecast_horizon)
            preds = output[0, :, -forecast_horizon:].cpu().numpy()
            preds = preds * std + mean

            all_samples.append(preds)

        # all_samples: list of (num_samples, forecast_horizon), one per channel
        # Stack and keep only first num_targets (discard covariate channel forecasts)
        result = np.stack(all_samples, axis=-1)[:, :, :num_targets]
        return result.astype(np.float64)
