import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Union, Optional, Tuple, Dict
from functools import partial

from typing import (
    Optional,
    NamedTuple,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
    Any,
    Sequence,
)


class DataProcessor(nnx.Module):

    def __init__(
        self,
        scaling_method: str,
        transform_method: str,
        in_length: int,
        out_length: int,
        in_features: int,
        out_features: int,
    ):

        self.scaling_method = scaling_method
        self.transform_method = transform_method
        self.in_length = in_length
        self.out_length = out_length
        self.in_features = in_features
        self.out_features = out_features

        scaling_methods = {
            "z_score": self.z_score_stats,
            "minmax": self.minmax_stats,
            "robust": self.robust_stats,
            "max_abs": self.max_abs_stats,
        }

        transform_methods = {
            "none": lambda x: x,
            "arcsinh": jnp.arcsinh,
            "log": jnp.log,
        }
        inverse_transform_methods = {
            "none": lambda x: x,
            "arcsinh": jnp.sinh,
            "log": jnp.exp,
        }

        if scaling_method not in scaling_methods:
            raise ValueError(
                f"Unsupported scaling method: {scaling_method}. Choose 'z_score' or 'minmax' or 'robust'."
            )

        self.y_get_stats = scaling_methods[scaling_method]
        self.y_transform = transform_methods[transform_method]
        self.y_inverse_transform = inverse_transform_methods[transform_method]

    @staticmethod
    def pad(
        data: jax.Array,
        target_length: Union[int, Sequence[int]],
        axis: Union[int, Sequence[int]],
        side: str = "right",
    ) -> Tuple[jax.Array, jax.Array]:
        """Pad or trim an N-D array along one or more axes on the specified side ('left' or 'right').
        Returns (data, mask) where mask is 1 for original elements and 0 for padding.
        """
        ndim = data.ndim
        axes = [axis] if isinstance(axis, int) else list(axis)

        if not axes:
            raise ValueError("axis must be an int or a non-empty sequence of ints.")
        if not all(-ndim <= a < ndim for a in axes):
            raise ValueError(f"All axes must be in [-{ndim}, {ndim-1}].")
        if side not in ("left", "right"):
            raise ValueError("Side must be 'left' or 'right'.")

        # Normalize negatives and deduplicate while preserving order
        axes = list(dict.fromkeys([(a + ndim) if a < 0 else a for a in axes]))

        # Normalize target_length to per-axis mapping
        if isinstance(target_length, Sequence):
            if len(target_length) != len(axes):
                raise ValueError(
                    "When target_length is a sequence, its length must match the number of axes."
                )
            target_length_map = {a: t for a, t in zip(axes, target_length)}
        else:
            target_length_map = {a: target_length for a in axes}

        shape = list(data.shape)

        # ---- 1) Trim per axis in one shot via slices ----
        trim_slices = [slice(None)] * ndim
        for a in axes:
            target = target_length_map[a]
            pad_len = target - shape[a]
            if pad_len <= 0:
                # keep last 'target' on left-trim, first 'target' on right-trim
                trim_slices[a] = (
                    slice(-target, None) if side == "left" else slice(None, target)
                )
                shape[a] = target

        data = data[tuple(trim_slices)]

        # ---- 2) Compute padding widths ----
        pad_width = [(0, 0)] * ndim
        for a in axes:
            target = target_length_map[a]
            pad_len = target - data.shape[a]
            if pad_len > 0:
                pad_width[a] = (pad_len, 0) if side == "left" else (0, pad_len)

        # ---- 3) Create and pad mask identically ----
        mask = jnp.ones_like(data, dtype=jnp.float32)
        if any(p != (0, 0) for p in pad_width):
            data = jnp.pad(data, tuple(pad_width), mode="constant", constant_values=0.0)
            mask = jnp.pad(mask, tuple(pad_width), mode="constant", constant_values=0.0)

        return data, mask

    def get_context_index_features(self, context_y, **kwargs) -> jax.Array:
        """Feature engineering based on index."""
        total_length = self.in_length + self.out_length

        context_y_length = context_y.shape[-2]

        context_x = jnp.arange(-context_y_length, 0) / total_length
        context_x = jnp.expand_dims(context_x, axis=(0, -1))
        context_x = jnp.broadcast_to(context_x, context_y.shape[:-1] + (1,))

        return context_x

    def get_target_index_features(self, target_y, **kwargs) -> jax.Array:
        """Feature engineering based on index."""
        total_length = self.in_length + self.out_length

        target_y_length = target_y.shape[-2]

        target_x = jnp.arange(1, target_y_length + 1) / total_length
        target_x = jnp.expand_dims(target_x, axis=(0, -1))
        target_x = jnp.broadcast_to(target_x, target_y.shape[:-1] + (1,))

        return target_x

    def get_index_features(
        self, context_y, target_y, **kwargs
    ) -> Tuple[jax.Array, jax.Array]:
        """Feature engineering based on index."""
        context_x = self.get_context_index_features(context_y)
        target_x = self.get_target_index_features(target_y)

        return context_x, target_x

    def create_time_features(self, context_y, target_y, **kwargs):
        context_x, target_x = self.get_index_features(context_y, target_y)

        data = {
            "context_x": context_x,
            "context_y": context_y,
            "target_x": target_x,
            "target_y": target_y,
        }

        return data

    def scale_context_y(self, context_y, min_val=1e-5, **kwargs):
        # Get scaler stats
        loc, scale = self.y_get_stats(context_y, axis=-2)  # Get stats from context_y

        # Scale y along time dimension
        context_y = self.scaler(
            context_y, loc, scale, min_val=min_val
        )  # Normalize context_y

        return context_y, loc, scale

    def scale_target_y(self, target_y, loc, scale, min_val=1e-5, **kwargs):
        # Scale y along time dimension
        target_y = self.scaler(
            target_y, loc, scale, min_val=min_val
        )  # Normalize context_y

        return target_y

    def scale_y(self, context_y, target_y, min_val=1e-5, **kwargs):

        context_y, loc, scale = self.scale_context_y(context_y, min_val=min_val)

        target_y = self.scale_target_y(target_y, loc, scale, min_val=min_val)

        return context_y, target_y, loc, scale

    def scale_data(
        self, context_x, context_y, target_x, target_y, min_val=1e-5, **kwargs
    ) -> Dict[str, jax.Array]:
        # Scale context_y and target_y
        context_y_scaled, target_y_scaled, loc, scale = self.scale_y(
            context_y=context_y,
            target_y=target_y,
            min_val=min_val,
        )

        data = {
            "context_x": context_x,
            "context_y": context_y_scaled,
            "target_x": target_x,
            "target_y": target_y_scaled,
            "loc": loc,
            "scale": scale,
        }

        return data

    def pad_context_x(self, context_x, **kwargs):
        # Pad context_x along time dimension
        context_x_padded, context_x_mask = self.pad(
            context_x, self.in_length, axis=-2, side="left"
        )

        return context_x_padded, context_x_mask

    def pad_context_y(self, context_y, **kwargs):
        # pad along time and feature dimensions
        context_y_padded, context_y_mask = self.pad(
            context_y, (self.in_length, self.in_features), axis=(-2, -1), side="left"
        )

        return context_y_padded, context_y_mask

    def pad_context(self, context_x, context_y, **kwargs):
        # NOTE: Right now it assumes all time-series are of the same length so context_x_mask is returned as the context mask but has to be generalized for setting where variates are of different lengths.
        # pad context_y along time and feature dimensions
        context_y_padded, _ = self.pad_context_y(context_y)

        # Pad context_x along time dimension
        context_x_padded, context_x_mask = self.pad_context_x(context_x)

        return context_x_padded, context_y_padded, context_x_mask

    def pad_target_y(self, target_y, **kwargs):
        # pad along time and feature dimensions
        target_y_padded, target_y_mask = self.pad(
            target_y,
            (self.out_length, self.out_features),
            axis=(-2, -1),
            side="right",
        )

        return target_y_padded, target_y_mask

    def pad_target_x(self, target_x, **kwargs):
        # Pad target_x along time dimension
        target_x_padded, target_x_mask = self.pad(
            target_x, self.out_length, axis=-2, side="right"
        )

        return target_x_padded, target_x_mask

    def pad_target(self, target_x, target_y, **kwargs):
        # NOTE: Right now it assumes all time-series are of the same length so target_x_mask is returned as the target mask but has to be generalized for setting where variates are of different lengths.

        # pad along time and feature dimensions
        target_y_padded, _ = self.pad_target_y(target_y)

        # Pad target_x along time dimension
        target_x_padded, target_x_mask = self.pad_target_x(target_x)

        return target_x_padded, target_y_padded, target_x_mask

    def pad_data(
        self, context_x, context_y, target_x, target_y, **kwargs
    ) -> Dict[str, jax.Array]:

        # Pad context
        context_x_padded, context_y_padded, context_mask = self.pad_context(
            context_x, context_y
        )

        # Pad target
        target_x_padded, target_y_padded, target_mask = self.pad_target(
            target_x, target_y
        )

        data = {
            "context_x": context_x_padded,
            "context_y": context_y_padded,
            "context_mask": context_mask,
            "target_x": target_x_padded,
            "target_y": target_y_padded,
            "target_mask": target_mask,
        }

        return data

    ######################### UTILS FUNCTIONS #########################

    def scaler(
        self, data: jax.Array, loc: jax.Array, scale: jax.Array, min_val: float = 1e-5
    ) -> jax.Array:
        z = (data - loc) / (scale + min_val)
        return self.y_transform(z)

    def descaler(
        self, data: jax.Array, loc: jax.Array, scale: jax.Array, min_val: float = 1e-5
    ) -> jax.Array:
        data = self.y_inverse_transform(data)
        data = data * (scale + min_val) + loc
        return data

    @staticmethod
    def z_score_stats(data: jax.Array, axis: int) -> Tuple[jax.Array, jax.Array]:
        mean = jnp.mean(data, axis=axis, keepdims=True)
        std = jnp.std(data, axis=axis, keepdims=True)

        return mean, std

    @staticmethod
    def minmax_stats(data: jax.Array, axis: int) -> Tuple[jax.Array, jax.Array]:
        min_data = jnp.min(data, axis=axis, keepdims=True)
        max_data = jnp.max(data, axis=axis, keepdims=True)
        scale = max_data - min_data

        return min_data, scale

    @staticmethod
    def robust_stats(data: jax.Array, axis: int) -> Tuple[jax.Array, jax.Array]:
        median = jnp.median(data, axis=axis, keepdims=True)
        q1 = jnp.percentile(data, 25, axis=axis, keepdims=True)
        q3 = jnp.percentile(data, 75, axis=axis, keepdims=True)
        iqr = q3 - q1

        return median, iqr

    @staticmethod
    def max_abs_stats(data: jax.Array, axis: int) -> Tuple[jax.Array, jax.Array]:
        max_abs = jnp.max(jnp.abs(data), axis=axis, keepdims=True)
        scale = max_abs

        return 0.0, scale  # type: ignore


class LAFNInferer(nnx.Module):
    def __init__(self, model: nnx.Module, data_processor: DataProcessor):
        self.model = model
        self.data_processor = data_processor

    def __call__(
        self,
        context_y: Union[np.ndarray, jax.Array],
        context_t: Union[np.ndarray, jax.Array, None] = None,
        target_t: Union[np.ndarray, jax.Array, None] = None,
        forecast_horizon: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Forecasting with LAFN model.

        Args:
            context_y: jax.Array of shape (batch_size, context_length, num_y_features) or (context_length, num_y_features) or (context_length,).
                The historical target values.
            context_t: jax.Array of shape (batch_size, context_length, 1) or (context_length, 1) or (context_length,).
        """
        if context_y.ndim == 1:
            context_y = jnp.expand_dims(
                context_y, axis=(0, -1)
            )  # (1, context_length, 1)

        elif context_y.ndim == 2:
            context_y = jnp.expand_dims(
                context_y, axis=-1
            )  # (batch_size, context_length, 1)

        elif context_y.ndim > 3:
            raise ValueError(
                "context_y must be of shape (context_length,) or (batch_size, context_length) or (batch_size, context_length, num_y_features)."
            )

        if target_t is None and forecast_horizon is None:
            raise ValueError("Either target_t or forecast_horizon must be provided.")

        elif target_t is not None and forecast_horizon is not None:
            raise ValueError(
                "Only one of target_t or forecast_horizon should be provided."
            )

        context_x, target_x = self.get_time_features(
            context_y=context_y,
            context_t=context_t,
            target_t=target_t,
            forecast_horizon=64,
        )

        data = self.scale_and_pad_data(
            context_x=context_x, context_y=context_y, target_x=target_x  # type: ignore
        )

        forecast_embed = self.model(**data)

        return forecast_embed, data["loc"], data["scale"]

    def get_time_features(
        self,
        context_y: Union[np.ndarray, jax.Array],
        context_t: Union[np.ndarray, jax.Array, None] = None,
        target_t: Union[np.ndarray, jax.Array, None] = None,
        forecast_horizon: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array]:

        if forecast_horizon is not None:  # forecast_horizon is not None
            target_t = self.data_processor.get_target_index_features(
                jnp.zeros((context_y.shape[0], forecast_horizon, 1))
            )
        else:
            target_t = jnp.asarray(target_t)
            if target_t.ndim == 1:
                target_t = jnp.expand_dims(
                    target_t, axis=(0, -1)
                )  # (1, target_length, 1)
            elif target_t.ndim == 2:
                target_t = jnp.expand_dims(
                    target_t, axis=-1
                )  # (batch_size, target_length, 1)

        if context_t is None:
            context_t = self.data_processor.get_context_index_features(context_y)

        else:
            context_t = jnp.asarray(context_t)

            if context_t.ndim == 1:
                context_t = jnp.expand_dims(
                    context_t, axis=(0, -1)
                )  # (1, context_length, 1)

            elif context_t.ndim == 2:
                context_t = jnp.expand_dims(
                    context_t, axis=-1
                )  # (batch_size, context_length, 1)

        return context_t, target_t

    def scale_and_pad_data(
        self, context_x: jax.Array, context_y: jax.Array, target_x: jax.Array
    ) -> Dict[str, jax.Array]:

        # Scale context_y data
        context_y_scaled, loc, scale = self.data_processor.scale_context_y(context_y)

        context_x_padded, context_y_padded, context_mask = (
            self.data_processor.pad_context(
                context_x=context_x, context_y=context_y_scaled
            )
        )

        target_x_padded, target_mask = self.data_processor.pad_target_x(
            target_x=target_x
        )

        data = {
            "context_x": context_x_padded,
            "context_y": context_y_padded,
            "context_mask": context_mask,
            "target_x": target_x_padded,
            "target_mask": target_mask,
            "loc": loc,
            "scale": scale,
        }

        return data

    def forecast(
        self,
        context_y: Union[np.ndarray, jax.Array],
        context_t: Optional[Union[np.ndarray, jax.Array, None]] = None,
        target_t: Optional[Union[np.ndarray, jax.Array, None]] = None,
        forecast_horizon: Optional[int] = None,
    ):
        forecast_embed, loc, scale = self.__call__(
            context_y=context_y,
            context_t=context_t,
            target_t=target_t,
            forecast_horizon=forecast_horizon,
        )
        num_features = context_y.shape[-1]
        pred_y = self.model.forecast(forecast_embed=forecast_embed)

        pred_y = pred_y[:, :forecast_horizon, -num_features:]
        pred_y_descaled = self.data_processor.descaler(pred_y, loc, scale)
        pred_y_descaled = jnp.squeeze(pred_y_descaled, axis=0)
        return pred_y_descaled

    def pdf(
        self,
        y: jax.Array,
        context_y: Union[np.ndarray, jax.Array],
        context_t: Optional[Union[np.ndarray, jax.Array, None]] = None,
        target_t: Optional[Union[np.ndarray, jax.Array, None]] = None,
        forecast_horizon: Optional[int] = None,
    ):
        forecast_embed, loc, scale = self.__call__(
            context_y=context_y,
            context_t=context_t,
            target_t=target_t,
            forecast_horizon=forecast_horizon,
        )

        pdfs = self.model.pdf(y=y, forecast_embed=forecast_embed, loc=loc, scale=scale)

        return pdfs

    def sample(
        self,
        context_y: Union[np.ndarray, jax.Array],
        context_t: Optional[Union[np.ndarray, jax.Array, None]] = None,
        target_t: Optional[Union[np.ndarray, jax.Array, None]] = None,
        forecast_horizon: Optional[int] = None,
        num_samples: int = 100,
        **kwargs,
    ):
        forecast_embed, loc, scale = self.__call__(
            context_y=context_y,
            context_t=context_t,
            target_t=target_t,
            forecast_horizon=forecast_horizon,
        )

        samples = self.model.sample(
            forecast_embed=forecast_embed, loc=0.0, scale=1.0, num_samples=num_samples
        )

        num_features = context_y.shape[-1]

        samples = jnp.squeeze(samples, axis=1)
        samples = samples[:, :forecast_horizon, -num_features:]

        samples_descaled = samples * scale + loc
        # pred_y_descaled = self.data_processor.descaler(pred_y, loc, scale)

        # samples_descaled = self.data_processor.descaler(samples, loc, scale)
        # pred_y_descaled = jnp.squeeze(pred_y_descaled, axis=0)

        return samples_descaled
