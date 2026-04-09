"""
Layout of variate columns for Lag-Llama channel-independent prediction.

The vendored model is univariate (``input_size=1``). TempusBench runs one forward pass
per target column and, when ``x_context`` is present, one pass per covariate column so
that past covariates extend the history the way other channel-independent baselines do.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def variates_and_num_targets_for_predict(
    y_context: np.ndarray,
    x_context: Optional[np.ndarray],
) -> tuple[np.ndarray, int]:
    """
    Return the 2D variate matrix (time, channels) and the number of target columns.

    Target columns are the leading ``num_targets`` columns; any trailing columns come
    from ``x_context`` and are not returned in the final forecast stack.
    """
    if y_context.ndim != 2:
        raise ValueError(
            f"y_context must be 2D (time, targets), got shape {y_context.shape}"
        )
    num_targets = int(y_context.shape[1])
    if x_context is not None:
        if x_context.ndim != 2:
            raise ValueError(
                f"x_context must be 2D (time, covariates), got shape {x_context.shape}"
            )
        if x_context.shape[0] != y_context.shape[0]:
            raise ValueError(
                "x_context and y_context must have the same number of time steps: "
                f"{x_context.shape[0]} vs {y_context.shape[0]}"
            )
        variates = np.concatenate([y_context, x_context], axis=1)
    else:
        variates = y_context
    return variates, num_targets
