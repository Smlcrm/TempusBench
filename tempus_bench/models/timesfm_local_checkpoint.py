"""Resolve a TimesFM PyTorch checkpoint *file* inside an HF snapshot directory.

``TimesFmCheckpoint.path`` must be a file path for ``torch.load``; ``MODEL_WEIGHTS_PATH``
layout provides a directory root per repo id.
"""

from __future__ import annotations

import os

# Basenames to try under the snapshot root (order matters).
_TIMESFM_LOCAL_CHECKPOINT_CANDIDATES: tuple[str, ...] = (
    "torch_model.ckpt",
    "pytorch_model.bin",
    "model.bin",
    "model.safetensors",
)


def local_timesfm_checkpoint_file(snapshot_dir: str) -> str:
    """Return the path to a checkpoint file under ``snapshot_dir``."""
    for name in _TIMESFM_LOCAL_CHECKPOINT_CANDIDATES:
        candidate = os.path.join(snapshot_dir, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No TimesFM PyTorch checkpoint file found under local weights directory "
        f"{snapshot_dir!r}; expected one of: {', '.join(_TIMESFM_LOCAL_CHECKPOINT_CANDIDATES)}"
    )
