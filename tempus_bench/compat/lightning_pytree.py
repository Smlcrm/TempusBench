"""
PyTorch Lightning uses ``LeafSpec()`` in ``pytorch_lightning.utilities._pytree``.

PyTorch 2.5+ deprecates constructing ``LeafSpec`` (``FutureWarning``: use
``TreeSpec.is_leaf()``). The registry exposes ``treespec_leaf()`` which returns
the cached leaf spec without tripping that warning. We replace Lightning's
``_tree_flatten`` with an equivalent that uses ``treespec_leaf()``.
"""

from __future__ import annotations

from typing import Any

_PATCH_ATTR = "_tempusbench_lightning_pytree_leafspec_patched"


def apply_lightning_pytree_leafspec_patch() -> None:
    """Idempotent runtime patch; no-op if torch / Lightning are missing or torch is too old."""
    try:
        import torch.utils._pytree as torch_pytree
        treespec_leaf = getattr(torch_pytree, "treespec_leaf", None)
        if treespec_leaf is None:
            return
        import pytorch_lightning.utilities._pytree as pl_pytree
    except ImportError:
        return

    if getattr(pl_pytree, _PATCH_ATTR, False):
        return

    from torch.utils._pytree import (
        SUPPORTED_NODES,
        PyTree,
        TreeSpec,
        _get_node_type,
    )

    _is_leaf_or_primitive_container = pl_pytree._is_leaf_or_primitive_container

    def _tree_flatten(pytree: PyTree) -> tuple[list[Any], TreeSpec]:
        if _is_leaf_or_primitive_container(pytree):
            return [pytree], treespec_leaf()

        node_type = _get_node_type(pytree)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        child_pytrees, context = flatten_fn(pytree)

        result: list[Any] = []
        children_specs: list[TreeSpec] = []
        for child in child_pytrees:
            flat, child_spec = _tree_flatten(child)
            result += flat
            children_specs.append(child_spec)

        return result, TreeSpec(node_type, context, children_specs)

    pl_pytree._tree_flatten = _tree_flatten
    setattr(pl_pytree, _PATCH_ATTR, True)
