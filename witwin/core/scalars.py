"""Scalar leaf types shared by Core world contracts.

Core keeps continuous physical quantities polymorphic on purpose: a value may
be a plain Python number when it is a fixed authored constant, or a scalar
``torch.Tensor`` when it is a differentiable leaf that a solver-owned compiler
must carry into its native stores with gradient state intact. These aliases
name that contract instead of spelling it ``Any``.

Tensor-valued leaves are never cloned, detached, or moved across devices by
Core; see :mod:`witwin.core.scene` for the placement rules.
"""

from __future__ import annotations

from typing import TypeAlias

import torch


ScalarLike: TypeAlias = float | torch.Tensor
"""A real scalar: a Python number, or a 0-dim floating-point tensor."""

ComplexLike: TypeAlias = complex | torch.Tensor
"""A complex scalar: a Python complex, or a complex/real tensor."""


__all__ = ["ComplexLike", "ScalarLike"]
