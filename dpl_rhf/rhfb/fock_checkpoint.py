"""Exact activation checkpoint support for finite-range Fock energy blocks.

Callers put the entire expensive pair-block contraction inside the callback and
pass every differentiable tensor it reads as an explicit tensor argument.
The callback is recomputed during backward, while its scalar output and small
input views replace saved Green-field and derivative intermediates.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint


def exact_energy_block(function: Callable[..., torch.Tensor], *tensor_inputs: torch.Tensor,
                       enabled: bool = True) -> torch.Tensor:
    """Evaluate a deterministic scalar block with exact backward recomputation.

    No approximation, detachment, or change to the contraction is introduced.
    During inference or when no input needs a gradient, the callback runs once
    without checkpoint machinery. ``preserve_rng_state=False`` is safe because
    EDF pair blocks are deterministic and contain no random operations.
    """
    if not callable(function):
        raise TypeError("checkpointed energy block requires a callable")
    if any(not isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("all checkpoint inputs must be tensors")
    if not torch.is_grad_enabled() or not enabled or not any(
        value.requires_grad for value in tensor_inputs
    ):
        return function(*tensor_inputs)
    result = checkpoint(function, *tensor_inputs, use_reentrant=False,
                        preserve_rng_state=False)
    if not isinstance(result, torch.Tensor) or result.numel() != 1:
        raise ValueError("checkpointed pair block must return one scalar tensor")
    return result
