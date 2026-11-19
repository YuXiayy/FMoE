"""Selective activation checkpointing that avoids stateful MoE routing."""

from __future__ import annotations

from functools import wraps

from torch.utils.checkpoint import checkpoint

from net.band_sequence import _FeedForward as BandFeedForward
from net.band_sequence import _RotarySelfAttention
from net.fmoe import (
    Attention,
    CrossAttention,
    FeedForward,
    SharedConditionedBandExpert,
)


_PURE_TYPES = (
    Attention,
    CrossAttention,
    FeedForward,
    SharedConditionedBandExpert,
    _RotarySelfAttention,
    BandFeedForward,
)


def checkpoint_pure_forward(module) -> None:
    if getattr(module, "_fmoe_activation_checkpointed", False):
        return
    original_forward = module.forward

    @wraps(original_forward)
    def checkpointed_forward(*args, __forward=original_forward, **kwargs):
        return checkpoint(
            __forward,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            **kwargs,
        )

    module.forward = checkpointed_forward
    module._fmoe_activation_checkpointed = True


def enable_pure_activation_checkpointing(module) -> int:
    """Checkpoint pure compute modules without replaying router EMA updates."""
    wrapped = 0
    for child in module.modules():
        if not isinstance(child, _PURE_TYPES):
            continue
        if getattr(child, "_fmoe_activation_checkpointed", False):
            continue
        checkpoint_pure_forward(child)
        wrapped += 1
    if wrapped < 1:
        raise RuntimeError("no pure attention/FFN modules were checkpointed")
    return wrapped
