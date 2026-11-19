"""Fail-fast training-budget projections.

Epoch counts are not comparable across datasets.  In particular, a batch-one
run over the full SonicMaster training split turns the historical 120-epoch
default into tens of millions of optimizer updates.  This module converts the
user-facing configuration into an explicit step budget before CUDA work begins.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TrainingBudget:
    dataset_size: int
    world_size: int
    micro_batch_size_per_device: int
    accumulate_grad_batches: int
    max_epochs: int
    max_steps: int
    samples_per_rank: int
    micro_batches_per_epoch: int
    optimizer_steps_per_epoch: int
    epoch_limited_optimizer_steps: int
    planned_optimizer_steps: int
    effective_global_batch_size: int

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def project_training_budget(
    *,
    dataset_size: int,
    micro_batch_size_per_device: int,
    accumulate_grad_batches: int,
    max_epochs: int,
    max_steps: int = -1,
    world_size: int = 1,
    drop_last: bool = True,
) -> TrainingBudget:
    """Project Lightning optimizer steps without constructing a Trainer.

    ``DistributedSampler`` pads each rank to ``ceil(N / world_size)`` samples.
    The subsequent dataloader either drops or retains its final micro-batch.
    Lightning performs one optimizer update per accumulation window, including
    a final partial window.
    """

    positive_values = {
        "dataset_size": dataset_size,
        "micro_batch_size_per_device": micro_batch_size_per_device,
        "accumulate_grad_batches": accumulate_grad_batches,
        "max_epochs": max_epochs,
        "world_size": world_size,
    }
    invalid = {name: value for name, value in positive_values.items() if value < 1}
    if invalid:
        details = ", ".join(f"{name}={value}" for name, value in invalid.items())
        raise ValueError(f"training-budget inputs must be positive: {details}")
    if max_steps == 0 or max_steps < -1:
        raise ValueError("max_steps must be -1 (disabled) or a positive integer")

    samples_per_rank = math.ceil(dataset_size / world_size)
    if drop_last:
        micro_batches_per_epoch = samples_per_rank // micro_batch_size_per_device
    else:
        micro_batches_per_epoch = math.ceil(
            samples_per_rank / micro_batch_size_per_device
        )
    if micro_batches_per_epoch < 1:
        raise ValueError(
            "the dataloader would yield zero batches; reduce batch size or disable drop_last"
        )

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / accumulate_grad_batches
    )
    epoch_limited_steps = optimizer_steps_per_epoch * max_epochs
    planned_steps = (
        min(epoch_limited_steps, max_steps) if max_steps > 0 else epoch_limited_steps
    )

    return TrainingBudget(
        dataset_size=dataset_size,
        world_size=world_size,
        micro_batch_size_per_device=micro_batch_size_per_device,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs,
        max_steps=max_steps,
        samples_per_rank=samples_per_rank,
        micro_batches_per_epoch=micro_batches_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        epoch_limited_optimizer_steps=epoch_limited_steps,
        planned_optimizer_steps=planned_steps,
        effective_global_batch_size=(
            micro_batch_size_per_device * world_size * accumulate_grad_batches
        ),
    )


def enforce_training_budget(
    budget: TrainingBudget,
    *,
    max_steps_without_override: int,
    allow_large_training_budget: bool,
) -> None:
    """Reject an accidental long run unless the configuration opts in."""

    if max_steps_without_override < 1:
        raise ValueError("max_steps_without_override must be positive")
    if (
        budget.planned_optimizer_steps > max_steps_without_override
        and not allow_large_training_budget
    ):
        raise RuntimeError(
            "Refusing projected training run with "
            f"{budget.planned_optimizer_steps:,} optimizer steps "
            f"({budget.optimizer_steps_per_epoch:,}/epoch for "
            f"{budget.max_epochs} epochs). Set --max_steps to a bounded pilot "
            "budget, or explicitly pass --allow_large_training_budget true "
            "after measuring throughput and storage requirements."
        )


def format_training_budget(budget: TrainingBudget) -> str:
    """Return a compact, log-friendly projection."""

    return (
        "[Training budget] "
        f"dataset={budget.dataset_size:,}, world_size={budget.world_size}, "
        f"micro_batch/device={budget.micro_batch_size_per_device}, "
        f"accumulate={budget.accumulate_grad_batches}, "
        f"effective_global_batch={budget.effective_global_batch_size}, "
        f"optimizer_steps/epoch={budget.optimizer_steps_per_epoch:,}, "
        f"epoch_limit={budget.epoch_limited_optimizer_steps:,}, "
        f"planned={budget.planned_optimizer_steps:,}"
    )
