"""Bounded-memory aggregation and export for sparse expert routing.

The accumulator intentionally keeps only fixed-size, CPU ``float64`` sums.  It
never retains a batch's gates or labels, so its memory use is independent of
the number of calls to :meth:`RoutingAnalysisAccumulator.update`.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float]]
_COMPOUND_LEVELS: Tuple[str, str, str] = ("single", "double", "triple")
_STATE_VERSION = 1
_DENSE_PROBABILITY_ALIASES: Tuple[str, ...] = (
    "dense_probabilities",
    "routing_probabilities",
    "probabilities",
    "dense_gates",
)
_STAGE_TENSOR_FIELDS: Tuple[str, ...] = (
    "degradation_selection",
    "degradation_samples",
    "degradation_slots",
    "compound_selection",
    "compound_samples",
    "compound_slots",
    "selection_sum",
    "load_sum",
    "importance_sum",
    "flops_sum",
    "parameters_sum",
)


def _zeros(*shape: int) -> torch.Tensor:
    """Create a CPU accumulation tensor."""

    return torch.zeros(shape, dtype=torch.float64, device="cpu")


def _to_cpu_float(value: TensorLike, name: str = "value") -> torch.Tensor:
    """Detach and convert a finite statistic to CPU float64.

    Routing failures must be visible to callers.  In particular, silently
    replacing NaN/Inf would turn a broken router into plausible-looking CSVs.
    """

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.is_complex():
        raise ValueError("{} must be real-valued".format(name))
    tensor = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("{} contains NaN or Inf".format(name))
    return tensor


def _first_present(
    values: Mapping[str, object], names: Iterable[str]
) -> Optional[object]:
    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return None


@dataclass
class _StageAccumulator:
    """All retained state for one decoder stage."""

    num_experts: int
    degradation_selection: torch.Tensor
    degradation_samples: torch.Tensor
    degradation_slots: torch.Tensor
    compound_selection: torch.Tensor
    compound_samples: torch.Tensor
    compound_slots: torch.Tensor
    selection_sum: torch.Tensor
    load_sum: torch.Tensor
    importance_sum: torch.Tensor
    flops_sum: torch.Tensor
    parameters_sum: torch.Tensor
    flops_updates: int = 0
    parameters_updates: int = 0
    update_count: int = 0
    routing_decisions: float = 0.0
    entropy_sum: float = 0.0
    entropy_square_sum: float = 0.0
    entropy_count: int = 0
    entropy_min: float = math.inf
    entropy_max: float = -math.inf

    @classmethod
    def create(cls, num_classes: int, num_experts: int) -> "_StageAccumulator":
        return cls(
            num_experts=num_experts,
            degradation_selection=_zeros(num_classes, num_experts),
            degradation_samples=_zeros(num_classes),
            degradation_slots=_zeros(num_classes),
            compound_selection=_zeros(3, num_experts),
            compound_samples=_zeros(3),
            compound_slots=_zeros(3),
            selection_sum=_zeros(num_experts),
            load_sum=_zeros(num_experts),
            importance_sum=_zeros(num_experts),
            flops_sum=_zeros(num_experts),
            parameters_sum=_zeros(num_experts),
        )

    def tensor_bytes(self) -> int:
        total = 0
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total


class RoutingAnalysisAccumulator:
    """Stream routing statistics into fixed-size per-stage summaries.

    Args:
        num_degradation_classes: Width of ``degradation_multi_hot``. F-MoE
            v2 uses 19 classes.
        num_experts: Optional common expert count, or a mapping from stage name
            to count.  When omitted, the count is inferred from ``gates`` or a
            per-expert statistic on the first update for each stage.
        degradation_names: Optional human-readable names used in CSV exports.
        eps: Numerical floor used for probability normalization.

    ``update`` accepts gates shaped ``[B, E]`` or ``[B, ..., E]`` and top-k
    indices shaped ``[B, K]`` or ``[B, ..., K]``.  Extra axes are routing
    positions.  Labels are always ``[B, num_degradation_classes]``.
    """

    def __init__(
        self,
        num_degradation_classes: int = 19,
        num_experts: Optional[Union[int, Mapping[str, int]]] = None,
        degradation_names: Optional[Sequence[str]] = None,
        eps: float = 1.0e-12,
    ) -> None:
        if num_degradation_classes <= 0:
            raise ValueError("num_degradation_classes must be positive")
        if isinstance(num_experts, int) and num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_degradation_classes = int(num_degradation_classes)
        self.num_experts = num_experts
        self.eps = float(eps)
        if degradation_names is None:
            self.degradation_names = tuple(
                "degradation_{}".format(index)
                for index in range(self.num_degradation_classes)
            )
        else:
            if len(degradation_names) != self.num_degradation_classes:
                raise ValueError(
                    "degradation_names must contain exactly {} entries".format(
                        self.num_degradation_classes
                    )
                )
            self.degradation_names = tuple(str(name) for name in degradation_names)
        self._stages: Dict[str, _StageAccumulator] = {}

    @property
    def stage_names(self) -> Tuple[str, ...]:
        """Return accumulated stage names in deterministic order."""

        return tuple(sorted(self._stages))

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by retained tensor sums (independent of batch count)."""

        return sum(stage.tensor_bytes() for stage in self._stages.values())

    def reset(self) -> None:
        """Discard all accumulated summaries."""

        self._stages.clear()

    def _configured_num_experts(self, stage_name: str) -> Optional[int]:
        if isinstance(self.num_experts, int):
            return self.num_experts
        if isinstance(self.num_experts, Mapping):
            value = self.num_experts.get(stage_name)
            if value is not None and int(value) <= 0:
                raise ValueError("num_experts for {!r} must be positive".format(stage_name))
            return None if value is None else int(value)
        return None

    def _infer_num_experts(
        self, stage_name: str, statistics: Mapping[str, object]
    ) -> int:
        candidates = []
        configured = self._configured_num_experts(stage_name)
        if configured is not None:
            candidates.append(("configured num_experts", configured))

        for probability_name in ("gates",) + _DENSE_PROBABILITY_ALIASES:
            probability_value = statistics.get(probability_name)
            if probability_value is not None:
                probabilities = (
                    probability_value
                    if isinstance(probability_value, torch.Tensor)
                    else torch.as_tensor(probability_value)
                )
                if probabilities.ndim < 2:
                    raise ValueError(
                        "{} for {!r} must have shape [B, ..., E]".format(
                            probability_name, stage_name
                        )
                    )
                candidates.append(
                    (probability_name, int(probabilities.shape[-1]))
                )

        for key_group in (
            ("load",),
            ("importance",),
            ("expert_flops", "average_flops", "flops"),
            ("expert_parameters", "expert_params", "parameters"),
        ):
            value = _first_present(statistics, key_group)
            if value is not None:
                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                candidates.append((key_group[0], int(tensor.numel())))

        if not candidates:
            indices_value = statistics.get("topk_indices")
            if indices_value is None:
                raise ValueError(
                    "Cannot infer the expert count for {!r}; provide gates, a "
                    "per-expert statistic, or num_experts".format(stage_name)
                )
            indices = indices_value if isinstance(indices_value, torch.Tensor) else torch.as_tensor(indices_value)
            valid = indices[indices >= 0]
            if valid.numel() == 0:
                raise ValueError(
                    "Cannot infer the expert count from empty topk_indices for {!r}".format(
                        stage_name
                    )
                )
            candidates.append(("topk_indices", int(valid.max().item()) + 1))

        expected = candidates[0][1]
        if expected <= 0:
            raise ValueError("num_experts for {!r} must be positive".format(stage_name))
        mismatches = [(source, value) for source, value in candidates if value != expected]
        if mismatches:
            details = ", ".join("{}={}".format(source, value) for source, value in candidates)
            raise ValueError("Inconsistent expert counts for {!r}: {}".format(stage_name, details))
        return expected

    def _stage(
        self, stage_name: str, statistics: Mapping[str, object]
    ) -> _StageAccumulator:
        num_experts = self._infer_num_experts(stage_name, statistics)
        state = self._stages.get(stage_name)
        if state is None:
            state = _StageAccumulator.create(
                self.num_degradation_classes, num_experts
            )
            self._stages[stage_name] = state
        elif state.num_experts != num_experts:
            raise ValueError(
                "Expert count for {!r} changed from {} to {}".format(
                    stage_name, state.num_experts, num_experts
                )
            )
        return state

    @staticmethod
    def _validate_batch(tensor: torch.Tensor, batch_size: int, name: str) -> None:
        if tensor.ndim < 2 or tensor.shape[0] != batch_size:
            raise ValueError(
                "{} must start with batch dimension {}; got {}".format(
                    name, batch_size, tuple(tensor.shape)
                )
            )

    def _probabilities(
        self,
        value: TensorLike,
        batch_size: int,
        num_experts: int,
        name: str,
    ) -> torch.Tensor:
        probabilities = _to_cpu_float(value, name)
        self._validate_batch(probabilities, batch_size, name)
        if probabilities.shape[-1] != num_experts:
            raise ValueError(
                "The final {} dimension must equal num_experts".format(name)
            )
        if bool((probabilities < 0.0).any()):
            raise ValueError("{} must be non-negative".format(name))
        return probabilities

    def _entropy_probabilities(
        self,
        statistics: Mapping[str, object],
        batch_size: int,
        num_experts: int,
        sparse_gates: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return pre-top-k probabilities when supplied, else sparse gates.

        With top-1 routing, entropy of the selected sparse gate is always zero.
        The router can therefore expose ``dense_probabilities`` to preserve the
        uncertainty of its pre-selection distribution.
        """

        for name in _DENSE_PROBABILITY_ALIASES:
            value = statistics.get(name)
            if value is not None:
                return self._probabilities(
                    value,  # type: ignore[arg-type]
                    batch_size,
                    num_experts,
                    name,
                )
        return sparse_gates

    def _selection_counts(
        self,
        statistics: Mapping[str, object],
        batch_size: int,
        num_experts: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return per-sample selections, route counts, and sparse gates."""

        gates_value = statistics.get("gates")
        gates = None
        if gates_value is not None:
            gates = self._probabilities(
                gates_value,  # type: ignore[arg-type]
                batch_size,
                num_experts,
                "gates",
            )

        indices_value = statistics.get("topk_indices")
        if indices_value is not None:
            indices_tensor = (
                indices_value
                if isinstance(indices_value, torch.Tensor)
                else torch.as_tensor(indices_value)
            )
            if indices_tensor.is_floating_point():
                if not bool(torch.isfinite(indices_tensor).all()):
                    raise ValueError("topk_indices contains NaN or Inf")
                if not bool(indices_tensor.eq(indices_tensor.round()).all()):
                    raise ValueError("topk_indices must contain integers")
            indices = indices_tensor.detach().to(device="cpu", dtype=torch.int64)
            self._validate_batch(indices, batch_size, "topk_indices")
            flat_indices = indices.reshape(batch_size, -1)
            valid = (flat_indices >= 0) & (flat_indices < num_experts)
            invalid = (flat_indices < -1) | (flat_indices >= num_experts)
            if bool(invalid.any()):
                raise ValueError("topk_indices contains an out-of-range expert index")

            selections = _zeros(batch_size, num_experts)
            safe_indices = flat_indices.clamp(min=0, max=num_experts - 1)
            selections.scatter_add_(1, safe_indices, valid.to(torch.float64))

            # The final dimension is K; intervening dimensions are route axes.
            route_count = 1
            for size in indices.shape[1:-1]:
                route_count *= int(size)
            route_counts = torch.full(
                (batch_size,), float(route_count), dtype=torch.float64
            )
            return selections, route_counts, gates

        if gates is None:
            return None, None, None

        flat_gates = gates.reshape(batch_size, -1, num_experts)
        selections = (flat_gates > 0.0).to(torch.float64).sum(dim=1)
        route_counts = torch.full(
            (batch_size,), float(flat_gates.shape[1]), dtype=torch.float64
        )
        return selections, route_counts, gates

    def _accumulate_vector(
        self,
        statistics: Mapping[str, object],
        aliases: Sequence[str],
        target: torch.Tensor,
        num_experts: int,
    ) -> bool:
        value = _first_present(statistics, aliases)
        if value is None:
            return False
        vector = _to_cpu_float(value, aliases[0])  # type: ignore[arg-type]
        if vector.numel() != num_experts:
            raise ValueError(
                "{} must contain one value per expert (expected {}, got {})".format(
                    aliases[0], num_experts, vector.numel()
                )
            )
        if bool((vector < 0.0).any()):
            raise ValueError("{} must be non-negative".format(aliases[0]))
        target.add_(vector.reshape(num_experts))
        return True

    def update(
        self,
        routing_stats: Mapping[str, Mapping[str, object]],
        degradation_multi_hot: torch.Tensor,
        degradation_label_mask: Optional[torch.Tensor] = None,
        compound_level: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one batch without retaining any batch-sized tensors.

        Args:
            routing_stats: Mapping from decoder stage name to statistics.  A
                stage should contain ``gates`` and/or ``topk_indices`` and may
                contain pre-top-k ``dense_probabilities``, ``load``,
                ``importance``, ``expert_flops``, and ``expert_parameters``.
            degradation_multi_hot: Multi-label targets ``[B, 19]`` by default.
            degradation_label_mask: Optional binary ``[B]`` mask.  Masked
                samples do not contribute to degradation or compound tables,
                but still contribute to global load/entropy statistics.
            compound_level: Optional integer ``[B]`` values: 0 means unknown,
                while 1/2/3 mean single/double/triple.  If omitted, the level
                is derived from the active label count for backward
                compatibility.
        """

        if degradation_multi_hot.ndim != 2:
            raise ValueError("degradation_multi_hot must have shape [B, num_classes]")
        if degradation_multi_hot.shape[1] != self.num_degradation_classes:
            raise ValueError(
                "Expected {} degradation classes, got {}".format(
                    self.num_degradation_classes,
                    degradation_multi_hot.shape[1],
                )
            )
        raw_labels = _to_cpu_float(
            degradation_multi_hot, "degradation_multi_hot"
        )
        if bool(((raw_labels < 0.0) | (raw_labels > 1.0)).any()):
            raise ValueError("degradation_multi_hot values must be in [0, 1]")
        batch_size = int(raw_labels.shape[0])

        if degradation_label_mask is None:
            label_mask = torch.ones(batch_size, dtype=torch.float64)
        else:
            label_mask = _to_cpu_float(
                degradation_label_mask, "degradation_label_mask"
            )
            if label_mask.ndim != 1:
                raise ValueError("degradation_label_mask must have shape [B]")
            if label_mask.numel() != batch_size:
                raise ValueError(
                    "degradation_label_mask must contain one value per sample"
                )
            if not bool(((label_mask == 0.0) | (label_mask == 1.0)).all()):
                raise ValueError("degradation_label_mask must be binary")

        labels = raw_labels.gt(0.0).to(torch.float64)
        labels.mul_(label_mask.unsqueeze(1))

        if compound_level is None:
            levels = raw_labels.gt(0.0).sum(dim=1).to(torch.int64)
            levels = torch.where(
                (levels >= 1) & (levels <= 3), levels, torch.zeros_like(levels)
            )
        else:
            raw_levels = _to_cpu_float(compound_level, "compound_level")
            if raw_levels.ndim != 1:
                raise ValueError("compound_level must have shape [B]")
            if raw_levels.numel() != batch_size:
                raise ValueError("compound_level must contain one value per sample")
            if not bool(raw_levels.eq(raw_levels.round()).all()):
                raise ValueError("compound_level must contain integers")
            if bool(((raw_levels < 0.0) | (raw_levels > 3.0)).any()):
                raise ValueError("compound_level values must be in {0, 1, 2, 3}")
            levels = raw_levels.to(torch.int64)
        compound = _zeros(batch_size, 3)
        for level in range(1, 4):
            compound[:, level - 1] = (
                (levels == level).to(torch.float64) * label_mask
            )

        for raw_stage_name, statistics in routing_stats.items():
            stage_name = str(raw_stage_name)
            if not isinstance(statistics, Mapping):
                raise TypeError("Routing statistics for each stage must be a mapping")
            state = self._stage(stage_name, statistics)
            selections, route_counts, gates = self._selection_counts(
                statistics, batch_size, state.num_experts
            )
            entropy_probabilities = self._entropy_probabilities(
                statistics, batch_size, state.num_experts, gates
            )

            if selections is not None and route_counts is not None:
                selection_slots = selections.sum(dim=1)
                state.selection_sum.add_(selections.sum(dim=0))
                state.routing_decisions += float(route_counts.sum().item())

                state.degradation_selection.add_(labels.transpose(0, 1).matmul(selections))
                state.degradation_samples.add_(labels.sum(dim=0))
                state.degradation_slots.add_(
                    labels.transpose(0, 1).matmul(selection_slots)
                )

                state.compound_selection.add_(compound.transpose(0, 1).matmul(selections))
                state.compound_samples.add_(compound.sum(dim=0))
                state.compound_slots.add_(
                    compound.transpose(0, 1).matmul(selection_slots)
                )

            load_present = self._accumulate_vector(
                statistics, ("load",), state.load_sum, state.num_experts
            )
            if not load_present and selections is not None:
                state.load_sum.add_(selections.sum(dim=0))

            importance_present = self._accumulate_vector(
                statistics,
                ("importance",),
                state.importance_sum,
                state.num_experts,
            )
            if not importance_present:
                if entropy_probabilities is not None:
                    flat_probabilities = entropy_probabilities.reshape(
                        -1, state.num_experts
                    )
                    state.importance_sum.add_(flat_probabilities.sum(dim=0))
                elif selections is not None:
                    state.importance_sum.add_(selections.sum(dim=0))

            if self._accumulate_vector(
                statistics,
                ("expert_flops", "average_flops", "flops"),
                state.flops_sum,
                state.num_experts,
            ):
                state.flops_updates += 1
            if self._accumulate_vector(
                statistics,
                ("expert_parameters", "expert_params", "parameters"),
                state.parameters_sum,
                state.num_experts,
            ):
                state.parameters_updates += 1

            if entropy_probabilities is not None:
                probabilities = entropy_probabilities.reshape(
                    -1, state.num_experts
                )
                totals = probabilities.sum(dim=-1, keepdim=True)
                valid = totals.squeeze(-1) > self.eps
                if bool(valid.any()):
                    probabilities = probabilities[valid] / totals[valid].clamp_min(self.eps)
                    entropy = -(
                        probabilities
                        * probabilities.clamp_min(self.eps).log()
                    ).sum(dim=-1)
                    state.entropy_sum += float(entropy.sum().item())
                    state.entropy_square_sum += float(entropy.square().sum().item())
                    state.entropy_count += int(entropy.numel())
                    state.entropy_min = min(state.entropy_min, float(entropy.min().item()))
                    state.entropy_max = max(state.entropy_max, float(entropy.max().item()))

            state.update_count += 1

    def _require_stage(self, stage_name: str) -> _StageAccumulator:
        try:
            return self._stages[stage_name]
        except KeyError as error:
            raise KeyError(
                "Unknown routing stage {!r}; available stages: {}".format(
                    stage_name, ", ".join(self.stage_names) or "<none>"
                )
            ) from error

    @staticmethod
    def _normalize_rows(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        denominator = denominator.unsqueeze(-1)
        return torch.where(
            denominator > 0.0,
            numerator / denominator.clamp_min(1.0e-12),
            torch.zeros_like(numerator),
        )

    def degradation_matrix(
        self, stage_name: str, normalize: bool = True
    ) -> torch.Tensor:
        """Return degradation-by-expert counts or frequencies ``[19, E]``."""

        state = self._require_stage(stage_name)
        if normalize:
            return self._normalize_rows(
                state.degradation_selection, state.degradation_slots
            )
        return state.degradation_selection.clone()

    def compound_matrix(
        self, stage_name: str, normalize: bool = True
    ) -> torch.Tensor:
        """Return single/double/triple-by-expert ``[3, E]`` statistics."""

        state = self._require_stage(stage_name)
        if normalize:
            return self._normalize_rows(state.compound_selection, state.compound_slots)
        return state.compound_selection.clone()

    @staticmethod
    def _fraction(vector: torch.Tensor) -> torch.Tensor:
        total = float(vector.sum().item())
        return vector / total if total > 0.0 else torch.zeros_like(vector)

    @staticmethod
    def _mean_vector(vector: torch.Tensor, updates: int) -> torch.Tensor:
        return vector / float(updates) if updates > 0 else torch.zeros_like(vector)

    @staticmethod
    def _configured_experts_state(
        configured: Optional[Union[int, Mapping[str, int]]]
    ) -> Optional[Union[int, Dict[str, int]]]:
        if isinstance(configured, Mapping):
            return {str(name): int(value) for name, value in configured.items()}
        return configured

    @staticmethod
    def _stage_shapes(num_classes: int, num_experts: int) -> Dict[str, Tuple[int, ...]]:
        return {
            "degradation_selection": (num_classes, num_experts),
            "degradation_samples": (num_classes,),
            "degradation_slots": (num_classes,),
            "compound_selection": (3, num_experts),
            "compound_samples": (3,),
            "compound_slots": (3,),
            "selection_sum": (num_experts,),
            "load_sum": (num_experts,),
            "importance_sum": (num_experts,),
            "flops_sum": (num_experts,),
            "parameters_sum": (num_experts,),
        }

    @staticmethod
    def _clone_stage(stage: _StageAccumulator) -> _StageAccumulator:
        clone = _StageAccumulator.create(0, stage.num_experts)
        for name, value in vars(stage).items():
            setattr(clone, name, value.clone() if isinstance(value, torch.Tensor) else value)
        return clone

    def state_dict(self) -> Dict[str, Any]:
        """Return a finite, bounded state suitable for checkpoints.

        All tensors are detached CPU clones.  Empty entropy ranges are encoded
        as zeros instead of the internal ``+/-inf`` sentinels, which also makes
        this state directly convertible to standards-compliant JSON.
        """

        stages: Dict[str, Dict[str, Any]] = {}
        for stage_name in self.stage_names:
            stage = self._stages[stage_name]
            stage_state: Dict[str, Any] = {
                "num_experts": stage.num_experts,
                "flops_updates": stage.flops_updates,
                "parameters_updates": stage.parameters_updates,
                "update_count": stage.update_count,
                "routing_decisions": stage.routing_decisions,
                "entropy_sum": stage.entropy_sum,
                "entropy_square_sum": stage.entropy_square_sum,
                "entropy_count": stage.entropy_count,
                "entropy_min": stage.entropy_min if stage.entropy_count else 0.0,
                "entropy_max": stage.entropy_max if stage.entropy_count else 0.0,
            }
            for field_name in _STAGE_TENSOR_FIELDS:
                stage_state[field_name] = getattr(stage, field_name).clone()
            stages[stage_name] = stage_state
        return {
            "version": _STATE_VERSION,
            "num_degradation_classes": self.num_degradation_classes,
            "num_experts": self._configured_experts_state(self.num_experts),
            "degradation_names": list(self.degradation_names),
            "eps": self.eps,
            "stages": stages,
        }

    @staticmethod
    def _finite_scalar(value: object, name: str, minimum: float = 0.0) -> float:
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError("{} must be numeric".format(name)) from error
        if not math.isfinite(result):
            raise ValueError("{} contains NaN or Inf".format(name))
        if result < minimum:
            raise ValueError("{} must be at least {}".format(name, minimum))
        return result

    @staticmethod
    def _nonnegative_int(value: object, name: str) -> int:
        scalar = RoutingAnalysisAccumulator._finite_scalar(value, name)
        if scalar != math.floor(scalar):
            raise ValueError("{} must be an integer".format(name))
        return int(scalar)

    @staticmethod
    def _parse_num_experts_configuration(
        value: object,
    ) -> Optional[Union[int, Dict[str, int]]]:
        if value is None:
            return None
        if isinstance(value, Mapping):
            parsed: Dict[str, int] = {}
            for stage_name, count in value.items():
                parsed_count = RoutingAnalysisAccumulator._nonnegative_int(
                    count, "num_experts[{!r}]".format(stage_name)
                )
                if parsed_count <= 0:
                    raise ValueError("configured num_experts must be positive")
                parsed[str(stage_name)] = parsed_count
            return parsed
        parsed_count = RoutingAnalysisAccumulator._nonnegative_int(
            value, "num_experts"
        )
        if parsed_count <= 0:
            raise ValueError("configured num_experts must be positive")
        return parsed_count

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict` or ``export_json``."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("routing accumulator state must be a mapping")
        version = self._nonnegative_int(state_dict.get("version"), "version")
        if version != _STATE_VERSION:
            raise ValueError(
                "Unsupported routing accumulator state version {}".format(version)
            )
        num_classes = self._nonnegative_int(
            state_dict.get("num_degradation_classes"),
            "num_degradation_classes",
        )
        if num_classes <= 0:
            raise ValueError("num_degradation_classes must be positive")
        raw_names = state_dict.get("degradation_names")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            raise ValueError("degradation_names must be a sequence")
        names = tuple(str(name) for name in raw_names)
        if len(names) != num_classes:
            raise ValueError("degradation_names length does not match num classes")
        eps = self._finite_scalar(state_dict.get("eps"), "eps")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        configured_experts = self._parse_num_experts_configuration(
            state_dict.get("num_experts")
        )
        raw_stages = state_dict.get("stages")
        if not isinstance(raw_stages, Mapping):
            raise ValueError("stages must be a mapping")

        restored_stages: Dict[str, _StageAccumulator] = {}
        for raw_stage_name, raw_stage in raw_stages.items():
            stage_name = str(raw_stage_name)
            if not isinstance(raw_stage, Mapping):
                raise ValueError("state for {!r} must be a mapping".format(stage_name))
            num_stage_experts = self._nonnegative_int(
                raw_stage.get("num_experts"),
                "stages.{0}.num_experts".format(stage_name),
            )
            if num_stage_experts <= 0:
                raise ValueError("stage num_experts must be positive")
            configured_count = (
                configured_experts
                if isinstance(configured_experts, int)
                else (
                    configured_experts.get(stage_name)
                    if isinstance(configured_experts, Mapping)
                    else None
                )
            )
            if configured_count is not None and configured_count != num_stage_experts:
                raise ValueError(
                    "Configured expert count for {!r} does not match its state".format(
                        stage_name
                    )
                )
            stage = _StageAccumulator.create(num_classes, num_stage_experts)
            for field_name, expected_shape in self._stage_shapes(
                num_classes, num_stage_experts
            ).items():
                if field_name not in raw_stage:
                    raise ValueError(
                        "Missing stages.{0}.{1}".format(stage_name, field_name)
                    )
                tensor = _to_cpu_float(
                    raw_stage[field_name],  # type: ignore[arg-type]
                    "stages.{0}.{1}".format(stage_name, field_name),
                )
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        "stages.{0}.{1} must have shape {2}, got {3}".format(
                            stage_name, field_name, expected_shape, tuple(tensor.shape)
                        )
                    )
                if bool((tensor < 0.0).any()):
                    raise ValueError(
                        "stages.{0}.{1} must be non-negative".format(
                            stage_name, field_name
                        )
                    )
                setattr(stage, field_name, tensor.clone())

            for field_name in (
                "flops_updates",
                "parameters_updates",
                "update_count",
                "entropy_count",
            ):
                setattr(
                    stage,
                    field_name,
                    self._nonnegative_int(
                        raw_stage.get(field_name),
                        "stages.{0}.{1}".format(stage_name, field_name),
                    ),
                )
            for field_name in (
                "routing_decisions",
                "entropy_sum",
                "entropy_square_sum",
            ):
                setattr(
                    stage,
                    field_name,
                    self._finite_scalar(
                        raw_stage.get(field_name),
                        "stages.{0}.{1}".format(stage_name, field_name),
                    ),
                )
            if stage.entropy_count:
                stage.entropy_min = self._finite_scalar(
                    raw_stage.get("entropy_min"),
                    "stages.{0}.entropy_min".format(stage_name),
                )
                stage.entropy_max = self._finite_scalar(
                    raw_stage.get("entropy_max"),
                    "stages.{0}.entropy_max".format(stage_name),
                )
                if stage.entropy_min > stage.entropy_max:
                    raise ValueError("entropy_min cannot exceed entropy_max")
            else:
                # Validate serialized sentinels while preserving efficient
                # internal min/max updates after the next observation.
                self._finite_scalar(
                    raw_stage.get("entropy_min"),
                    "stages.{0}.entropy_min".format(stage_name),
                )
                self._finite_scalar(
                    raw_stage.get("entropy_max"),
                    "stages.{0}.entropy_max".format(stage_name),
                )
                stage.entropy_min = math.inf
                stage.entropy_max = -math.inf
            restored_stages[stage_name] = stage

        # Apply only after the entire state has passed validation.
        self.num_degradation_classes = num_classes
        self.num_experts = configured_experts
        self.degradation_names = names
        self.eps = eps
        self._stages = restored_stages

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, object]
    ) -> "RoutingAnalysisAccumulator":
        """Construct an accumulator from checkpoint or JSON state."""

        accumulator = cls()
        accumulator.load_state_dict(state_dict)
        return accumulator

    def merge_(self, other: "RoutingAnalysisAccumulator") -> "RoutingAnalysisAccumulator":
        """Merge another worker/rank summary into this accumulator in place."""

        if not isinstance(other, RoutingAnalysisAccumulator):
            raise TypeError("other must be a RoutingAnalysisAccumulator")
        if other is self:
            other = self.from_state_dict(self.state_dict())
        if self.num_degradation_classes != other.num_degradation_classes:
            raise ValueError("Cannot merge accumulators with different class counts")
        if self.degradation_names != other.degradation_names:
            raise ValueError("Cannot merge accumulators with different class names")
        if self.eps != other.eps:
            raise ValueError("Cannot merge accumulators with different eps values")

        for stage_name in other.stage_names:
            incoming = other._stages[stage_name]
            configured_count = self._configured_num_experts(stage_name)
            if configured_count is not None and configured_count != incoming.num_experts:
                raise ValueError(
                    "Cannot merge stage {!r}: configured expert count differs".format(
                        stage_name
                    )
                )
            current = self._stages.get(stage_name)
            if current is None:
                self._stages[stage_name] = self._clone_stage(incoming)
                continue
            if current.num_experts != incoming.num_experts:
                raise ValueError(
                    "Cannot merge stage {!r} with different expert counts".format(
                        stage_name
                    )
                )
            for field_name in _STAGE_TENSOR_FIELDS:
                getattr(current, field_name).add_(getattr(incoming, field_name))
            current.flops_updates += incoming.flops_updates
            current.parameters_updates += incoming.parameters_updates
            current.update_count += incoming.update_count
            current.routing_decisions += incoming.routing_decisions
            current.entropy_sum += incoming.entropy_sum
            current.entropy_square_sum += incoming.entropy_square_sum
            current.entropy_count += incoming.entropy_count
            if incoming.entropy_count:
                current.entropy_min = min(current.entropy_min, incoming.entropy_min)
                current.entropy_max = max(current.entropy_max, incoming.entropy_max)
        return self

    @staticmethod
    def _jsonable(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, Mapping):
            return {
                str(key): RoutingAnalysisAccumulator._jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RoutingAnalysisAccumulator._jsonable(item) for item in value]
        return value

    def export_json(self, output_path: Union[str, Path]) -> Path:
        """Export complete mergeable accumulator state as UTF-8 JSON.

        A directory path produces ``routing_summary.json``; a path ending in
        ``.json`` is used verbatim.  The file can be read with ``json.load``
        and passed directly to :meth:`from_state_dict`.
        """

        destination = Path(output_path)
        if destination.suffix.lower() != ".json":
            destination = destination / "routing_summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._jsonable(self.state_dict())
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        return destination

    def export_csv(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Export the four routing-analysis CSV files used by F-MoE."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths = {
            "routing_by_degradation": output_path / "routing_by_degradation.csv",
            "routing_by_compound_level": output_path / "routing_by_compound_level.csv",
            "expert_load": output_path / "expert_load.csv",
            "routing_entropy": output_path / "routing_entropy.csv",
        }
        self._export_degradation(paths["routing_by_degradation"])
        self._export_compound(paths["routing_by_compound_level"])
        self._export_load(paths["expert_load"])
        self._export_entropy(paths["routing_entropy"])
        return paths

    def _export_degradation(self, path: Path) -> None:
        fieldnames = (
            "stage",
            "degradation_id",
            "degradation_name",
            "expert_id",
            "selection_count",
            "labeled_samples",
            "selection_slots",
            "selection_frequency",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for stage_name in self.stage_names:
                state = self._stages[stage_name]
                frequencies = self.degradation_matrix(stage_name, normalize=True)
                for degradation_id in range(self.num_degradation_classes):
                    for expert_id in range(state.num_experts):
                        writer.writerow(
                            {
                                "stage": stage_name,
                                "degradation_id": degradation_id,
                                "degradation_name": self.degradation_names[degradation_id],
                                "expert_id": expert_id,
                                "selection_count": float(
                                    state.degradation_selection[degradation_id, expert_id]
                                ),
                                "labeled_samples": float(
                                    state.degradation_samples[degradation_id]
                                ),
                                "selection_slots": float(
                                    state.degradation_slots[degradation_id]
                                ),
                                "selection_frequency": float(
                                    frequencies[degradation_id, expert_id]
                                ),
                            }
                        )

    def _export_compound(self, path: Path) -> None:
        fieldnames = (
            "stage",
            "compound_level",
            "expert_id",
            "selection_count",
            "samples",
            "selection_slots",
            "selection_frequency",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for stage_name in self.stage_names:
                state = self._stages[stage_name]
                frequencies = self.compound_matrix(stage_name, normalize=True)
                for level_id, level_name in enumerate(_COMPOUND_LEVELS):
                    for expert_id in range(state.num_experts):
                        writer.writerow(
                            {
                                "stage": stage_name,
                                "compound_level": level_name,
                                "expert_id": expert_id,
                                "selection_count": float(
                                    state.compound_selection[level_id, expert_id]
                                ),
                                "samples": float(state.compound_samples[level_id]),
                                "selection_slots": float(state.compound_slots[level_id]),
                                "selection_frequency": float(
                                    frequencies[level_id, expert_id]
                                ),
                            }
                        )

    def _export_load(self, path: Path) -> None:
        fieldnames = (
            "stage",
            "expert_id",
            "routing_decisions",
            "selection_count",
            "selection_fraction",
            "load_sum",
            "load_mean_per_update",
            "load_fraction",
            "importance_sum",
            "importance_mean_per_update",
            "importance_fraction",
            "expert_flops",
            "expert_parameters",
            "estimated_selected_flops",
            "estimated_selected_parameters",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for stage_name in self.stage_names:
                state = self._stages[stage_name]
                selection_fraction = self._fraction(state.selection_sum)
                load_fraction = self._fraction(state.load_sum)
                importance_fraction = self._fraction(state.importance_sum)
                mean_flops = self._mean_vector(state.flops_sum, state.flops_updates)
                mean_parameters = self._mean_vector(
                    state.parameters_sum, state.parameters_updates
                )
                for expert_id in range(state.num_experts):
                    writer.writerow(
                        {
                            "stage": stage_name,
                            "expert_id": expert_id,
                            "routing_decisions": state.routing_decisions,
                            "selection_count": float(state.selection_sum[expert_id]),
                            "selection_fraction": float(selection_fraction[expert_id]),
                            "load_sum": float(state.load_sum[expert_id]),
                            "load_mean_per_update": float(
                                state.load_sum[expert_id] / max(state.update_count, 1)
                            ),
                            "load_fraction": float(load_fraction[expert_id]),
                            "importance_sum": float(state.importance_sum[expert_id]),
                            "importance_mean_per_update": float(
                                state.importance_sum[expert_id]
                                / max(state.update_count, 1)
                            ),
                            "importance_fraction": float(
                                importance_fraction[expert_id]
                            ),
                            "expert_flops": (
                                float(mean_flops[expert_id])
                                if state.flops_updates
                                else ""
                            ),
                            "expert_parameters": (
                                float(mean_parameters[expert_id])
                                if state.parameters_updates
                                else ""
                            ),
                            "estimated_selected_flops": (
                                float(
                                    mean_flops[expert_id]
                                    * state.selection_sum[expert_id]
                                )
                                if state.flops_updates
                                else ""
                            ),
                            "estimated_selected_parameters": (
                                float(
                                    mean_parameters[expert_id]
                                    * state.selection_sum[expert_id]
                                )
                                if state.parameters_updates
                                else ""
                            ),
                        }
                    )

    def _export_entropy(self, path: Path) -> None:
        fieldnames = (
            "stage",
            "routing_decisions",
            "entropy_observations",
            "entropy_mean",
            "entropy_std",
            "entropy_min",
            "entropy_max",
            "effective_experts",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for stage_name in self.stage_names:
                state = self._stages[stage_name]
                if state.entropy_count:
                    mean = state.entropy_sum / state.entropy_count
                    variance = max(
                        state.entropy_square_sum / state.entropy_count - mean * mean,
                        0.0,
                    )
                    entropy_min = state.entropy_min
                    entropy_max = state.entropy_max
                else:
                    mean = 0.0
                    variance = 0.0
                    entropy_min = 0.0
                    entropy_max = 0.0
                writer.writerow(
                    {
                        "stage": stage_name,
                        "routing_decisions": state.routing_decisions,
                        "entropy_observations": state.entropy_count,
                        "entropy_mean": mean,
                        "entropy_std": math.sqrt(variance),
                        "entropy_min": entropy_min,
                        "entropy_max": entropy_max,
                        "effective_experts": math.exp(mean),
                    }
                )

    def plot_heatmap(
        self,
        stage_name: str,
        output_path: Union[str, Path],
        kind: str = "degradation",
        normalize: bool = True,
        title: Optional[str] = None,
    ) -> Path:
        """Render a degradation or compound routing heatmap with matplotlib."""

        if kind == "degradation":
            matrix = self.degradation_matrix(stage_name, normalize=normalize)
            row_labels = self.degradation_names
        elif kind == "compound":
            matrix = self.compound_matrix(stage_name, normalize=normalize)
            row_labels = _COMPOUND_LEVELS
        else:
            raise ValueError("kind must be 'degradation' or 'compound'")
        return plot_routing_heatmap(
            matrix,
            output_path,
            row_labels=row_labels,
            title=title or "{} routing: {}".format(kind, stage_name),
        )


# A concise alias for call sites that already use this common name.
RoutingAccumulator = RoutingAnalysisAccumulator
RoutingStatsAccumulator = RoutingAnalysisAccumulator


def plot_routing_heatmap(
    matrix: TensorLike,
    output_path: Union[str, Path],
    row_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
) -> Path:
    """Save an expert-selection heatmap.

    Matplotlib is imported lazily so routing accumulation and CSV export do not
    require a plotting backend on training workers.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on deployment extras
        raise RuntimeError(
            "matplotlib is required only for routing heatmap generation"
        ) from error

    values = matrix if isinstance(matrix, torch.Tensor) else torch.as_tensor(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must have shape [rows, experts]")
    values = _to_cpu_float(values, "matrix").to(dtype=torch.float32)
    if row_labels is not None and len(row_labels) != values.shape[0]:
        raise ValueError("row_labels length must match the matrix row count")

    width = max(5.0, 0.8 * float(values.shape[1]))
    height = max(3.0, 0.35 * float(values.shape[0]) + 1.5)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(values.numpy(), aspect="auto", interpolation="nearest", cmap=cmap)
    axis.set_xlabel("Expert")
    axis.set_ylabel("Degradation group")
    axis.set_xticks(range(values.shape[1]))
    if row_labels is not None:
        axis.set_yticks(range(values.shape[0]))
        axis.set_yticklabels(list(row_labels))
    if title:
        axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Selection frequency")
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


__all__ = [
    "RoutingAccumulator",
    "RoutingAnalysisAccumulator",
    "RoutingStatsAccumulator",
    "plot_routing_heatmap",
]
