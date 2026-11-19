"""Strict checkpoint loading, warm starts, and auditable training resumes."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Mapping, MutableMapping, Tuple

import torch
import torch.nn as nn


_DERIVED_BUFFER_SUFFIXES: Tuple[str, ...] = (
    ".routing.complexity",
    ".routing.complexity_bias",
    ".routing.importance_ema",
    ".routing.load_ema",
)

_STEP_PATTERN = re.compile(r"(?:^|-)step=(?P<step>\d+)(?:-|\.|$)")
_BEST_SCORE_PATTERN = re.compile(
    r"(?:^|-)val_restoration=(?P<score>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _step_from_name(path: Path) -> int:
    match = _STEP_PATTERN.search(path.name)
    return int(match.group("step")) if match else -1


def resolve_checkpoint_path(
    *,
    ckpt_dir: str,
    reference: str | None = None,
    explicit_path: str | None = None,
    selection: str = "latest",
) -> str:
    """Resolve an explicit file or a run directory without silent fallback.

    ``best`` requires validation scores embedded in checkpoint filenames.  It
    deliberately refuses legacy ``best-step=...`` files whose relative quality
    cannot be inferred, instead of pretending the newest checkpoint is best.
    """

    if selection not in {"best", "latest", "last"}:
        raise ValueError("selection must be best, latest, or last")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return str(path.resolve())
    if not reference:
        raise ValueError("checkpoint reference is required when no explicit path is given")

    raw_reference = Path(reference).expanduser()
    direct_candidates = (raw_reference, Path(ckpt_dir) / raw_reference)
    for candidate in direct_candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    run_dir = next((path for path in direct_candidates if path.is_dir()), None)
    if run_dir is None:
        raise FileNotFoundError(
            f"Checkpoint run directory not found for {reference!r} under {ckpt_dir!r}"
        )

    last_path = run_dir / "last.ckpt"
    if selection == "last":
        if not last_path.is_file():
            raise FileNotFoundError(f"last.ckpt not found in {run_dir}")
        return str(last_path.resolve())

    if selection == "latest":
        if last_path.is_file():
            return str(last_path.resolve())
        latest = sorted(
            run_dir.glob("latest-step=*.ckpt"), key=_step_from_name, reverse=True
        )
        if not latest:
            raise FileNotFoundError(f"No resumable checkpoint found in {run_dir}")
        return str(latest[0].resolve())

    scored = []
    unscored = []
    for path in run_dir.glob("best-step=*.ckpt"):
        match = _BEST_SCORE_PATTERN.search(path.name)
        if match:
            scored.append((float(match.group("score")), _step_from_name(path), path))
        else:
            unscored.append(path)
    if not scored:
        detail = (
            f"; found {len(unscored)} legacy unscored best checkpoint(s)"
            if unscored
            else ""
        )
        raise FileNotFoundError(
            f"No score-bearing best checkpoint found in {run_dir}{detail}. "
            "Pass --checkpoint_path explicitly for legacy runs."
        )
    _, _, best_path = min(scored, key=lambda item: (item[0], -item[1]))
    return str(best_path.resolve())


def _legacy_objective_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Reconstruct objective fields for checkpoints created before contract v2."""
    loss_type = str(config.get("loss_type", "")).lower()
    if loss_type not in {"audio", "audio_multi"}:
        return {}

    lambda_balance = config.get("lambda_balance")
    if lambda_balance is None:
        lambda_balance = config.get("balance_loss_weight", 0.1)

    return {
        "objective": {
            "name": "fmoe_5loss_v1",
            "weights": {
                "complex": float(config.get("lambda_complex", 1.0)),
                "wave": float(config.get("lambda_wave", 1.0)),
                "mrstft": float(config.get("lambda_mrstft", 1.0)),
                "degradation": float(config.get("lambda_degradation", 0.1)),
                "balance": float(lambda_balance),
            },
            "stft": {
                "n_fft": int(config.get("n_fft", 1024)),
                "hop_length": int(config.get("hop_length", 256)),
                "win_length": int(config.get("win_length", 1024)),
                "center": bool(config.get("stft_center", True)),
            },
            "mrstft": {
                "fft_sizes": [int(value) for value in config.get(
                    "mrstft_fft_sizes", [512, 1024, 2048, 4096]
                )],
                "hop_sizes": [int(value) for value in config.get(
                    "mrstft_hop_sizes", [128, 256, 512, 1024]
                )],
                "win_lengths": [int(value) for value in config.get(
                    "mrstft_win_lengths", [512, 1024, 2048, 4096]
                )],
                "spectral_convergence_normalization": str(
                    config.get("mrstft_sc_normalization", "symmetric")
                ),
            },
            "check_finite": bool(config.get("loss_check_finite", True)),
        },
        "audio": {
            "sample_rate": int(config.get("sample_rate", 44100)),
            "segment_seconds": float(config.get("segment_seconds", 4.0)),
            "representation": str(config.get("representation", "complex")),
            "stereo_mode": str(config.get("stereo_mode", "mono")),
            "spec_log": bool(config.get("spec_log", True)),
        },
    }


def validate_resume_training_contract(
    checkpoint_path: str,
    current_contract: Mapping[str, object],
) -> Dict[str, object]:
    """Fail closed when an exact resume would change optimizer-step semantics.

    Lightning restores the scheduler's ``last_epoch`` but reconstructs its
    lambda from the current run.  Changing a horizon, world size, batch
    contract, or optimizer settings would therefore create a hybrid schedule
    that is neither the old run nor a clean fine-tune.
    """

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": False,
        "mmap": True,
    }
    try:
        checkpoint = torch.load(path, **load_kwargs)
    except TypeError:  # Compatibility with older supported torch releases.
        load_kwargs.pop("weights_only")
        load_kwargs.pop("mmap")
        checkpoint = torch.load(path, **load_kwargs)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    saved_contract = hyper_parameters.get("training_contract")
    if not isinstance(saved_contract, Mapping):
        raise RuntimeError(
            "Checkpoint has no auditable training_contract and cannot be resumed "
            "exactly. Use --fine_tune_from for a weights-only warm start."
        )

    # Older full-cover checkpoints stored the complete parsed config in
    # hyper_parameters.config but omitted objective/audio fields from the
    # training_contract. Reconstruct them so unchanged runs remain resumable.
    saved_contract = dict(saved_contract)
    saved_config = hyper_parameters.get("config", {})
    if isinstance(saved_config, Mapping):
        for key, value in _legacy_objective_from_config(saved_config).items():
            saved_contract.setdefault(key, value)

    missing = sorted(set(current_contract) - set(saved_contract))
    changed = {
        key: {"checkpoint": saved_contract[key], "current": current_contract[key]}
        for key in sorted(set(current_contract) & set(saved_contract))
        if saved_contract[key] != current_contract[key]
    }
    if missing or changed:
        raise RuntimeError(
            "Resume training contract mismatch; exact scheduler/optimizer state "
            f"would be invalid. missing={missing}, changed={changed}. "
            "Use the original configuration or --fine_tune_from."
        )
    return {
        "checkpoint_path": str(path.resolve()),
        "validated_keys": sorted(current_contract),
        "training_contract": dict(saved_contract),
    }


def _strip_distributed_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _target_key(source_key: str, target: Mapping[str, torch.Tensor]) -> str:
    """Resolve raw-model, Lightning, and DDP key prefixes conservatively."""

    key = _strip_distributed_prefix(source_key)
    candidates = (key, f"net.{key}", key[4:] if key.startswith("net.") else "")
    for candidate in candidates:
        if candidate and candidate in target:
            return candidate
    return key


def normalize_legacy_routing_state_dict(
    source_state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Adapt checkpoints from feature-plus-condition routing to global-z_deg routing.

    The old local-feature gate is intentionally discarded.  The former
    degradation projection is retained as the new condition-only projection,
    since both map the same global condition vector to expert logits.
    """

    normalized: Dict[str, torch.Tensor] = {}
    discarded = []
    for raw_key, value in source_state.items():
        key = _strip_distributed_prefix(str(raw_key))
        if ".routing.gate." in key or key.startswith("routing.gate."):
            discarded.append(str(raw_key))
            continue
        key = key.replace(
            ".routing.freq_gate.", ".routing.condition_projection."
        )
        if key.startswith("routing.freq_gate."):
            key = key.replace(
                "routing.freq_gate.", "routing.condition_projection.", 1
            )
        normalized[key] = value
    return normalized, tuple(sorted(discarded))


def migrate_state_dict(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Warm-start ``module`` from compatible tensors and return an audit report.

    The migration is intentionally shape-safe.  New heads/experts remain at
    their initialized values, obsolete tensors are ignored, and routing
    complexity buffers are always recomputed by the target architecture.
    This is for cross-phase fine-tuning; exact training resume remains
    Lightning's strict checkpoint path.
    """

    target_state = module.state_dict()
    normalized_source, skipped_legacy_routing = normalize_legacy_routing_state_dict(
        source_state
    )
    compatible: MutableMapping[str, torch.Tensor] = {}
    skipped_missing = []
    skipped_shape = []
    skipped_derived = []
    for raw_key, value in normalized_source.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = _target_key(str(raw_key), target_state)
        if key.endswith(_DERIVED_BUFFER_SUFFIXES):
            skipped_derived.append(key)
            continue
        target_value = target_state.get(key)
        if target_value is None:
            skipped_missing.append(str(raw_key))
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped_shape.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        compatible[key] = value

    incompatible = module.load_state_dict(compatible, strict=False)
    report: Dict[str, object] = {
        "loaded_tensor_count": len(compatible),
        "source_tensor_count": sum(
            isinstance(value, torch.Tensor) for value in source_state.values()
        ),
        "target_tensor_count": len(target_state),
        "skipped_missing_keys": sorted(skipped_missing),
        "skipped_shape": sorted(skipped_shape, key=lambda item: str(item["key"])),
        "skipped_derived_buffers": sorted(set(skipped_derived)),
        "skipped_legacy_feature_router_keys": list(skipped_legacy_routing),
        "target_uninitialized_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
    }
    return report


def load_compatible_checkpoint(
    module: nn.Module,
    checkpoint_path: str,
) -> Dict[str, object]:
    """Load a raw or Lightning checkpoint through :func:`migrate_state_dict`."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")
    report = migrate_state_dict(module, state)  # type: ignore[arg-type]
    report["checkpoint_path"] = str(path.resolve())
    return report


def load_strict_model_checkpoint(
    module: nn.Module,
    checkpoint_path: str,
) -> Dict[str, object]:
    """Strictly load inference tensors from a raw or Lightning checkpoint.

    Training checkpoints may contain loss-module state and the training-only
    degradation classifier. Those tensors are intentionally omitted from the
    inference model. Legacy local-feature router weights are also discarded;
    their global-condition projection is migrated to the z_deg-only router.
    All remaining restoration, conditioning, and routing tensors remain strict:
    missing, extra, or shape-incompatible model keys fail loudly.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Inference checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")

    target = module.state_dict()
    normalized_source, skipped_legacy_routing = normalize_legacy_routing_state_dict(
        source
    )
    selected: Dict[str, torch.Tensor] = {}
    unexpected_model_keys = []
    lightning_model_prefix = any(
        _strip_distributed_prefix(str(key)).startswith("net.")
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
    )
    for raw_key, value in normalized_source.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = _strip_distributed_prefix(str(raw_key))
        key = _target_key(normalized_key, target)
        if key in target:
            selected[key] = value
        elif normalized_key.startswith(
            (
                "net.degradation_encoder.degradation_head.",
                "degradation_encoder.degradation_head.",
            )
        ):
            # The multi-label degradation classifier is training-only. It is
            # intentionally absent from the inference architecture, while all
            # restoration and routing tensors remain subject to strict loading.
            continue
        elif normalized_key.startswith("net.") or not lightning_model_prefix:
            unexpected_model_keys.append(str(raw_key))
    missing = sorted(set(target) - set(selected))
    if missing or unexpected_model_keys:
        raise RuntimeError(
            "Strict inference checkpoint mismatch: missing={} unexpected_model={}".format(
                missing, sorted(unexpected_model_keys)
            )
        )
    module.load_state_dict(selected, strict=True)
    return {
        "checkpoint_path": str(path.resolve()),
        "loaded_tensor_count": len(selected),
        "ignored_training_tensor_count": sum(
            isinstance(value, torch.Tensor) for value in source.values()
        )
        - len(selected),
        "ignored_legacy_feature_router_tensor_count": len(
            skipped_legacy_routing
        ),
    }


__all__ = [
    "load_compatible_checkpoint",
    "load_strict_model_checkpoint",
    "migrate_state_dict",
    "normalize_legacy_routing_state_dict",
    "resolve_checkpoint_path",
    "validate_resume_training_contract",
]
