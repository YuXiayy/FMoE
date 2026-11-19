"""Single source of truth for constructing the released F-MoE architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from net.fmoe import FMoE
from utils.checkpoint import (
    load_strict_model_checkpoint,
    normalize_legacy_routing_state_dict,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHITECTURE = ROOT / "configs" / "fmoe_architecture.yaml"


def fmoe_kwargs(opt, *, inference: bool = False) -> Dict[str, object]:
    """Return the complete model contract encoded by ``opt``.

    Inference omits the training-only classifier and disables diagnostic/FLOP
    collection. Restoration, routing parameters, and persistent EMA buffers
    retain the exact released checkpoint schema.
    """

    return {
        "inp_channels": opt.inp_channels,
        "out_channels": opt.out_channels,
        "dim": opt.dim,
        "num_blocks": opt.num_blocks,
        "num_dec_blocks": opt.num_dec_blocks,
        "levels": len(opt.num_blocks),
        "heads": opt.heads,
        "num_refinement_blocks": opt.num_refinement_blocks,
        "num_experts": opt.num_exp_blocks,
        "input_freq_bins": opt.input_freq_bins,
        "expert_embed_dims": opt.expert_embed_dims,
        "expert_num_bands": opt.expert_num_bands,
        "expert_depths": opt.expert_depths,
        "expert_num_heads": opt.expert_num_heads,
        "sample_rate": opt.sample_rate,
        "n_fft": opt.n_fft,
        "degradation_embed_dim": opt.degradation_embed_dim,
        "num_degradation_classes": opt.num_degradation_classes,
        "include_training_head": not inference,
        "routing_balance_ema_decay": opt.routing_balance_ema_decay,
        "enable_routing_logging": (
            bool(opt.enable_routing_logging) and not inference
        ),
        "profile_expert_flops": bool(opt.profile_expert_flops) and not inference,
    }


def build_fmoe(opt, *, inference: bool = False) -> FMoE:
    """Construct F-MoE from the canonical option mapping."""

    return FMoE(**fmoe_kwargs(opt, inference=inference))


def build_inference_model(
    config_path: Path | str = DEFAULT_ARCHITECTURE,
) -> FMoE:
    """Build the inference architecture from the public YAML contract."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("architecture config must contain a 'model' mapping")
    return FMoE(**config["model"])


def load_pretrained_fmoe(
    checkpoint_path: Path | str,
    config_path: Path | str = DEFAULT_ARCHITECTURE,
    device: str | torch.device = "cpu",
) -> Tuple[FMoE, Dict[str, Any]]:
    """Load a public inference checkpoint with strict shape validation."""

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a mapping")
    model = build_inference_model(config_path)
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("model_state_dict must be a tensor mapping")
        state_dict, _ = normalize_legacy_routing_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
    elif "state_dict" in payload:
        load_strict_model_checkpoint(model, str(checkpoint_path))
    else:
        raise ValueError(
            "expected a public checkpoint with 'model_state_dict' or a "
            "Lightning checkpoint with 'state_dict'"
        )
    model = model.to(device).eval()

    metadata = payload.get("metadata", payload.get("hyper_parameters", {}))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    return model, metadata


__all__ = [
    "build_fmoe",
    "build_inference_model",
    "fmoe_kwargs",
    "load_pretrained_fmoe",
]
