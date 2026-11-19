from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.utils.checkpoint import load_strict_model_checkpoint
from src.net import model_factory


class _InferenceWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 2)


def test_strict_loader_ignores_training_only_degradation_classifier(tmp_path: Path):
    model = _InferenceWrapper()
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    state.update(
        {
            "net.degradation_encoder.degradation_head.weight": torch.randn(19, 128),
            "net.degradation_encoder.degradation_head.bias": torch.randn(19),
            "loss_module.some_buffer": torch.tensor(1.0),
        }
    )
    checkpoint = tmp_path / "training.ckpt"
    torch.save({"state_dict": state}, checkpoint)

    report = load_strict_model_checkpoint(model, str(checkpoint))

    assert report["loaded_tensor_count"] == len(model.state_dict())
    assert report["ignored_training_tensor_count"] == 3


def test_strict_loader_still_rejects_unexpected_restoration_parameters(tmp_path: Path):
    model = _InferenceWrapper()
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    state["net.unexpected_restoration.weight"] = torch.randn(2, 2)
    checkpoint = tmp_path / "incompatible.ckpt"
    torch.save({"state_dict": state}, checkpoint)

    with pytest.raises(RuntimeError, match="unexpected_model"):
        load_strict_model_checkpoint(model, str(checkpoint))


class _GlobalRoutingWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.routing = nn.Module()
        self.net.routing.condition_projection = nn.Linear(3, 2, bias=False)


def test_strict_loader_migrates_legacy_feature_conditioned_router(tmp_path: Path):
    model = _GlobalRoutingWrapper()
    projection = torch.randn_like(model.net.routing.condition_projection.weight)
    legacy_state = {
        "net.routing.freq_gate.weight": projection.clone(),
        "net.routing.gate.2.weight": torch.randn(2, 5),
    }
    checkpoint = tmp_path / "legacy_router.ckpt"
    torch.save({"state_dict": legacy_state}, checkpoint)

    report = load_strict_model_checkpoint(model, str(checkpoint))

    torch.testing.assert_close(
        model.net.routing.condition_projection.weight, projection
    )
    assert report["loaded_tensor_count"] == 1
    assert report["ignored_legacy_feature_router_tensor_count"] == 1


def test_public_loader_accepts_a_lightning_training_checkpoint(
    tmp_path: Path, monkeypatch
):
    model = _GlobalRoutingWrapper()
    source = {
        f"net.{key}": value.detach().clone()
        for key, value in model.state_dict().items()
    }
    source["net.degradation_encoder.degradation_head.weight"] = torch.randn(19, 3)
    checkpoint = tmp_path / "trained.ckpt"
    torch.save(
        {
            "state_dict": source,
            "hyper_parameters": {"run_id": "test"},
        },
        checkpoint,
    )
    monkeypatch.setattr(
        model_factory, "build_inference_model", lambda _config_path: model
    )

    loaded, metadata = model_factory.load_pretrained_fmoe(checkpoint)

    assert loaded is model
    assert metadata == {"run_id": "test"}
