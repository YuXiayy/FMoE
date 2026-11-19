import pytest
import torch

from net.degradation_encoder import (
    CompoundDegradationEncoder,
    StereoCueEncoder,
)


def test_compound_encoder_stereo_shape_gradient_and_batch_safety():
    torch.manual_seed(0)
    encoder = CompoundDegradationEncoder(
        degradation_embed_dim=24,
        branch_dim=8,
    )
    channels = torch.randn(4, 4, 17, 11, requires_grad=True)

    embedding, logits = encoder(channels)

    assert embedding.shape == (4, 24)
    assert logits is None
    assert torch.isfinite(embedding).all()

    loss = embedding.square().mean()
    loss.backward()
    assert channels.grad is not None
    assert torch.isfinite(channels.grad).all()
    assert channels.grad.abs().sum() > 0


def test_same_encoder_supports_mono_and_stereo():
    encoder = CompoundDegradationEncoder(
        degradation_embed_dim=20,
        branch_dim=8,
    )
    mono = torch.randn(3, 2, 9, 7, requires_grad=True)

    embedding, logits = encoder(mono)

    assert embedding.shape == (3, 20)
    assert logits is None
    embedding.sum().backward()
    assert mono.grad is not None
    assert torch.isfinite(mono.grad).all()

    stereo = torch.randn(2, 4, 9, 7)
    stereo_embedding, stereo_logits = encoder(stereo)
    assert stereo_embedding.shape == (2, 20)
    assert stereo_logits is None


def test_stereo_statistics_match_mid_side_and_phase_definitions():
    stereo_encoder = StereoCueEncoder(output_dim=8)
    left = torch.ones(2, 5, 4, dtype=torch.complex64) * (1.0 + 1.0j)
    identical_stereo = torch.stack((left, left), dim=1)

    statistics = stereo_encoder.statistics(identical_stereo)

    assert statistics.shape == (2, 6)
    assert torch.allclose(statistics[:, 1], torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(statistics[:, 2], torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(statistics[:, 3], torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(statistics[:, 4], torch.ones(2), atol=1.0e-6)
    assert torch.allclose(statistics[:, 5], torch.zeros(2), atol=1.0e-6)

    mono_statistics = stereo_encoder.statistics(left.unsqueeze(1))
    assert torch.equal(mono_statistics, torch.zeros_like(mono_statistics))
    mono_embedding = stereo_encoder(left.unsqueeze(1))
    assert torch.equal(mono_embedding, torch.zeros_like(mono_embedding))


def test_nonfinite_input_is_sanitized_and_mixed_precision_is_finite():
    encoder = CompoundDegradationEncoder(
        degradation_embed_dim=16,
        branch_dim=8,
    ).eval()
    channels = torch.randn(2, 4, 9, 7)
    channels[0, 0, 0, 0] = float("nan")
    channels[0, 1, 0, 1] = float("inf")
    channels[1, 2, 1, 0] = float("-inf")

    embedding, logits = encoder(channels)
    assert torch.isfinite(embedding).all()
    assert logits is None

    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed_embedding, mixed_logits = encoder(channels)
    except RuntimeError as error:  # pragma: no cover - old CPU builds only
        pytest.skip("CPU bfloat16 autocast is unavailable: {}".format(error))
    assert torch.isfinite(mixed_embedding).all()
    assert mixed_logits is None


def test_training_head_is_multilabel_and_backpropagates():
    encoder = CompoundDegradationEncoder(
        degradation_embed_dim=16,
        branch_dim=8,
        num_degradation_classes=19,
        include_training_head=True,
    )
    channels = torch.randn(2, 4, 9, 7, requires_grad=True)
    embedding, logits = encoder(channels)

    assert embedding.shape == (2, 16)
    assert logits is not None and logits.shape == (2, 19)
    logits.square().mean().backward()
    assert encoder.degradation_head is not None
    assert encoder.degradation_head.weight.grad is not None


def test_rejects_noncanonical_channels():
    encoder = CompoundDegradationEncoder(degradation_embed_dim=16, branch_dim=8)
    with pytest.raises(ValueError, match="mono"):
        encoder(torch.randn(2, 3, 8, 6))
