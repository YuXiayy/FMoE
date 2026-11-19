import pytest
import torch

from src.net.band_sequence import (
    BandProjection,
    BandReconstruction,
    BandSequenceExpert,
    BandSpec,
    build_multiresolution_bands,
    validate_bands,
)


def _intervals(bands):
    return [(band.start_bin, band.end_bin) for band in bands]


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def test_band_spec_is_half_open_and_rejects_empty_intervals():
    band = BandSpec(2, 5)
    assert band.width == 3
    assert list(range(band.start_bin, band.end_bin)) == [2, 3, 4]
    with pytest.raises(ValueError):
        BandSpec(3, 3)


def test_multiresolution_bands_are_complete_and_include_nyquist_tail():
    # n_fft=16 gives nine one-sided bins at 1 kHz spacing, including 8 kHz.
    bands = build_multiresolution_bands(
        sample_rate=16000,
        n_fft=16,
        boundaries_hz=[2000, 4000, 6000],
        bins_per_band=[1, 2, 3],
    )
    assert _intervals(bands) == [(0, 1), (1, 2), (2, 4), (4, 6), (6, 9)]
    assert bands[-1].end_bin == 16 // 2 + 1
    validate_bands(bands, 9)


def test_multiresolution_uses_actual_decoder_f_and_exact_requested_k():
    bands = build_multiresolution_bands(
        sample_rate=44100,
        n_fft=2048,
        num_bins=17,
        num_bands=5,
    )
    assert len(bands) == 5
    assert bands[0].start_bin == 0
    assert bands[-1].end_bin == 17
    validate_bands(bands, 17)


def test_projection_and_per_band_reconstruction_shapes():
    bands = build_multiresolution_bands(
        sample_rate=44100, n_fft=1024, num_bins=13, num_bands=4
    )
    projection = BandProjection(channels=3, bands=bands, embed_dim=12)
    reconstruction = BandReconstruction(
        channels=3, bands=bands, embed_dim=12, zero_init=False
    )
    x = torch.randn(2, 3, 13, 7, requires_grad=True)
    z = projection(x)
    restored = reconstruction(z)

    assert z.shape == (2, 7, 4, 12)
    assert restored.shape == x.shape
    restored.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_expert_is_batch_safe_shape_preserving_and_differentiable():
    expert = BandSequenceExpert(
        channels=4,
        freq_bins=17,
        embed_dim=16,
        num_bands=5,
        num_blocks=1,
        num_heads=4,
        sample_rate=44100,
        n_fft=2048,
    )
    x = torch.randn(3, 4, 17, 9, requires_grad=True)
    output = expert(x)

    assert output.shape == x.shape
    # Zero-initialized per-band output heads make the initial residual exact.
    torch.testing.assert_close(output, x)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in expert.parameters())


def test_expert_capacity_increases_via_width_and_depth_not_only_band_count():
    configs = [
        dict(embed_dim=16, num_blocks=1, num_heads=2),
        dict(embed_dim=24, num_blocks=2, num_heads=4),
        dict(embed_dim=32, num_blocks=3, num_heads=4),
        dict(embed_dim=48, num_blocks=4, num_heads=8),
    ]
    experts = [
        BandSequenceExpert(
            channels=4,
            freq_bins=24,
            num_bands=6,  # Deliberately fixed: K is not the capacity proxy.
            **config,
        )
        for config in configs
    ]
    counts = [_parameter_count(expert) for expert in experts]

    assert all(len(expert.band_specs) == 6 for expert in experts)
    assert counts == sorted(counts)
    assert all(left < right for left, right in zip(counts, counts[1:]))


def test_expert_rejects_more_bands_than_actual_frequency_rows():
    with pytest.raises(ValueError, match="K <= F"):
        BandSequenceExpert(
            channels=4,
            freq_bins=7,
            embed_dim=16,
            num_bands=8,
            num_blocks=1,
            num_heads=4,
        )
