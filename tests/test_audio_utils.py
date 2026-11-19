"""Unit tests for legacy and canonical complex audio representations."""

import pytest
import torch

from src.utils.audio_utils import (
    ComplexSTFT,
    channels_to_complex,
    complex_to_channels,
    ensure_audio_channels,
    istft_complex,
    istft_from_channels,
    istft_from_mag_phase,
    pad_spectrogram_for_fmoe,
    stft_complex,
    stft_mag_phase,
    unpad_spectrogram,
)


STFT_CONFIG = {"n_fft": 64, "hop_length": 16, "win_length": 64}


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("audio_channels", [1, 2])
def test_complex_channel_conversion_is_interleaved_and_invertible(
    batched: bool,
    audio_channels: int,
) -> None:
    """Conversion is exact and stereo order is [ReL, ImL, ReR, ImR]."""
    shape = (2, audio_channels, 9, 7) if batched else (audio_channels, 9, 7)
    real = torch.randn(shape)
    imag = torch.randn(shape)
    spec = torch.complex(real, imag)

    encoded = complex_to_channels(spec)
    channel_dim = 1 if batched else 0
    for channel in range(audio_channels):
        torch.testing.assert_close(
            encoded.select(channel_dim, 2 * channel),
            real.select(channel_dim, channel),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            encoded.select(channel_dim, 2 * channel + 1),
            imag.select(channel_dim, channel),
            rtol=0.0,
            atol=0.0,
        )

    reconstructed = channels_to_complex(encoded)
    assert (reconstructed - spec).abs().max().item() < 1e-6


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1024),
        (2, 1024),
        (3, 1, 1024),
        (3, 2, 1024),
    ],
)
def test_complex_stft_istft_roundtrip_mono_stereo_and_batch(shape: tuple[int, ...]) -> None:
    waveform = torch.randn(shape)
    spec = stft_complex(waveform, **STFT_CONFIG)
    restored = istft_complex(spec, length=waveform.shape[-1], **STFT_CONFIG)

    assert restored.shape == waveform.shape
    assert (restored - waveform).abs().max().item() < 1e-6


def test_channel_function_and_cached_module_roundtrip_with_exact_length() -> None:
    waveform = torch.randn(2, 2, 997)
    transform = ComplexSTFT(**STFT_CONFIG)

    assert "window" in dict(transform.named_buffers())
    channels = transform.to_channels(waveform)
    assert channels.shape[:2] == (2, 4)
    restored = transform.inverse_channels(channels, length=torch.tensor([997, 997]))
    direct = istft_from_channels(channels, length=997, **STFT_CONFIG)

    assert restored.shape == waveform.shape
    torch.testing.assert_close(restored, direct, rtol=0.0, atol=0.0)
    assert (restored - waveform).abs().max().item() < 1e-6


def test_padding_metadata_new_and_legacy_unpad_apis() -> None:
    spec = torch.randn(2, 4, 33, 19)
    padded, pad_info = pad_spectrogram_for_fmoe(spec, multiple=8)

    assert padded.shape == (2, 4, 40, 24)
    assert pad_info == (7, 5)
    torch.testing.assert_close(unpad_spectrogram(padded, pad_info), spec)
    torch.testing.assert_close(unpad_spectrogram(padded, 33, 19), spec)
    torch.testing.assert_close(
        unpad_spectrogram(padded, target_f_bins=33, target_t_frames=19), spec
    )
    torch.testing.assert_close(unpad_spectrogram(padded, pad_info=pad_info), spec)

    collated_pad_info = [torch.tensor([7, 7]), torch.tensor([5, 5])]
    torch.testing.assert_close(unpad_spectrogram(padded, collated_pad_info), spec)


def test_padded_complex_channels_reconstruct_after_unpad() -> None:
    waveform = torch.randn(2, 1001)
    channels = complex_to_channels(stft_complex(waveform, **STFT_CONFIG))
    padded, pad_info = pad_spectrogram_for_fmoe(channels, multiple=8)
    restored = istft_from_channels(
        padded,
        length=waveform.shape[-1],
        pad_info=pad_info,
        **STFT_CONFIG,
    )

    assert restored.shape == waveform.shape
    assert (restored - waveform).abs().max().item() < 1e-6


def test_legacy_magnitude_api_is_preserved() -> None:
    waveform = torch.randn(1, 1024)
    magnitude, phase, length = stft_mag_phase(waveform, spec_log=True, **STFT_CONFIG)
    restored = istft_from_mag_phase(
        magnitude,
        phase,
        length=length,
        spec_log=True,
        **STFT_CONFIG,
    )

    assert magnitude.ndim == 3 and magnitude.shape[0] == 1
    assert phase.shape == magnitude.shape[1:]
    assert restored.shape == waveform.shape
    assert (restored - waveform).abs().max().item() < 1e-6


def test_ensure_audio_channels_has_explicit_mono_stereo_policy() -> None:
    mono = torch.randn(1, 128)
    stereo = ensure_audio_channels(mono, 2)
    torch.testing.assert_close(stereo[0], mono[0])
    torch.testing.assert_close(stereo[1], mono[0])

    surround = torch.stack([torch.full((128,), float(index)) for index in range(4)])
    torch.testing.assert_close(ensure_audio_channels(surround, 1), surround.mean(0, keepdim=True))
    torch.testing.assert_close(ensure_audio_channels(surround, 2), surround[:2])
    with pytest.raises(ValueError):
        ensure_audio_channels(surround, 2, multichannel_policy="error")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_low_precision_stft_is_promoted_and_finite(dtype: torch.dtype) -> None:
    waveform = torch.randn(1, 512).to(dtype=dtype)
    spec = stft_complex(wav=waveform, **STFT_CONFIG)
    restored = istft_complex(spec, length=waveform.shape[-1], **STFT_CONFIG)

    assert spec.dtype == torch.complex64
    assert restored.dtype == torch.float32
    assert torch.isfinite(restored).all()


def test_invalid_complex_channels_are_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        channels_to_complex(torch.randn(3, 9, 7))
    with pytest.raises(TypeError, match="complex"):
        complex_to_channels(torch.randn(2, 9, 7))
