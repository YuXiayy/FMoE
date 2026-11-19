"""Gradient, finite-value, and mixed-precision tests for audio objectives."""

import pytest
import torch

from src.utils.audio_utils import (
    complex_to_channels,
    pad_spectrogram_for_fmoe,
    stft_complex,
)
from src.utils.loss_utils import (
    AudioRestorationLoss,
    ComplexL1Loss,
    MultiResolutionSTFTLoss,
    WaveformL1Loss,
)


STFT_CONFIG = {"n_fft": 64, "hop_length": 16, "win_length": 64, "center": True}
MRSTFT_CONFIG = {
    "fft_sizes": (32, 64, 128),
    "hop_sizes": (8, 16, 32),
    "win_lengths": (32, 64, 128),
}


def _assert_backward_finite(loss: torch.Tensor, leaf: torch.Tensor) -> None:
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()


def test_complex_l1_backward_and_finite() -> None:
    pred = torch.randn(2, 4, 17, 13, requires_grad=True)
    target = torch.randn_like(pred)
    _assert_backward_finite(ComplexL1Loss()(pred, target), pred)


def test_waveform_l1_reconstructs_channels_and_backpropagates() -> None:
    target_wave = torch.randn(2, 2, 513)
    target_channels = complex_to_channels(stft_complex(target_wave, **STFT_CONFIG))
    pred_channels = (target_channels + 0.01 * torch.randn_like(target_channels)).requires_grad_()

    criterion = WaveformL1Loss(**STFT_CONFIG)
    loss = criterion(pred_channels, target_wave, length=torch.tensor([513, 513]))
    _assert_backward_finite(loss, pred_channels)


def test_waveform_l1_unpads_before_inverse() -> None:
    target_wave = torch.randn(2, 511)
    channels = complex_to_channels(stft_complex(target_wave, **STFT_CONFIG))
    padded, pad_info = pad_spectrogram_for_fmoe(channels, multiple=8)
    pred = padded.detach().requires_grad_()

    loss = WaveformL1Loss(**STFT_CONFIG)(
        pred,
        target_wave,
        length=target_wave.shape[-1],
        pad_info=pad_info,
    )
    _assert_backward_finite(loss, pred)
    assert loss.item() < 1e-6


def test_multi_resolution_stft_buffers_backward_and_finite() -> None:
    pred = torch.randn(2, 2, 512, requires_grad=True)
    target = torch.randn_like(pred)
    criterion = MultiResolutionSTFTLoss(**MRSTFT_CONFIG)

    assert {"window_0", "window_1", "window_2"}.issubset(dict(criterion.named_buffers()))
    _assert_backward_finite(criterion(pred, target), pred)


def test_symmetric_spectral_convergence_is_bounded_for_silent_targets() -> None:
    pred = torch.full((2, 1, 256), 1.0e-3, requires_grad=True)
    target = torch.zeros_like(pred)
    criterion = MultiResolutionSTFTLoss(
        fft_sizes=(64,),
        hop_sizes=(16,),
        win_lengths=(64,),
        spectral_convergence_weight=1.0,
        log_magnitude_weight=0.0,
        spectral_convergence_normalization="symmetric",
    )

    loss = criterion(pred, target)

    assert loss.item() <= 2.0 + 1.0e-6
    _assert_backward_finite(loss, pred)


def test_invalid_spectral_convergence_normalization_is_rejected() -> None:
    with pytest.raises(ValueError, match="spectral_convergence_normalization"):
        MultiResolutionSTFTLoss(
            fft_sizes=(64,),
            hop_sizes=(16,),
            win_lengths=(64,),
            spectral_convergence_normalization="unsafe",
        )


def test_audio_restoration_loss_returns_components_and_backpropagates() -> None:
    clean_wave = torch.randn(2, 2, 512)
    clean_spec = complex_to_channels(stft_complex(clean_wave, **STFT_CONFIG))
    restored_spec = (clean_spec + 0.03 * torch.randn_like(clean_spec)).requires_grad_()
    final_wave = (clean_wave + 0.02 * torch.randn_like(clean_wave)).requires_grad_()
    balance = torch.tensor(0.25, requires_grad=True)
    degradation_logits = torch.zeros(2, 19, requires_grad=True)
    criterion = AudioRestorationLoss(
        lambda_complex=1.0,
        lambda_wave=0.7,
        lambda_mrstft=0.5,
        lambda_balance=0.05,
        stft_config=STFT_CONFIG,
        mrstft_config=MRSTFT_CONFIG,
    )
    losses = criterion(
        {
            "restored_spec": restored_spec,
            "final_wave": final_wave,
            "degradation_logits": degradation_logits,
        },
        {
            "clean_spec": clean_spec,
            "clean_wave": clean_wave,
            "audio_length": 512,
            "degradation_multi_hot": torch.zeros(2, 19),
            "degradation_label_mask": torch.ones(2),
        },
        routing_losses={"balance": balance},
    )

    assert set(losses) == {
        "total",
        "complex",
        "wave",
        "mrstft",
        "degradation",
        "balance",
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    expected = (
        losses["complex"]
        + 0.7 * losses["wave"]
        + 0.5 * losses["mrstft"]
        + 0.1 * losses["degradation"]
        + 0.05 * losses["balance"]
    )
    torch.testing.assert_close(losses["total"], expected)
    losses["total"].backward()
    for leaf in (restored_spec, final_wave, degradation_logits, balance):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_audio_restoration_loss_can_reconstruct_wave_and_disable_every_other_term() -> None:
    clean_wave = torch.randn(2, 2, 509)
    clean_spec = complex_to_channels(stft_complex(clean_wave, **STFT_CONFIG))
    padded, pad_info = pad_spectrogram_for_fmoe(clean_spec, multiple=8)
    restored_spec = padded.detach().requires_grad_()
    criterion = AudioRestorationLoss(
        lambda_complex=0.0,
        lambda_wave=1.0,
        lambda_mrstft=0.0,
        lambda_degradation=0.0,
        lambda_balance=0.0,
        stft_config=STFT_CONFIG,
        mrstft_config=MRSTFT_CONFIG,
    )
    losses = criterion(
        {"restored_spec": restored_spec},
        {
            "clean_wave": clean_wave,
            "audio_length": torch.tensor([509, 509]),
            "pad_info": pad_info,
        },
    )

    _assert_backward_finite(losses["total"], restored_spec)
    assert losses["wave"].item() < 1e-6
    for name in ("complex", "mrstft", "degradation", "balance"):
        assert losses[name].item() == 0.0


def test_zero_weight_term_does_not_require_its_inputs() -> None:
    pred_spec = torch.randn(1, 2, 17, 9, requires_grad=True)
    target_spec = torch.randn_like(pred_spec)
    criterion = AudioRestorationLoss(
        lambda_complex=1.0,
        lambda_wave=0.0,
        lambda_mrstft=0.0,
        lambda_degradation=0.0,
        lambda_balance=0.0,
        stft_config=STFT_CONFIG,
        mrstft_config=MRSTFT_CONFIG,
    )
    losses = criterion({"restored_spec": pred_spec}, {"clean_spec": target_spec})
    _assert_backward_finite(losses["total"], pred_spec)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_losses_accept_cpu_low_precision_inputs(dtype: torch.dtype) -> None:
    pred_spec = torch.randn(1, 2, 17, 9).to(dtype).requires_grad_()
    target_spec = torch.randn_like(pred_spec)
    complex_loss = ComplexL1Loss()(pred_spec, target_spec)
    _assert_backward_finite(complex_loss, pred_spec)

    for criterion in (
        MultiResolutionSTFTLoss(
            fft_sizes=(32,), hop_sizes=(8,), win_lengths=(32,)
        ),
    ):
        pred_wave = torch.randn(1, 2, 128).to(dtype).requires_grad_()
        target_wave = torch.randn_like(pred_wave)
        _assert_backward_finite(criterion(pred_wave, target_wave), pred_wave)


def test_autocast_bfloat16_objective_is_finite() -> None:
    clean_wave = torch.randn(1, 2, 256)
    clean_spec = complex_to_channels(stft_complex(clean_wave, **STFT_CONFIG))
    restored_spec = clean_spec.detach().clone().requires_grad_()
    final_wave = clean_wave.detach().clone().requires_grad_()
    degradation_logits = torch.zeros(1, 19, requires_grad=True)
    criterion = AudioRestorationLoss(
        stft_config=STFT_CONFIG,
        mrstft_config={
            "fft_sizes": (32, 64),
            "hop_sizes": (8, 16),
            "win_lengths": (32, 64),
        },
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = criterion(
            {
                "restored_spec": restored_spec,
                "final_wave": final_wave,
                "degradation_logits": degradation_logits,
            },
            {
                "clean_spec": clean_spec,
                "clean_wave": clean_wave,
                "degradation_multi_hot": torch.zeros(1, 19),
                "degradation_label_mask": torch.ones(1),
            },
            routing_losses=torch.tensor(0.0),
        )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert torch.isfinite(restored_spec.grad).all()
    assert torch.isfinite(final_wave.grad).all()
    assert torch.isfinite(degradation_logits.grad).all()


def test_nonfinite_inputs_fail_fast() -> None:
    pred = torch.zeros(1, 2, 8, 8, requires_grad=True)
    pred.data[..., 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="pred_channels"):
        ComplexL1Loss()(pred, torch.zeros_like(pred))
