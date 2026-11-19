"""Audio representation helpers used by the magnitude and complex pipelines.

Complex spectra use one canonical, interleaved real-channel layout throughout
this project.  Channel ``c`` is represented by real channel ``2 * c`` and
imaginary channel ``2 * c + 1``.  A stereo spectrum is therefore ordered as
``[Re(L), Im(L), Re(R), Im(R)]``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


PadInfo = Tuple[int, int]
LengthLike = Union[int, torch.Tensor, Sequence[int]]


def _stft_work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a dtype supported reliably by CPU/CUDA FFT implementations."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _normalise_length(length: LengthLike) -> int:
    """Convert a scalar or an equal-valued batch of lengths to ``int``."""
    if isinstance(length, int):
        value = length
    elif isinstance(length, torch.Tensor):
        values = length.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if values.numel() == 0:
            raise ValueError("length must not be empty")
        if not torch.all(values == values[0]):
            raise ValueError("batched iSTFT currently requires equal audio lengths")
        value = int(values[0].item())
    else:
        values = tuple(int(item) for item in length)
        if not values:
            raise ValueError("length must not be empty")
        if any(item != values[0] for item in values):
            raise ValueError("batched iSTFT currently requires equal audio lengths")
        value = values[0]
    if value < 0:
        raise ValueError(f"length must be non-negative, got {value}")
    return value


def _uniform_int(value: object, name: str) -> int:
    """Read an integer that may have been collated into a tensor/list."""
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if flat.numel() == 0 or not torch.all(flat == flat[0]):
            raise ValueError(f"{name} must contain one uniform value per batch")
        return int(flat[0].item())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(item) for item in value)
        if not values or any(item != values[0] for item in values):
            raise ValueError(f"{name} must contain one uniform value per batch")
        return values[0]
    return int(value)  # type: ignore[arg-type]


def _parse_pad_info(pad_info: object) -> PadInfo:
    """Parse serializable padding metadata, including DataLoader-collated data."""
    if isinstance(pad_info, Mapping):
        pad_f = pad_info.get("pad_f", pad_info.get("frequency"))
        pad_t = pad_info.get("pad_t", pad_info.get("time"))
        if pad_f is None or pad_t is None:
            raise ValueError("pad_info mapping requires 'pad_f' and 'pad_t'")
    elif isinstance(pad_info, Sequence) and not isinstance(pad_info, (str, bytes)):
        if len(pad_info) != 2:
            raise ValueError("pad_info must contain exactly (pad_f, pad_t)")
        pad_f, pad_t = pad_info
    else:
        raise TypeError("pad_info must be a 2-tuple/list or mapping")
    result = (_uniform_int(pad_f, "pad_f"), _uniform_int(pad_t, "pad_t"))
    if result[0] < 0 or result[1] < 0:
        raise ValueError(f"padding values must be non-negative, got {result}")
    return result


def ensure_audio_channels(
    waveform: torch.Tensor,
    channels: int,
    multichannel_policy: str = "first",
) -> torch.Tensor:
    """Convert a waveform to a requested number of audio channels.

    Args:
        waveform: Floating tensor shaped ``[N]``, ``[C, N]``, or ``[B, C, N]``.
        channels: Requested positive output channel count.
        multichannel_policy: When reducing a multi-channel input to more than
            one channel, ``"first"`` selects the first ``channels`` channels;
            ``"error"`` rejects the conversion.  Conversion to mono always
            averages all input channels.

    Returns:
        Tensor shaped ``[channels, N]`` for a non-batched input or
        ``[B, channels, N]`` for a batched input.  A ``[N]`` input is treated
        as non-batched mono.  Mono-to-multichannel conversion replicates the
        channel (stereo becomes identical left/right channels).
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim not in (2, 3):
        raise ValueError(
            "waveform must have shape [N], [C, N], or [B, C, N], "
            f"got {tuple(waveform.shape)}"
        )

    channel_dim = 0 if waveform.ndim == 2 else 1
    current = waveform.shape[channel_dim]
    if current == channels:
        return waveform
    if channels == 1:
        return waveform.mean(dim=channel_dim, keepdim=True)
    if current == 1:
        repeats = [1] * waveform.ndim
        repeats[channel_dim] = channels
        return waveform.repeat(*repeats)
    if current > channels and multichannel_policy == "first":
        index = [slice(None)] * waveform.ndim
        index[channel_dim] = slice(0, channels)
        return waveform[tuple(index)]
    if multichannel_policy not in ("first", "error"):
        raise ValueError(
            "multichannel_policy must be either 'first' or 'error', "
            f"got {multichannel_policy!r}"
        )
    raise ValueError(f"cannot convert {current} channels to {channels} channels")


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a ``[C, N]`` waveform to mono ``[1, N]`` by averaging."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def fit_audio_length(
    waveform: torch.Tensor,
    target_len: int,
    train: bool = True,
) -> torch.Tensor:
    """Crop or right-pad ``[..., N]`` audio to ``[..., target_len]``."""
    cur_len = waveform.size(-1)
    if cur_len == target_len:
        return waveform
    if cur_len > target_len:
        if train:
            start = torch.randint(0, cur_len - target_len + 1, (1,)).item()
        else:
            start = (cur_len - target_len) // 2
        return waveform[..., start:start + target_len]
    pad_len = target_len - cur_len
    return F.pad(waveform, (0, pad_len), mode="constant", value=0.0)


def stft_complex(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    *,
    window: Optional[torch.Tensor] = None,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Compute a mono/stereo complex STFT without changing leading axes.

    Args:
        wav: Real tensor shaped ``[C, N]`` or ``[B, C, N]``.  A
            convenience ``[N]`` input is also accepted.
        n_fft: FFT size.
        hop_length: Frame hop in samples.
        win_length: Hann-window length, no larger than ``n_fft``.
        center: Whether to center frames using PyTorch STFT padding.
        window: Optional cached Hann window shaped ``[win_length]``.
        pad_mode: Padding mode used when ``center=True``.

    Returns:
        Complex tensor shaped ``[C, F, T]`` or ``[B, C, F, T]`` (and
        ``[F, T]`` for a ``[N]`` input), where ``F = n_fft // 2 + 1``.
        Half/bfloat16 inputs are promoted to complex64 for FFT stability.
    """
    if not wav.is_floating_point():
        raise TypeError(f"waveform must be floating point, got {wav.dtype}")
    if wav.ndim not in (1, 2, 3):
        raise ValueError(
            "waveform must have shape [N], [C, N], or [B, C, N], "
            f"got {tuple(wav.shape)}"
        )
    if not (0 < win_length <= n_fft):
        raise ValueError("win_length must satisfy 0 < win_length <= n_fft")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    work_dtype = _stft_work_dtype(wav.dtype)
    work_waveform = wav.to(dtype=work_dtype)
    if window is None:
        work_window = torch.hann_window(
            win_length,
            device=wav.device,
            dtype=work_dtype,
        )
    else:
        if window.ndim != 1 or window.numel() != win_length:
            raise ValueError(f"window must have shape [{win_length}]")
        work_window = window.to(device=wav.device, dtype=work_dtype)

    original_shape = work_waveform.shape[:-1]
    flat_waveform = work_waveform.reshape(-1, work_waveform.shape[-1])
    spec = torch.stft(
        flat_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=work_window,
        center=center,
        pad_mode=pad_mode,
        return_complex=True,
    )
    return spec.reshape(*original_shape, spec.shape[-2], spec.shape[-1])


def istft_complex(
    spec: torch.Tensor,
    length: LengthLike,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    *,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Invert a mono/stereo complex STFT and preserve leading axes.

    Args:
        spec: Complex tensor shaped ``[C, F, T]`` or ``[B, C, F, T]``;
            ``[F, T]`` is also accepted.
        length: Exact requested output sample count.  A batched tensor/list is
            accepted when all values are identical.
        n_fft: FFT size used to create ``spec``.
        hop_length: STFT frame hop.
        win_length: Hann-window length.
        center: Centering convention used to create ``spec``.
        window: Optional cached Hann window shaped ``[win_length]``.

    Returns:
        Real waveform shaped ``[C, length]``, ``[B, C, length]``, or
        ``[length]``, matching the leading axes of ``spec``.
    """
    if not torch.is_complex(spec):
        raise TypeError("spec must be a complex tensor")
    if spec.ndim not in (2, 3, 4):
        raise ValueError(
            "spec must have shape [F, T], [C, F, T], or [B, C, F, T], "
            f"got {tuple(spec.shape)}"
        )
    output_length = _normalise_length(length)
    work_complex_dtype = (
        torch.complex64
        if spec.real.dtype in (torch.float16, torch.bfloat16)
        else spec.dtype
    )
    work_spec = spec.to(dtype=work_complex_dtype)
    real_dtype = work_spec.real.dtype
    if window is None:
        work_window = torch.hann_window(
            win_length,
            device=spec.device,
            dtype=real_dtype,
        )
    else:
        if window.ndim != 1 or window.numel() != win_length:
            raise ValueError(f"window must have shape [{win_length}]")
        work_window = window.to(device=spec.device, dtype=real_dtype)

    original_shape = work_spec.shape[:-2]
    flat_spec = work_spec.reshape(-1, work_spec.shape[-2], work_spec.shape[-1])
    waveform = torch.istft(
        flat_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=work_window,
        center=center,
        length=output_length,
    )
    return waveform.reshape(*original_shape, output_length)


def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    """Convert complex audio channels to canonical interleaved real channels.

    Args:
        spec: Complex tensor shaped ``[C, F, T]`` or ``[B, C, F, T]``.

    Returns:
        Real tensor shaped ``[2*C, F, T]`` or ``[B, 2*C, F, T]``.
        Its order is ``[Re(ch0), Im(ch0), Re(ch1), Im(ch1), ...]``; stereo
        is exactly ``[Re(L), Im(L), Re(R), Im(R)]``.
    """
    if not torch.is_complex(spec):
        raise TypeError("spec must be a complex tensor")
    if spec.ndim not in (3, 4):
        raise ValueError(
            "spec must have shape [C, F, T] or [B, C, F, T], "
            f"got {tuple(spec.shape)}"
        )
    real_imag = torch.view_as_real(spec)
    if spec.ndim == 3:
        return real_imag.permute(0, 3, 1, 2).reshape(
            spec.shape[0] * 2, spec.shape[1], spec.shape[2]
        )
    return real_imag.permute(0, 1, 4, 2, 3).reshape(
        spec.shape[0], spec.shape[1] * 2, spec.shape[2], spec.shape[3]
    )


def channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    """Restore complex audio channels from canonical interleaved channels.

    Args:
        channels: Real tensor shaped ``[2*C, F, T]`` or
            ``[B, 2*C, F, T]``.  Consecutive pairs are real/imaginary parts.

    Returns:
        Complex tensor shaped ``[C, F, T]`` or ``[B, C, F, T]``.  For a
        stereo input, channels ``[0, 1]`` form left and ``[2, 3]`` form right.
    """
    if torch.is_complex(channels) or not channels.is_floating_point():
        raise TypeError("channels must be a real floating-point tensor")
    if channels.ndim not in (3, 4):
        raise ValueError(
            "channels must have shape [2*C, F, T] or [B, 2*C, F, T], "
            f"got {tuple(channels.shape)}"
        )
    channel_dim = channels.ndim - 3
    channel_count = channels.shape[channel_dim]
    if channel_count == 0 or channel_count % 2:
        raise ValueError(f"real channel count must be positive and even, got {channel_count}")
    work_dtype = _stft_work_dtype(channels.dtype)
    work_channels = channels.to(dtype=work_dtype)
    if channels.ndim == 3:
        paired = work_channels.reshape(
            channel_count // 2, 2, channels.shape[-2], channels.shape[-1]
        ).permute(0, 2, 3, 1)
    else:
        paired = work_channels.reshape(
            channels.shape[0],
            channel_count // 2,
            2,
            channels.shape[-2],
            channels.shape[-1],
        ).permute(0, 1, 3, 4, 2)
    return torch.view_as_complex(paired.contiguous())


def istft_from_channels(
    channels: torch.Tensor,
    length: LengthLike,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    *,
    window: Optional[torch.Tensor] = None,
    pad_info: Optional[object] = None,
) -> torch.Tensor:
    """Invert interleaved real STFT channels to a length-exact waveform.

    Args:
        channels: ``[2*C, F, T]`` or ``[B, 2*C, F, T]`` real tensor in
            ``[Re(ch0), Im(ch0), ...]`` order.
        length: Requested waveform sample count.
        n_fft: FFT size.
        hop_length: STFT frame hop.
        win_length: Hann-window length.
        center: STFT centering convention.
        window: Optional cached Hann window.
        pad_info: Optional ``(pad_f, pad_t)`` returned by
            :func:`pad_spectrogram_for_fmoe`.

    Returns:
        Waveform ``[C, length]`` or ``[B, C, length]``.
    """
    if pad_info is not None:
        channels = unpad_spectrogram(channels, pad_info)
    return istft_complex(
        channels_to_complex(channels),
        length=length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        window=window,
    )


class ComplexSTFT(nn.Module):
    """Reusable complex STFT/iSTFT with a registered Hann-window buffer.

    ``forward`` accepts ``[C, N]`` or ``[B, C, N]`` real waveforms and
    returns complex ``[C, F, T]`` or ``[B, C, F, T]`` spectra.  ``inverse``
    performs the reverse operation and enforces the requested sample length.
    The ``window`` buffer follows module device/dtype moves and is not rebuilt
    on every forward pass.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(n_fft if win_length is None else win_length)
        self.center = bool(center)
        self.pad_mode = pad_mode
        if not (0 < self.win_length <= self.n_fft):
            raise ValueError("win_length must satisfy 0 < win_length <= n_fft")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map ``[C, N]``/``[B, C, N]`` real audio to complex STFT."""
        return stft_complex(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=self.window,
            pad_mode=self.pad_mode,
        )

    def inverse(self, spec: torch.Tensor, length: LengthLike) -> torch.Tensor:
        """Map complex STFT back to ``[C, N]``/``[B, C, N]`` audio."""
        return istft_complex(
            spec,
            length=length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=self.window,
        )

    def to_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map waveform to canonical interleaved real STFT channels."""
        return complex_to_channels(self.forward(waveform))

    def inverse_channels(
        self,
        channels: torch.Tensor,
        length: LengthLike,
        pad_info: Optional[object] = None,
    ) -> torch.Tensor:
        """Invert canonical real STFT channels to exact-length audio."""
        return istft_from_channels(
            channels,
            length=length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=self.window,
            pad_info=pad_info,
        )


def stft_mag_phase(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    spec_log: bool = True,
):
    """Compute the legacy mono magnitude/phase representation.

    Args:
        waveform: Mono waveform ``[1, N]``.
        n_fft: FFT size.
        hop_length: Frame hop.
        win_length: Hann-window length.
        spec_log: Return ``log1p`` magnitude when true.

    Returns:
        Tuple ``(magnitude, phase, length)`` with shapes ``[1, F, T]``,
        ``[F, T]``, and integer ``N``.  This API intentionally preserves the
        original magnitude baseline contract.
    """
    assert waveform.dim() == 2 and waveform.size(0) == 1, (
        "Expected mono waveform shape [1, T]."
    )
    spec = stft_complex(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    mag = spec.abs()
    if spec_log:
        mag = torch.log1p(mag)
    phase = torch.angle(spec)
    return mag.unsqueeze(0), phase, waveform.size(-1)


def pad_spectrogram_for_fmoe(
    spec: torch.Tensor,
    multiple: int = 8,
) -> Tuple[torch.Tensor, PadInfo]:
    """Right-pad frequency/time axes to a model-compatible multiple.

    Args:
        spec: Tensor shaped ``[..., F, T]``; legacy ``[1, F, T]``, complex
            ``[C, F, T]``, and batched ``[B, C, F, T]`` are all supported.
        multiple: Positive divisor required by the encoder/decoder hierarchy.

    Returns:
        ``(padded_spec, (pad_f, pad_t))``.  The tuple is intentionally simple
        and serializable; :func:`unpad_spectrogram` also accepts its
        DataLoader-collated tensor/list form.
    """
    if spec.ndim < 2:
        raise ValueError(f"spec must end in [F, T], got {tuple(spec.shape)}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    f_bins, t_frames = spec.shape[-2:]
    pad_f = (multiple - (f_bins % multiple)) % multiple
    pad_t = (multiple - (t_frames % multiple)) % multiple
    if pad_f == 0 and pad_t == 0:
        return spec, (0, 0)
    padded = F.pad(spec, (0, pad_t, 0, pad_f), mode="constant", value=0.0)
    return padded, (pad_f, pad_t)


def unpad_spectrogram(
    spec: torch.Tensor,
    target_f_bins: Optional[object] = None,
    target_t_frames: Optional[int] = None,
    *,
    pad_info: Optional[object] = None,
) -> torch.Tensor:
    """Remove model padding while retaining the legacy target-size API.

    Args:
        spec: Tensor shaped ``[..., F_pad, T_pad]``.
        target_f_bins: New positional API: serializable ``(pad_f, pad_t)``
            metadata.  Legacy API: integer target frequency-bin count.
        target_t_frames: Legacy target frame count.  Omit when passing
            ``pad_info``.
        pad_info: Named form of the new padding metadata API.  It is mutually
            exclusive with ``target_f_bins`` and ``target_t_frames``.

    Returns:
        Cropped tensor ``[..., F, T]``.
    """
    if spec.ndim < 2:
        raise ValueError(f"spec must end in [F, T], got {tuple(spec.shape)}")
    if pad_info is not None:
        if target_f_bins is not None or target_t_frames is not None:
            raise TypeError("pad_info cannot be combined with legacy target sizes")
        target_f_bins = pad_info
    if target_f_bins is None:
        raise TypeError("provide pad_info or both target_f_bins and target_t_frames")
    if target_t_frames is None:
        pad_f, pad_t = _parse_pad_info(target_f_bins)
        if pad_f > spec.shape[-2] or pad_t > spec.shape[-1]:
            raise ValueError(
                f"pad_info {(pad_f, pad_t)} exceeds spectrogram shape {tuple(spec.shape[-2:])}"
            )
        target_f_bins = spec.shape[-2] - pad_f
        target_t = spec.shape[-1] - pad_t
    else:
        target_f_bins = int(target_f_bins)  # legacy signature
        target_t = int(target_t_frames)
        if not (0 <= target_f_bins <= spec.shape[-2]):
            raise ValueError("target_f_bins is outside the padded spectrogram")
        if not (0 <= target_t <= spec.shape[-1]):
            raise ValueError("target_t_frames is outside the padded spectrogram")
    return spec[..., :target_f_bins, :target_t]


def istft_from_mag_phase(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    spec_log: bool = True,
) -> torch.Tensor:
    """Reconstruct legacy mono ``[1, N]`` audio from magnitude and phase.

    ``magnitude`` must be ``[1, F, T]`` and ``phase`` must be ``[F, T]``.
    The returned waveform has exactly ``length`` samples.
    """
    mag = magnitude.squeeze(0)
    if spec_log:
        mag = torch.expm1(torch.clamp(mag, min=0.0))
    spec = torch.polar(mag, phase)
    wav = istft_complex(
        spec,
        length=length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return wav.unsqueeze(0)
