from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_utils import ComplexSTFT, LengthLike, istft_from_channels


def _assert_finite(tensor: torch.Tensor, name: str, enabled: bool = True) -> None:
    """Raise a focused error instead of letting NaN/Inf silently reach training."""
    if enabled and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _float_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Promote low-precision inputs for stable CPU and mixed-precision losses."""
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    common_dtype = torch.promote_types(pred.dtype, target.dtype)
    return pred.to(dtype=common_dtype), target.to(dtype=common_dtype)


def _validate_wave_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Validate waveform tensors with shape ``[N]``, ``[C,N]``, or ``[B,C,N]``."""
    if pred.shape != target.shape:
        raise ValueError(
            f"predicted and target waveform shapes differ: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if pred.ndim not in (1, 2, 3):
        raise ValueError(
            "waveforms must have shape [N], [C, N], or [B, C, N], "
            f"got {tuple(pred.shape)}"
        )
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("waveforms must be floating-point tensors")


class ComplexL1Loss(nn.Module):
    """L1 loss on canonical interleaved real/imaginary STFT channels.

    Inputs are real tensors shaped ``[2*C, F, T]`` or ``[B, 2*C, F, T]``.
    Channel order is ``[Re(ch0), Im(ch0), Re(ch1), Im(ch1), ...]``.  The
    returned tensor is scalar for ``reduction='mean'`` or ``'sum'``.
    """

    def __init__(self, reduction: str = "mean", check_finite: bool = True) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"unsupported reduction: {reduction!r}")
        self.reduction = reduction
        self.check_finite = bool(check_finite)

    def forward(self, pred_channels: torch.Tensor, target_channels: torch.Tensor) -> torch.Tensor:
        """Compare matching ``[2*C,F,T]``/``[B,2*C,F,T]`` real tensors."""
        if pred_channels.shape != target_channels.shape:
            raise ValueError(
                "predicted and target complex-channel shapes differ: "
                f"{tuple(pred_channels.shape)} vs {tuple(target_channels.shape)}"
            )
        if pred_channels.ndim not in (3, 4):
            raise ValueError("complex channels must have shape [2*C,F,T] or [B,2*C,F,T]")
        if pred_channels.shape[-3] % 2:
            raise ValueError("complex-channel dimension must be even")
        if (
            torch.is_complex(pred_channels)
            or torch.is_complex(target_channels)
            or not pred_channels.is_floating_point()
            or not target_channels.is_floating_point()
        ):
            raise TypeError("complex-channel inputs must be real floating-point tensors")
        _assert_finite(pred_channels, "pred_channels", self.check_finite)
        _assert_finite(target_channels, "target_channels", self.check_finite)
        pred, target = _float_pair(pred_channels, target_channels)
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        _assert_finite(loss, "complex_l1", self.check_finite)
        return loss


class WaveformL1Loss(nn.Module):
    """Reconstruct STFT channels and compare them with a target waveform.

    ``pred_spec_channels`` is ``[2*C,F,T]`` or ``[B,2*C,F,T]`` in canonical
    interleaved order.  ``target_wave`` is ``[C,N]`` or ``[B,C,N]``.  The
    output is a scalar L1 loss by default.  A cached Hann-window buffer is used
    when the optional per-call ``stft_config`` matches the constructor config.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        center: bool = True,
        reduction: str = "mean",
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"unsupported reduction: {reduction!r}")
        self.transform = ComplexSTFT(n_fft, hop_length, win_length, center=center)
        self.reduction = reduction
        self.check_finite = bool(check_finite)

    def forward(
        self,
        pred_spec_channels: torch.Tensor,
        target_wave: torch.Tensor,
        length: Optional[LengthLike] = None,
        stft_config: Optional[Mapping[str, object]] = None,
        pad_info: Optional[object] = None,
    ) -> torch.Tensor:
        """Return L1 for spectrum ``[...,2*C,F,T]`` versus audio ``[...,C,N]``."""
        _assert_finite(pred_spec_channels, "pred_spec_channels", self.check_finite)
        _assert_finite(target_wave, "target_wave", self.check_finite)
        if length is None:
            length = target_wave.shape[-1]

        if stft_config is None:
            pred_wave = self.transform.inverse_channels(
                pred_spec_channels,
                length=length,
                pad_info=pad_info,
            )
        else:
            n_fft = int(stft_config.get("n_fft", self.transform.n_fft))
            hop_length = int(stft_config.get("hop_length", self.transform.hop_length))
            win_length = int(stft_config.get("win_length", self.transform.win_length))
            center = bool(stft_config.get("center", self.transform.center))
            if (
                n_fft == self.transform.n_fft
                and hop_length == self.transform.hop_length
                and win_length == self.transform.win_length
                and center == self.transform.center
            ):
                pred_wave = self.transform.inverse_channels(
                    pred_spec_channels,
                    length=length,
                    pad_info=pad_info,
                )
            else:
                pred_wave = istft_from_channels(
                    pred_spec_channels,
                    length=length,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    pad_info=pad_info,
                )

        if target_wave.shape[-1] < pred_wave.shape[-1]:
            raise ValueError("target waveform is shorter than the requested iSTFT length")
        target_wave = target_wave[..., :pred_wave.shape[-1]]
        _validate_wave_pair(pred_wave, target_wave)
        pred, target = _float_pair(pred_wave, target_wave)
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        _assert_finite(loss, "waveform_l1", self.check_finite)
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution spectral convergence and log-magnitude loss.

    Inputs are real waveforms ``[N]``, ``[C,N]``, or ``[B,C,N]``.  Audio and
    batch/channel axes are flattened only for the STFT operation.  Each Hann
    window is registered as a module buffer for device/DDP correctness.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048, 4096),
        hop_sizes: Sequence[int] = (128, 256, 512, 1024),
        win_lengths: Sequence[int] = (512, 1024, 2048, 4096),
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        spectral_convergence_normalization: str = "target",
        center: bool = True,
        eps: float = 1e-7,
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)) or not fft_sizes:
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have equal non-zero length")
        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.hop_sizes = tuple(int(value) for value in hop_sizes)
        self.win_lengths = tuple(int(value) for value in win_lengths)
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            if not (0 < win <= n_fft) or hop <= 0:
                raise ValueError("each resolution requires 0 < win_length <= n_fft and hop > 0")
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        if spectral_convergence_normalization not in {"target", "symmetric"}:
            raise ValueError(
                "spectral_convergence_normalization must be 'target' or 'symmetric'"
            )
        self.spectral_convergence_normalization = (
            spectral_convergence_normalization
        )
        self.center = bool(center)
        self.eps = float(eps)
        self.check_finite = bool(check_finite)
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        for index, win_length in enumerate(self.win_lengths):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length))

    def _window(self, index: int, reference: torch.Tensor) -> torch.Tensor:
        window = getattr(self, f"window_{index}")
        return window.to(device=reference.device, dtype=reference.dtype)

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
        """Compare matching waveforms and return one scalar averaged over scales."""
        _validate_wave_pair(pred_wave, target_wave)
        _assert_finite(pred_wave, "pred_wave", self.check_finite)
        _assert_finite(target_wave, "target_wave", self.check_finite)
        pred, target = _float_pair(pred_wave, target_wave)
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        scale_losses = []
        for index, (n_fft, hop, win_length) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            window = self._window(index, pred)
            pred_spec = torch.stft(
                pred,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                window=window,
                center=self.center,
                pad_mode="constant",
                return_complex=True,
            )
            target_spec = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                window=window,
                center=self.center,
                pad_mode="constant",
                return_complex=True,
            )
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()
            numerator = torch.linalg.vector_norm(
                (target_mag - pred_mag).reshape(target_mag.shape[0], -1), dim=-1
            )
            target_norm = torch.linalg.vector_norm(
                target_mag.reshape(target_mag.shape[0], -1), dim=-1
            )
            if self.spectral_convergence_normalization == "symmetric":
                # 2*||P-T||/(||P||+||T||) is scale-invariant and bounded by
                # two.  Unlike the legacy target-only denominator it cannot
                # explode when a valid crop is silent or nearly silent.
                pred_norm = torch.linalg.vector_norm(
                    pred_mag.reshape(pred_mag.shape[0], -1), dim=-1
                )
                denominator = (pred_norm + target_norm).clamp_min(self.eps)
                spectral_convergence = (2.0 * numerator / denominator).mean()
            else:
                denominator = target_norm.clamp_min(self.eps)
                spectral_convergence = (numerator / denominator).mean()
            log_magnitude = F.l1_loss(
                torch.log(pred_mag + self.eps),
                torch.log(target_mag + self.eps),
            )
            scale_loss = (
                self.spectral_convergence_weight * spectral_convergence
                + self.log_magnitude_weight * log_magnitude
            )
            scale_losses.append(scale_loss)

        loss = torch.stack(scale_losses).mean()
        _assert_finite(loss, "multi_resolution_stft", self.check_finite)
        return loss


class AudioRestorationLoss(nn.Module):
    """Public F-MoE training objective."""

    COMPONENTS = (
        "complex",
        "wave",
        "mrstft",
        "degradation",
        "balance",
    )

    def __init__(
        self,
        lambda_complex: float = 1.0,
        lambda_wave: float = 1.0,
        lambda_mrstft: float = 1.0,
        lambda_degradation: float = 0.1,
        lambda_balance: float = 0.1,
        stft_config: Optional[Mapping[str, object]] = None,
        mrstft_config: Optional[Mapping[str, object]] = None,
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        self.weights = {
            "complex": float(lambda_complex),
            "wave": float(lambda_wave),
            "mrstft": float(lambda_mrstft),
            "degradation": float(lambda_degradation),
            "balance": float(lambda_balance),
        }
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("audio restoration loss weights must be non-negative")
        self.check_finite = bool(check_finite)

        stft = dict(stft_config or {})
        n_fft = int(stft.get("n_fft", 2048))
        win_length_value = stft.get("win_length")
        win_length = n_fft if win_length_value is None else int(win_length_value)
        self.complex_loss = ComplexL1Loss(check_finite=check_finite)
        self.waveform_loss = WaveformL1Loss(
            n_fft=n_fft,
            hop_length=int(stft.get("hop_length", 512)),
            win_length=win_length,
            center=bool(stft.get("center", True)),
            check_finite=check_finite,
        )

        mrstft = dict(mrstft_config or {})
        mrstft.setdefault("check_finite", check_finite)
        self.mrstft_loss = MultiResolutionSTFTLoss(**mrstft)

    @staticmethod
    def _tensor(mapping: Mapping[str, object], *names: str) -> Optional[torch.Tensor]:
        for name in names:
            value = mapping.get(name)
            if isinstance(value, torch.Tensor):
                return value
        return None

    @staticmethod
    def _base_tensor(outputs: Mapping[str, object], batch: Mapping[str, object]) -> torch.Tensor:
        for mapping in (outputs, batch):
            for value in mapping.values():
                if isinstance(value, torch.Tensor):
                    return value
        return torch.zeros((), dtype=torch.float32)

    def _balance_value(
        self,
        outputs: Mapping[str, object],
        routing_losses: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        source: object = routing_losses
        if source is None:
            source = outputs.get("balance_loss", outputs.get("routing_losses"))
        if isinstance(source, torch.Tensor):
            return source.mean()
        if isinstance(source, Mapping):
            for name in ("balance", "balance_loss", "total"):
                value = source.get(name)
                if isinstance(value, torch.Tensor):
                    return value.mean()
            tensors = [
                value.mean()
                for value in source.values()
                if isinstance(value, torch.Tensor)
            ]
            if tensors:
                return torch.stack(tensors).sum()
        return zero

    def forward(
        self,
        outputs: Mapping[str, object],
        batch: Mapping[str, object],
        routing_losses: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the five-term objective and return every scalar component."""
        base = self._base_tensor(outputs, batch)
        zero = base.sum() * 0.0
        losses: Dict[str, torch.Tensor] = {
            name: zero for name in self.COMPONENTS
        }
        pred_spec = self._tensor(outputs, "restored_spec", "pred_spec")
        target_spec = self._tensor(batch, "clean_spec", "target_spec")

        if self.weights["complex"] != 0.0:
            if pred_spec is None or target_spec is None:
                raise KeyError(
                    "active complex loss requires restored_spec and clean_spec"
                )
            losses["complex"] = self.complex_loss(pred_spec, target_spec)

        waveform_terms_active = any(
            self.weights[name] != 0.0
            for name in ("wave", "mrstft")
        )
        pred_wave = self._tensor(
            outputs, "final_wave", "coarse_wave", "restored_wave"
        )
        target_wave = self._tensor(batch, "clean_wave", "target_wave")
        if waveform_terms_active:
            if target_wave is None:
                raise KeyError("active waveform-domain losses require clean_wave")
            if pred_wave is None:
                if pred_spec is None:
                    raise KeyError(
                        "waveform losses require final_wave/coarse_wave or restored_spec"
                    )
                length = batch.get("audio_length", target_wave.shape[-1])
                pred_wave = self.waveform_loss.transform.inverse_channels(
                    pred_spec,
                    length=length,  # type: ignore[arg-type]
                    pad_info=batch.get("pad_info"),
                )
            if target_wave.shape[-1] < pred_wave.shape[-1]:
                raise ValueError("clean_wave is shorter than predicted waveform")
            target_wave = target_wave[..., :pred_wave.shape[-1]]
            _validate_wave_pair(pred_wave, target_wave)
            _assert_finite(pred_wave, "pred_wave", self.check_finite)
            _assert_finite(target_wave, "target_wave", self.check_finite)

        if self.weights["wave"] != 0.0:
            assert pred_wave is not None and target_wave is not None
            pred_float, target_float = _float_pair(pred_wave, target_wave)
            losses["wave"] = F.l1_loss(pred_float, target_float)
            _assert_finite(losses["wave"], "wave", self.check_finite)

        if self.weights["mrstft"] != 0.0:
            assert pred_wave is not None and target_wave is not None
            losses["mrstft"] = self.mrstft_loss(pred_wave, target_wave)

        if self.weights["degradation"] != 0.0:
            logits = self._tensor(outputs, "degradation_logits")
            targets = self._tensor(batch, "degradation_multi_hot")
            label_mask = self._tensor(batch, "degradation_label_mask")
            if logits is None or targets is None or label_mask is None:
                raise KeyError(
                    "degradation loss requires logits, multi-hot targets, and "
                    "one label mask value per sample"
                )
            if logits.ndim != 2 or targets.shape != logits.shape:
                raise ValueError(
                    "degradation logits and targets must match [B, classes]"
                )
            targets = targets.to(device=logits.device, dtype=torch.float32)
            if not bool(torch.isfinite(targets).all()):
                raise FloatingPointError("degradation targets contain NaN or Inf")
            if bool(((targets < 0.0) | (targets > 1.0)).any()):
                raise ValueError("degradation targets must be in [0, 1]")
            per_class = F.binary_cross_entropy_with_logits(
                logits.float(), targets, reduction="none"
            )
            class_mask = self._tensor(batch, "degradation_class_mask")
            if class_mask is None:
                class_mask = torch.ones_like(per_class)
            else:
                class_mask = class_mask.to(
                    device=per_class.device, dtype=per_class.dtype
                )
                if class_mask.shape != per_class.shape:
                    raise ValueError(
                        "degradation_class_mask must match logits [B, classes]"
                    )
                if not bool(torch.isfinite(class_mask).all()):
                    raise FloatingPointError(
                        "degradation_class_mask contains NaN or Inf"
                    )
                if bool(((class_mask < 0.0) | (class_mask > 1.0)).any()):
                    raise ValueError("degradation_class_mask must be in [0, 1]")
            valid_classes = class_mask.sum(dim=-1)
            per_sample = torch.where(
                valid_classes > 0,
                (per_class * class_mask).sum(dim=-1)
                / valid_classes.clamp_min(1.0),
                logits.float().sum(dim=-1) * 0.0,
            )
            label_mask = label_mask.to(
                device=per_sample.device, dtype=per_sample.dtype
            ).reshape(-1)
            if label_mask.numel() != per_sample.numel():
                raise ValueError(
                    "degradation_label_mask must contain one value per sample"
                )
            if not bool(torch.isfinite(label_mask).all()):
                raise FloatingPointError(
                    "degradation_label_mask contains NaN or Inf"
                )
            if bool(((label_mask < 0.0) | (label_mask > 1.0)).any()):
                raise ValueError("degradation_label_mask must be in [0, 1]")
            valid_count = label_mask.sum()
            losses["degradation"] = torch.where(
                valid_count > 0,
                (per_sample * label_mask).sum() / valid_count.clamp_min(1.0),
                logits.sum() * 0.0,
            )
            _assert_finite(
                losses["degradation"], "degradation", self.check_finite
            )

        if self.weights["balance"] != 0.0:
            losses["balance"] = self._balance_value(
                outputs, routing_losses, zero
            )
            _assert_finite(losses["balance"], "balance", self.check_finite)

        total = zero
        for name in self.COMPONENTS:
            total = total + self.weights[name] * losses[name]
        _assert_finite(total, "total_audio_restoration_loss", self.check_finite)
        return {"total": total, **losses}

