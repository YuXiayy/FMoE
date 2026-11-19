"""Audio-derived conditioning for compound degradation-aware routing.

The encoder consumes the project's canonical interleaved complex STFT layout:
``[B, 2*C, F, T]`` with channel order ``[Re(ch0), Im(ch0), ...]``.  A single
instance accepts both mono (``C=1``) and stereo (``C=2``) batches.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from utils.audio_utils import channels_to_complex


def _finite(tensor: torch.Tensor) -> torch.Tensor:
    """Replace invalid values without detaching finite inputs from autograd."""

    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose a GroupNorm divisor while retaining multiple channels per group."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0 and (channels // groups >= 2 or groups == 1):
            return groups
    return 1


def _attention_pool(
    features: torch.Tensor, score_layer: nn.Conv1d
) -> torch.Tensor:
    """Attention-pool ``[B, D, L]`` features into ``[B, D]`` safely."""

    scores = _finite(score_layer(features)).squeeze(1)
    # Softmax in float32 avoids underflow for fp16/bf16 mixed precision.
    weights = torch.softmax(scores.float(), dim=-1).to(dtype=features.dtype)
    return _finite((features * weights.unsqueeze(1)).sum(dim=-1))


class FrequencyCueEncoder(nn.Module):
    """Extract multi-scale spectral-shape cues.

    Input:
        ``complex_channels``: standardized mono/stereo channels ``[B, 4, F, T]``.
    Output:
        Frequency embedding ``[B, output_dim]``.
    """

    def __init__(
        self,
        output_dim: int,
        in_channels: int = 4,
        kernels: Sequence[int] = (3, 9, 21),
    ) -> None:
        super().__init__()
        if output_dim <= 0 or in_channels <= 0:
            raise ValueError("output_dim and in_channels must be positive")
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("frequency kernels must be non-empty positive odd integers")

        # Three D/4 paths follow the task design; a projection resolves rounding
        # and makes output_dim unrestricted.
        path_dim = max(4, math.ceil(output_dim / 4))
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    path_dim,
                    kernel_size=(kernel, 1),
                    padding=(kernel // 2, 0),
                )
                for kernel in kernels
            ]
        )
        merged_dim = path_dim * len(kernels)
        self.merge_norm = nn.GroupNorm(_group_count(merged_dim), merged_dim)
        self.projection = nn.Sequential(
            nn.Conv2d(merged_dim, output_dim, kernel_size=1),
            nn.GroupNorm(_group_count(output_dim), output_dim),
            nn.GELU(),
        )
        self.frequency_attention = nn.Conv1d(output_dim, 1, kernel_size=1)

    def forward(self, complex_channels: torch.Tensor) -> torch.Tensor:
        if complex_channels.ndim != 4:
            raise ValueError("complex_channels must have shape [B, 4, F, T]")
        paths = [convolution(complex_channels) for convolution in self.convolutions]
        features = torch.cat(paths, dim=1)
        features = torch.nn.functional.gelu(self.merge_norm(features))
        features = self.projection(features)
        # Degradation frequency response should not depend on clip duration.
        frequency_sequence = features.mean(dim=-1)
        return _attention_pool(frequency_sequence, self.frequency_attention)


class _DilatedTemporalBlock(nn.Module):
    """Residual dilated Conv1d block preserving ``[B, D, T]`` shape."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.output_norm = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.output_norm(sequence + self.block(sequence)))


class TemporalCueEncoder(nn.Module):
    """Extract envelope and long-tail cues with dilated temporal convolutions.

    Input:
        ``complex_channels``: standardized channels ``[B, 4, F, T]``.
    Output:
        Temporal embedding ``[B, output_dim]``.
    """

    def __init__(
        self,
        output_dim: int,
        in_channels: int = 4,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if output_dim <= 0 or in_channels <= 0:
            raise ValueError("output_dim and in_channels must be positive")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must be non-empty positive integers")
        self.input_projection = nn.Conv1d(in_channels, output_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_DilatedTemporalBlock(output_dim, int(dilation)) for dilation in dilations]
        )
        self.temporal_attention = nn.Conv1d(output_dim, 1, kernel_size=1)

    def forward(self, complex_channels: torch.Tensor) -> torch.Tensor:
        if complex_channels.ndim != 4:
            raise ValueError("complex_channels must have shape [B, 4, F, T]")
        # [B, 4, F, T] -> [B, 4, T]
        sequence = self.input_projection(complex_channels.mean(dim=2))
        for block in self.blocks:
            sequence = block(sequence)
        return _attention_pool(sequence, self.temporal_attention)


class StereoCueEncoder(nn.Module):
    """Encode Mid/Side energy and inter-channel phase statistics.

    Input:
        ``complex_spec``: complex STFT ``[B, C, F, T]``, where ``C`` is 1 or 2.
    Output:
        Stereo embedding ``[B, output_dim]``.  Mono inputs use zero statistics,
        keeping the forward graph and fusion shape identical across channel modes.
    """

    num_statistics = 6

    def __init__(self, output_dim: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        hidden_dim = max(16, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_statistics, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def statistics(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Return six stable M/S statistics shaped ``[B, 6]``.

        Columns are mean ``|M|``, mean ``|S|``, Side/Mid energy ratio,
        mean L-R magnitude difference, and energy-weighted cosine/sine of the
        inter-channel phase difference.  Mono inputs return zeros.
        """

        if complex_spec.ndim != 4 or not torch.is_complex(complex_spec):
            raise ValueError("complex_spec must be complex with shape [B, C, F, T]")
        if complex_spec.shape[1] not in (1, 2):
            raise ValueError("StereoCueEncoder supports only mono or stereo audio")

        batch_size = int(complex_spec.shape[0])
        if complex_spec.shape[1] == 1:
            return torch.zeros(
                batch_size,
                self.num_statistics,
                device=complex_spec.device,
                dtype=complex_spec.real.dtype,
            )

        # Work in complex64: ComplexHalf has incomplete CPU/GPU operator support.
        spectrum = complex_spec.to(dtype=torch.complex64)
        left = spectrum[:, 0]
        right = spectrum[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        left_magnitude = left.abs()
        right_magnitude = right.abs()
        mid_magnitude = mid.abs()
        side_magnitude = side.abs()
        reduce_dims = (-2, -1)

        mean_mid = mid_magnitude.mean(dim=reduce_dims)
        mean_side = side_magnitude.mean(dim=reduce_dims)
        mid_energy = mid_magnitude.square().mean(dim=reduce_dims)
        side_energy = side_magnitude.square().mean(dim=reduce_dims)
        energy_ratio = (side_energy / mid_energy.clamp_min(self.eps)).clamp(max=1.0e6)
        magnitude_difference = (left_magnitude - right_magnitude).abs().mean(
            dim=reduce_dims
        )

        cross_spectrum = left * right.conj()
        phase_weight = left_magnitude * right_magnitude
        weight_sum = phase_weight.sum(dim=reduce_dims).clamp_min(self.eps)
        phase_cosine = cross_spectrum.real.sum(dim=reduce_dims) / weight_sum
        phase_sine = cross_spectrum.imag.sum(dim=reduce_dims) / weight_sum

        values = torch.stack(
            (
                mean_mid,
                mean_side,
                energy_ratio,
                magnitude_difference,
                phase_cosine,
                phase_sine,
            ),
            dim=-1,
        )
        return _finite(values)

    def forward(self, complex_spec: torch.Tensor) -> torch.Tensor:
        values = self.statistics(complex_spec)
        parameter = next(self.mlp.parameters())
        values = values.to(device=parameter.device, dtype=parameter.dtype)
        embedding = self.mlp(values)
        # The branch is definitionally stereo-only.  Multiplication (instead of
        # bypassing the MLP) keeps every parameter in the autograd graph for
        # mono-only DDP batches while making its actual mono contribution zero.
        stereo_mask = float(complex_spec.shape[1] == 2)
        return _finite(embedding * stereo_mask)


class CompoundDegradationEncoder(nn.Module):
    """Fuse frequency, temporal, and stereo degradation evidence.

    Args:
        degradation_embed_dim: Output embedding width ``D``.
        num_degradation_classes: Width of the training-only multi-label head.
        branch_dim: Width of each cue branch.  Defaults to ``max(D/2, 16)``.
        include_training_head: Include the auxiliary degradation head used in training.
        eps: Numerical floor for stereo ratios.

    Input:
        ``complex_channels``: canonical interleaved real STFT ``[B, 2*C, F, T]``
        for mono (``C=1``) or stereo (``C=2``).
    Output:
        Tuple containing ``z_deg`` and optional degradation logits.
    """

    def __init__(
        self,
        degradation_embed_dim: int = 128,
        num_degradation_classes: int = 19,
        branch_dim: Optional[int] = None,
        include_training_head: bool = False,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if degradation_embed_dim <= 0:
            raise ValueError("degradation_embed_dim must be positive")
        if num_degradation_classes <= 0:
            raise ValueError("num_degradation_classes must be positive")
        self.degradation_embed_dim = int(degradation_embed_dim)
        self.num_degradation_classes = int(num_degradation_classes)
        resolved_branch_dim = (
            max(self.degradation_embed_dim // 2, 16)
            if branch_dim is None
            else int(branch_dim)
        )
        if resolved_branch_dim <= 0:
            raise ValueError("branch_dim must be positive")
        self.branch_dim = resolved_branch_dim

        self.frequency_encoder = FrequencyCueEncoder(resolved_branch_dim)
        self.temporal_encoder = TemporalCueEncoder(resolved_branch_dim)
        self.stereo_encoder = StereoCueEncoder(resolved_branch_dim, eps=eps)

        fusion_input_dim = 3 * resolved_branch_dim
        fusion_hidden_dim = max(self.degradation_embed_dim * 2, fusion_input_dim)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, self.degradation_embed_dim),
            nn.LayerNorm(self.degradation_embed_dim),
        )
        self.degradation_head = (
            nn.Linear(self.degradation_embed_dim, self.num_degradation_classes)
            if include_training_head
            else None
        )

    @property
    def frequency_branch(self) -> FrequencyCueEncoder:
        """Compatibility name matching the architecture description."""

        return self.frequency_encoder

    @property
    def temporal_branch(self) -> TemporalCueEncoder:
        """Compatibility name matching the architecture description."""

        return self.temporal_encoder

    @property
    def stereo_branch(self) -> StereoCueEncoder:
        """Compatibility name matching the architecture description."""

        return self.stereo_encoder

    @staticmethod
    def _validate_input(complex_channels: torch.Tensor) -> None:
        if torch.is_complex(complex_channels) or not complex_channels.is_floating_point():
            raise TypeError("complex_channels must be a real floating-point tensor")
        if complex_channels.ndim != 4:
            raise ValueError(
                "complex_channels must have shape [B, 2*C, F, T], got {}".format(
                    tuple(complex_channels.shape)
                )
            )
        if complex_channels.shape[0] <= 0:
            raise ValueError("batch dimension must be non-empty")
        if complex_channels.shape[1] not in (2, 4):
            raise ValueError(
                "CompoundDegradationEncoder supports canonical mono [B,2,F,T] "
                "or stereo [B,4,F,T] input"
            )
        if complex_channels.shape[2] <= 0 or complex_channels.shape[3] <= 0:
            raise ValueError("frequency and time dimensions must be non-empty")

    @staticmethod
    def _standardize_branch_channels(complex_channels: torch.Tensor) -> torch.Tensor:
        """Map mono/stereo interleaved input to a fixed ``[B, 4, F, T]``."""

        if complex_channels.shape[1] == 4:
            return complex_channels
        # Duplicating mono produces L == R and therefore a physically sensible
        # zero Side signal, while keeping learned branch input widths fixed.
        return torch.cat((complex_channels, complex_channels), dim=1)

    def forward(
        self, complex_channels: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._validate_input(complex_channels)
        clean_channels = _finite(complex_channels)

        # All convolution branches follow the module's current parameter dtype;
        # autocast may still select fp16/bf16 kernels at call time.
        parameter = next(self.frequency_encoder.parameters())
        branch_channels = self._standardize_branch_channels(clean_channels).to(
            device=parameter.device, dtype=parameter.dtype
        )
        z_frequency = self.frequency_encoder(branch_channels)
        z_temporal = self.temporal_encoder(branch_channels)

        # This shared helper is the single authority for interleaved layout.
        complex_spec = channels_to_complex(clean_channels)
        z_stereo = self.stereo_encoder(complex_spec)

        fused = torch.cat((z_frequency, z_temporal, z_stereo), dim=-1)
        z_deg = _finite(self.fusion_mlp(fused))

        logits = (
            _finite(self.degradation_head(z_deg))
            if self.degradation_head is not None
            else None
        )
        return z_deg, logits


__all__ = [
    "CompoundDegradationEncoder",
    "FrequencyCueEncoder",
    "StereoCueEncoder",
    "TemporalCueEncoder",
]
