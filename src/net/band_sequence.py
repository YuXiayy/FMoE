"""Band-sequence restoration blocks for decoder latent feature maps.

The expert in this module borrows the alternating temporal/band sequence
structure used by band-split audio models, but operates on a decoder latent
``[B, C, F, T]`` rather than on a raw complex STFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


__all__ = [
    "BandSpec",
    "build_multiresolution_bands",
    "validate_bands",
    "BandProjection",
    "TemporalSequenceBlock",
    "BandSequenceBlock",
    "BandReconstruction",
    "BandSequenceExpert",
]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class BandSpec:
    """A contiguous frequency band using the half-open interval ``[start, end)``."""

    start_bin: int
    end_bin: int

    def __post_init__(self) -> None:
        if isinstance(self.start_bin, bool) or not isinstance(self.start_bin, int):
            raise TypeError("start_bin must be an integer")
        if isinstance(self.end_bin, bool) or not isinstance(self.end_bin, int):
            raise TypeError("end_bin must be an integer")
        if self.start_bin < 0:
            raise ValueError(f"start_bin must be non-negative, got {self.start_bin}")
        if self.end_bin <= self.start_bin:
            raise ValueError(
                "a band must contain at least one bin: "
                f"[{self.start_bin}, {self.end_bin})"
            )

    @property
    def width(self) -> int:
        """Number of frequency bins in this band."""

        return self.end_bin - self.start_bin


def validate_bands(bands: Sequence[BandSpec], num_bins: int) -> None:
    """Validate that ``bands`` cover ``[0, num_bins)`` exactly once."""

    _positive_int("num_bins", num_bins)
    if not bands:
        raise ValueError("bands must not be empty")

    expected_start = 0
    for index, band in enumerate(bands):
        if not isinstance(band, BandSpec):
            raise TypeError(f"bands[{index}] must be a BandSpec, got {type(band)!r}")
        if band.start_bin != expected_start:
            relation = "overlap" if band.start_bin < expected_start else "gap"
            raise ValueError(
                f"band {index} introduces a {relation}: expected start "
                f"{expected_start}, got {band.start_bin}"
            )
        if band.end_bin > num_bins:
            raise ValueError(
                f"band {index} ends at {band.end_bin}, beyond num_bins={num_bins}"
            )
        expected_start = band.end_bin

    if expected_start != num_bins:
        raise ValueError(
            f"bands stop at {expected_start}; expected complete coverage through {num_bins}"
        )


def _partition_interval(start: int, end: int, width: int) -> List[BandSpec]:
    bands: List[BandSpec] = []
    while start < end:
        next_end = min(start + width, end)
        bands.append(BandSpec(start, next_end))
        start = next_end
    return bands


def _fit_band_count(
    bands: Sequence[BandSpec], num_bins: int, num_bands: int
) -> List[BandSpec]:
    """Merge or split an ordered multiresolution partition to exactly K bands."""

    if len(bands) == num_bands:
        return list(bands)

    if len(bands) > num_bands:
        # Equal groups in *band space* retain more boundaries where the base
        # multiresolution layout is dense (normally at low frequencies).
        fitted: List[BandSpec] = []
        for index in range(num_bands):
            first = (index * len(bands)) // num_bands
            last = ((index + 1) * len(bands)) // num_bands
            fitted.append(BandSpec(bands[first].start_bin, bands[last - 1].end_bin))
        validate_bands(fitted, num_bins)
        return fitted

    fitted = list(bands)
    while len(fitted) < num_bands:
        splittable = [
            (band.width, index) for index, band in enumerate(fitted) if band.width > 1
        ]
        if not splittable:  # Guarded by num_bands <= num_bins, kept for clarity.
            raise RuntimeError("cannot produce the requested number of non-empty bands")
        _, index = max(splittable)
        band = fitted[index]
        midpoint = band.start_bin + band.width // 2
        fitted[index : index + 1] = [
            BandSpec(band.start_bin, midpoint),
            BandSpec(midpoint, band.end_bin),
        ]

    validate_bands(fitted, num_bins)
    return fitted


def build_multiresolution_bands(
    sample_rate: int,
    n_fft: int,
    boundaries_hz: Sequence[float] = (1000, 2000, 4000, 8000, 16000),
    bins_per_band: Sequence[int] = (2, 4, 12, 24, 48),
    *,
    num_bins: Optional[int] = None,
    num_bands: Optional[int] = None,
) -> List[BandSpec]:
    """Build contiguous bands that are narrow at low and wide at high frequency.

    Args:
        sample_rate: Audio sample rate used to interpret ``boundaries_hz``.
        n_fft: Reference STFT size.  With no ``num_bins`` override the output
            covers ``n_fft // 2 + 1`` one-sided bins.
        boundaries_hz: Increasing upper frequency boundaries for resolution
            regions.
        bins_per_band: Target width in reference-STFT bins per region.  It may
            contain one width per boundary (the last is reused for the tail),
            or one additional explicit tail width.
        num_bins: Actual decoder frequency dimension.  Target widths and Hz
            boundaries are scaled onto this dimension, so the function does
            not assume that a decoder stage has raw-STFT ``F``.
        num_bands: If given, merge/split the multiresolution layout to exactly
            this many non-empty bands while preserving full coverage.

    Returns:
        Ordered, non-overlapping ``BandSpec`` objects covering ``[0, F)``.
    """

    _positive_int("sample_rate", sample_rate)
    _positive_int("n_fft", n_fft)
    raw_num_bins = n_fft // 2 + 1
    actual_num_bins = raw_num_bins if num_bins is None else _positive_int(
        "num_bins", num_bins
    )

    boundaries = tuple(float(value) for value in boundaries_hz)
    widths = tuple(bins_per_band)
    if any(not math.isfinite(value) or value <= 0 for value in boundaries):
        raise ValueError("boundaries_hz must contain positive finite values")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries_hz must be strictly increasing")
    if not widths:
        raise ValueError("bins_per_band must not be empty")
    if boundaries and len(widths) not in (len(boundaries), len(boundaries) + 1):
        raise ValueError(
            "bins_per_band must have one entry per boundary, with an optional "
            "additional entry for the Nyquist tail"
        )
    if not boundaries and len(widths) != 1:
        raise ValueError("without boundaries_hz, bins_per_band must contain one width")
    for index, width in enumerate(widths):
        _positive_int(f"bins_per_band[{index}]", width)

    if num_bands is not None:
        _positive_int("num_bands", num_bands)
        if num_bands > actual_num_bins:
            raise ValueError(
                "num_bands must not exceed the actual decoder frequency bins "
                f"(K <= F), got {num_bands} > {actual_num_bins}"
            )

    # Decoder down/up-sampling changes the number of latent rows, not the
    # represented 0..highest-one-sided-frequency span.  Scale both boundaries
    # and target widths to the actual latent F.
    raw_last_index = max(raw_num_bins - 1, 1)
    actual_last_index = max(actual_num_bins - 1, 1)
    highest_frequency = (n_fft // 2) * float(sample_rate) / n_fft
    width_scale = actual_last_index / raw_last_index

    scaled_widths = [max(1, int(round(width * width_scale))) for width in widths]
    region_ends: List[int] = []
    for boundary in boundaries:
        if actual_num_bins == 1 or highest_frequency == 0:
            end = actual_num_bins
        else:
            end = int(math.ceil(boundary / highest_frequency * actual_last_index))
            end = min(max(end, 0), actual_num_bins)
        region_ends.append(end)

    bands: List[BandSpec] = []
    start = 0
    for region_index, region_end in enumerate(region_ends):
        if region_end > start:
            bands.extend(
                _partition_interval(start, region_end, scaled_widths[region_index])
            )
            start = region_end

    tail_width = scaled_widths[-1]
    if start < actual_num_bins:
        bands.extend(_partition_interval(start, actual_num_bins, tail_width))

    # A boundary above the represented maximum can consume the whole range.
    if not bands:
        bands = [BandSpec(0, actual_num_bins)]

    validate_bands(bands, actual_num_bins)
    if num_bands is not None:
        bands = _fit_band_count(bands, actual_num_bins, num_bands)
    return bands


class BandProjection(nn.Module):
    """Project variable-width bands from ``[B, C, F, T]`` to ``[B, T, K, D]``."""

    def __init__(
        self, channels: int, bands: Sequence[BandSpec], embed_dim: int
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.bands = tuple(bands)
        if not self.bands:
            raise ValueError("bands must not be empty")
        self.freq_bins = self.bands[-1].end_bin
        validate_bands(self.bands, self.freq_bins)

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.channels * band.width),
                    nn.Linear(self.channels * band.width, self.embed_dim),
                )
                for band in self.bands
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``x`` from ``[B, C, F, T]`` to ``[B, T, K, D]``."""

        if x.ndim != 4:
            raise ValueError(f"expected [B,C,F,T], got shape {tuple(x.shape)}")
        _, channels, freq_bins, _ = x.shape
        if channels != self.channels or freq_bins != self.freq_bins:
            raise ValueError(
                "BandProjection was built for "
                f"C={self.channels}, F={self.freq_bins}; got C={channels}, F={freq_bins}"
            )

        projected = []
        for band, projection in zip(self.bands, self.projections):
            # [B,C,F_k,T] -> [B,T,C*F_k]
            band_input = x[:, :, band.start_bin : band.end_bin, :]
            flattened = band_input.permute(0, 3, 1, 2).reshape(
                x.shape[0], x.shape[3], self.channels * band.width
            )
            projected.append(projection(flattened))
        return torch.stack(projected, dim=2)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.rotary_dim = head_dim - (head_dim % 2)
        if self.rotary_dim:
            exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
            exponent = exponent / self.rotary_dim
            inv_freq = base ** (-exponent)
        else:
            inv_freq = torch.empty(0, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == 0:
            return x
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.float())
        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

        rotary, remainder = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        even, odd = rotary[..., 0::2], rotary[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)
        return torch.cat((rotated, remainder), dim=-1)


class _RotarySelfAttention(nn.Module):
    """Batch-first multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        _positive_int("embed_dim", embed_dim)
        _positive_int("num_heads", num_heads)
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.rotary = _RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q = self.rotary(q)
        k = self.rotary(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        mixed = torch.matmul(attention, value)
        mixed = mixed.transpose(1, 2).contiguous().reshape(batch, length, self.embed_dim)
        return self.output(mixed)


class _FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Linear(embed_dim * expansion, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TemporalSequenceBlock(nn.Module):
    """Model the T axis of ``[B,T,K,D]`` with pre-norm residual mixing and FFN."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.mixer_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.mixer = _RotarySelfAttention(embed_dim, num_heads)
        self.ffn = _FeedForward(embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Process T via ``[B,T,K,D] -> [B*K,T,D] -> [B,T,K,D]``."""

        if z.ndim != 4 or z.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected [B,T,K,{self.embed_dim}], got shape {tuple(z.shape)}"
            )
        batch, time, bands, dim = z.shape
        normalized = self.mixer_norm(z)
        sequence = normalized.permute(0, 2, 1, 3).reshape(batch * bands, time, dim)
        sequence = self.mixer(sequence)
        mixed = sequence.reshape(batch, bands, time, dim).permute(0, 2, 1, 3)
        z = z + mixed
        return z + self.ffn(self.ffn_norm(z))


class BandSequenceBlock(nn.Module):
    """Model the K axis of ``[B,T,K,D]`` using RoPE attention and an FFN."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.attention = _RotarySelfAttention(embed_dim, num_heads)
        self.ffn = _FeedForward(embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Process K via ``[B,T,K,D] -> [B*T,K,D] -> [B,T,K,D]``."""

        if z.ndim != 4 or z.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected [B,T,K,{self.embed_dim}], got shape {tuple(z.shape)}"
            )
        batch, time, bands, dim = z.shape
        sequence = self.attention_norm(z).reshape(batch * time, bands, dim)
        attended = self.attention(sequence).reshape(batch, time, bands, dim)
        z = z + attended
        return z + self.ffn(self.ffn_norm(z))


class _BandSequenceLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.temporal = TemporalSequenceBlock(embed_dim, num_heads)
        self.band = BandSequenceBlock(embed_dim, num_heads)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.band(self.temporal(z))


class BandReconstruction(nn.Module):
    """Reconstruct each band and concatenate ``[B,T,K,D] -> [B,C,F,T]``."""

    def __init__(
        self,
        channels: int,
        bands: Sequence[BandSpec],
        embed_dim: int,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.bands = tuple(bands)
        if not self.bands:
            raise ValueError("bands must not be empty")
        self.freq_bins = self.bands[-1].end_bin
        validate_bands(self.bands, self.freq_bins)
        self.projections = nn.ModuleList(
            [nn.Linear(embed_dim, self.channels * band.width) for band in self.bands]
        )
        if zero_init:
            for projection in self.projections:
                nn.init.zeros_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)

    @property
    def output_projections(self) -> nn.ModuleList:
        """Per-band output projections (exposed for initialization inspection)."""

        return self.projections

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Transform ``z`` from ``[B,T,K,D]`` to ``[B,C,F,T]``."""

        if z.ndim != 4:
            raise ValueError(f"expected [B,T,K,D], got shape {tuple(z.shape)}")
        batch, time, num_bands, dim = z.shape
        if num_bands != len(self.bands) or dim != self.embed_dim:
            raise ValueError(
                f"expected K={len(self.bands)}, D={self.embed_dim}; "
                f"got K={num_bands}, D={dim}"
            )

        reconstructed = []
        for index, (band, projection) in enumerate(zip(self.bands, self.projections)):
            decoded = projection(z[:, :, index, :])
            decoded = decoded.reshape(batch, time, self.channels, band.width)
            reconstructed.append(decoded.permute(0, 2, 3, 1))
        return torch.cat(reconstructed, dim=2)


class BandSequenceExpert(nn.Module):
    """Alternating temporal/band expert for a decoder latent feature map.

    Input and output both have shape ``[B, channels, freq_bins, T]``.  The
    reconstruction projections are zero-initialized and are applied as a
    residual, so a newly constructed expert is exactly shape- and value-safe.
    """

    def __init__(
        self,
        channels: int,
        freq_bins: int,
        embed_dim: int = 64,
        num_bands: int = 16,
        num_blocks: int = 2,
        num_heads: int = 4,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        *,
        boundaries_hz: Sequence[float] = (1000, 2000, 4000, 8000, 16000),
        bins_per_band: Sequence[int] = (2, 4, 12, 24, 48),
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.freq_bins = _positive_int("freq_bins", freq_bins)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.num_bands = _positive_int("num_bands", num_bands)
        self.num_blocks = _positive_int("num_blocks", num_blocks)
        self.num_heads = _positive_int("num_heads", num_heads)
        self.sample_rate = _positive_int("sample_rate", sample_rate)
        self.n_fft = _positive_int("n_fft", n_fft)
        if self.num_bands > self.freq_bins:
            raise ValueError(
                "num_bands must not exceed freq_bins (K <= F), "
                f"got {self.num_bands} > {self.freq_bins}"
            )
        if self.embed_dim % self.num_heads:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        bands = build_multiresolution_bands(
            self.sample_rate,
            self.n_fft,
            boundaries_hz,
            bins_per_band,
            num_bins=self.freq_bins,
            num_bands=self.num_bands,
        )
        self.band_specs: Tuple[BandSpec, ...] = tuple(bands)
        # A convenient read-only alias for callers that use the shorter name.
        self.bands = self.band_specs

        self.projection = BandProjection(
            self.channels, self.band_specs, self.embed_dim
        )
        self.blocks = nn.ModuleList(
            [
                _BandSequenceLayer(self.embed_dim, self.num_heads)
                for _ in range(self.num_blocks)
            ]
        )
        self.reconstruction = BandReconstruction(
            self.channels, self.band_specs, self.embed_dim, zero_init=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore ``x`` while preserving ``[B,C,F,T]`` exactly."""

        if x.ndim != 4:
            raise ValueError(f"expected [B,C,F,T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels or x.shape[2] != self.freq_bins:
            raise ValueError(
                f"expert was built for C={self.channels}, F={self.freq_bins}; "
                f"got C={x.shape[1]}, F={x.shape[2]}"
            )
        z = self.projection(x)
        for block in self.blocks:
            z = block(z)
        residual = self.reconstruction(z)
        return x + residual
