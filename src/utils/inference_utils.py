"""Canonical long-audio segmentation and overlap-add for F-MoE inference."""

from __future__ import annotations

from typing import Sequence

import torch


def inference_chunk_starts(
    total_length: int,
    segment_length: int,
    overlap_ratio: float,
) -> list[int]:
    """Return deterministic chunk starts with complete tail coverage."""

    if total_length < 1:
        raise ValueError("audio must contain at least one sample")
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if total_length <= segment_length:
        return [0]
    if overlap_ratio == 0.0:
        return list(range(0, total_length, segment_length))

    hop = max(1, int(round(segment_length * (1.0 - overlap_ratio))))
    last_start = total_length - segment_length
    starts = list(range(0, last_start + 1, hop))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def crossfade_audio_chunks(
    chunks: Sequence[torch.Tensor],
    starts: Sequence[int],
    total_length: int,
) -> torch.Tensor:
    """Combine ``[C, T]`` chunks with the paper evaluation Hann weighting."""

    if not chunks:
        raise ValueError("at least one restored chunk is required")
    if len(chunks) != len(starts):
        raise ValueError("chunk and start counts must match")
    if total_length < 1:
        raise ValueError("total_length must be positive")

    first = chunks[0]
    if first.ndim != 2:
        raise ValueError("restored chunks must have shape [channels, time]")
    output = first.new_zeros((first.size(0), total_length))
    weights = first.new_zeros((1, total_length))
    use_window = len(chunks) > 1 and any(
        starts[index] < starts[index - 1] + chunks[index - 1].size(-1)
        for index in range(1, len(starts))
    )

    for chunk, start in zip(chunks, starts):
        if chunk.ndim != 2 or chunk.size(0) != output.size(0):
            raise ValueError("all restored chunks must share their channel count")
        if start < 0:
            raise ValueError("chunk starts must be non-negative")
        length = min(int(chunk.size(-1)), total_length - int(start))
        if length <= 0:
            continue
        if use_window and length > 1:
            window = torch.hann_window(
                length,
                periodic=False,
                dtype=chunk.dtype,
                device=chunk.device,
            ).clamp_min(1.0e-3)
        else:
            window = chunk.new_ones(length)
        output[..., start : start + length] += chunk[..., :length] * window
        weights[..., start : start + length] += window

    if bool((weights <= 0).any()):
        raise RuntimeError("inference overlap-add left uncovered output samples")
    return output / weights


__all__ = ["crossfade_audio_chunks", "inference_chunk_starts"]
