"""Deterministic, non-overlapping full-track windows for F-MoE training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from data.audio_dataset import AudioTrainDataset
from utils.audio_utils import ensure_audio_channels

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None


@dataclass(frozen=True)
class WindowSpec:
    start: int
    valid_frames: int
    pad_right: int


def enumerate_nonoverlap_windows(
    num_frames: int, window_frames: int, tail_policy: str = "pad"
) -> Tuple[WindowSpec, ...]:
    """Cover every valid frame exactly once with stride equal to window size.

    The final short window is right-zero-padded. Padding never counts as source
    coverage, so valid source intervals remain disjoint and exhaustive.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    if tail_policy != "pad":
        raise ValueError(f"unsupported tail_policy={tail_policy!r}; expected 'pad'")
    result: List[WindowSpec] = []
    for start in range(0, num_frames, window_frames):
        valid = min(window_frames, num_frames - start)
        result.append(
            WindowSpec(
                start=start,
                valid_frames=valid,
                pad_right=window_frames - valid,
            )
        )
    return tuple(result)


def _header_num_frames(path: str, expected_sample_rate: int) -> int:
    if sf is not None:
        info = sf.info(path)
        if int(info.samplerate) != expected_sample_rate:
            return int(round(info.frames * expected_sample_rate / info.samplerate))
        return int(info.frames)
    if torchaudio is None:
        raise RuntimeError("soundfile or torchaudio is required for window indexing")
    info = torchaudio.info(path)
    if int(info.sample_rate) != expected_sample_rate:
        return int(round(info.num_frames * expected_sample_rate / info.sample_rate))
    return int(info.num_frames)


class NonOverlappingSegmentDataset(AudioTrainDataset):
    """Enumerate every non-overlapping window of every pre-degraded pair."""

    def __init__(self, args, split: str = "train"):
        super().__init__(args, split=split)
        if not self.use_predegraded_pairs:
            raise ValueError("non-overlapping segment training requires paired audio")
        self.tail_policy = "pad"
        self.window_stride = self.segment_len

        # A clean track is shared by its DT1..DT7 pairs. Read its header once.
        frame_cache: Dict[str, int] = {}
        self.window_index: List[Tuple[int, WindowSpec]] = []
        self.track_frame_counts: Dict[str, int] = {}
        for pair_index, (_, clean_path, _) in enumerate(self.pairs):
            if clean_path not in frame_cache:
                frame_cache[clean_path] = _header_num_frames(
                    clean_path, int(args.sample_rate)
                )
            num_frames = frame_cache[clean_path]
            self.track_frame_counts[clean_path] = num_frames
            for window in enumerate_nonoverlap_windows(
                num_frames, self.segment_len, self.tail_policy
            ):
                self.window_index.append((pair_index, window))
        self.effective_len = len(self.window_index)
        if not self.window_index:
            raise RuntimeError(f"empty segment index for split={split!r}")

    def get_waveform_pair(self, idx: int, return_metadata: bool = False):
        pair_index, window = self.window_index[idx]
        degraded_path, clean_path, dt_raw = self.pairs[pair_index]
        de_id = self.de_dict[dt_raw]

        degraded_wav, sr = self._load_audio(degraded_path)
        clean_wav, clean_sr = self._load_audio(clean_path)
        if clean_sr != sr:
            raise ValueError(
                f"sample-rate mismatch: {degraded_path} ({sr}) vs {clean_path} ({clean_sr})"
            )
        degraded_wav = ensure_audio_channels(degraded_wav, self.audio_channels)
        clean_wav = ensure_audio_channels(clean_wav, self.audio_channels)
        degraded_len = int(degraded_wav.size(-1))
        clean_len = int(clean_wav.size(-1))
        if abs(degraded_len - clean_len) > self.pair_length_tolerance_samples:
            raise ValueError(
                "decoded pair length mismatch: degraded={} clean={} tolerance={} "
                "degraded_path={!r} clean_path={!r}".format(
                    degraded_len,
                    clean_len,
                    self.pair_length_tolerance_samples,
                    degraded_path,
                    clean_path,
                )
            )
        actual_frames = min(degraded_len, clean_len)
        indexed_frames = self.track_frame_counts[clean_path]
        if actual_frames != indexed_frames:
            raise RuntimeError(
                f"header/decode frame mismatch for {clean_path!r}: "
                f"indexed={indexed_frames}, decoded={actual_frames}"
            )
        stop = window.start + window.valid_frames
        degraded_wav = degraded_wav[..., window.start:stop]
        clean_wav = clean_wav[..., window.start:stop]
        if window.pad_right:
            degraded_wav = torch.nn.functional.pad(
                degraded_wav, (0, window.pad_right), value=0.0
            )
            clean_wav = torch.nn.functional.pad(
                clean_wav, (0, window.pad_right), value=0.0
            )
        if degraded_wav.shape != (self.audio_channels, self.segment_len):
            raise RuntimeError(
                f"unexpected degraded shape {tuple(degraded_wav.shape)}; "
                f"expected {(self.audio_channels, self.segment_len)}"
            )
        if clean_wav.shape != degraded_wav.shape:
            raise RuntimeError(
                f"clean/degraded shape mismatch: {clean_wav.shape} vs {degraded_wav.shape}"
            )

        if return_metadata:
            chain = self._degradation_chain(clean_path, dt_raw, de_id)
            chain = dict(chain)
            chain["window_start_frame"] = window.start
            chain["window_valid_frames"] = window.valid_frames
            chain["window_pad_right"] = window.pad_right
            return degraded_wav, clean_wav, sr, clean_path, de_id, chain
        return degraded_wav, clean_wav, sr, clean_path, de_id

    def coverage_summary(self) -> dict:
        frame_counts = list(self.track_frame_counts.values())
        return {
            "split": self.split,
            "pairs": len(self.pairs),
            "unique_tracks": len(self.track_frame_counts),
            "windows": len(self.window_index),
            "window_frames": self.segment_len,
            "window_stride_frames": self.window_stride,
            "tail_policy": self.tail_policy,
            "min_track_frames": min(frame_counts),
            "max_track_frames": max(frame_counts),
            "mean_windows_per_pair": len(self.window_index) / len(self.pairs),
            "stereo_channels": self.audio_channels,
        }
