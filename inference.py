"""Restore one audio file or every supported file in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Iterable

import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net.model_factory import load_pretrained_fmoe
from utils.audio_utils import (
    complex_to_channels,
    ensure_audio_channels,
    istft_from_channels,
    pad_spectrogram_for_fmoe,
    stft_complex,
    unpad_spectrogram,
)
from utils.inference_utils import crossfade_audio_chunks, inference_chunk_starts

SUPPORTED_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _inputs(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported audio extension: {path.suffix}")
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES:
            yield item


@torch.inference_mode()
def restore_waveform(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    sample_rate: int,
    device: torch.device,
    segment_seconds: float = 5.0,
    overlap: float = 0.25,
) -> torch.Tensor:
    target_sr = 44_100
    waveform = ensure_audio_channels(waveform, 2)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    total = waveform.shape[-1]
    if total < 1:
        raise ValueError("input audio must contain at least one sample")
    segment = max(1, round(segment_seconds * target_sr))
    starts = inference_chunk_starts(total, segment, overlap)
    restored_chunks = []

    for start in starts:
        valid = min(segment, total - start)
        chunk = waveform[:, start : start + valid]
        if valid < segment:
            chunk = torch.nn.functional.pad(chunk, (0, segment - valid))
        chunk = chunk.to(device)
        channels = complex_to_channels(
            stft_complex(chunk, n_fft=1024, hop_length=256, win_length=1024)
        )
        padded, pad_info = pad_spectrogram_for_fmoe(channels, multiple=8)
        prediction = model(padded.unsqueeze(0))
        restored = prediction["restored_spec"] if isinstance(prediction, dict) else prediction
        restored = unpad_spectrogram(restored.squeeze(0), pad_info=pad_info)
        restored = istft_from_channels(
            restored,
            length=segment,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )[:, :valid]
        restored_chunks.append(restored.cpu())

    return crossfade_audio_chunks(restored_chunks, starts, total).clamp(-1.0, 1.0)


def _destination_paths(source_root: Path, files: list[Path], output_dir: Path) -> dict[Path, Path]:
    """Build collision-free output paths while preserving folder structure."""

    relative_paths = {
        source: source.relative_to(source_root) if source_root.is_dir() else Path(source.name)
        for source in files
    }
    base_keys = [
        (relative.parent, relative.stem.casefold())
        for relative in relative_paths.values()
    ]
    duplicate_counts = Counter(base_keys)
    destinations = {}
    for source, relative in relative_paths.items():
        key = (relative.parent, relative.stem.casefold())
        suffix_tag = f"_{source.suffix[1:].lower()}" if duplicate_counts[key] > 1 else ""
        destinations[source] = (
            output_dir
            / relative.parent
            / f"{relative.stem}{suffix_tag}_restored.wav"
        )
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "inference_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_pretrained_fmoe(args.checkpoint, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = list(_inputs(args.input))
    if not files:
        raise RuntimeError(f"no supported audio files found under {args.input}")
    destinations = _destination_paths(args.input, files, args.output_dir)
    for source in files:
        waveform, sample_rate = torchaudio.load(source)
        restored = restore_waveform(
            model,
            waveform,
            sample_rate,
            device,
            segment_seconds=args.segment_seconds,
            overlap=args.overlap,
        )
        destination = destinations[source]
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, restored.transpose(0, 1).numpy(), 44_100, subtype="PCM_24")
        print(destination)


if __name__ == "__main__":
    main()
