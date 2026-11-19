"""SonicMaster paired test-set evaluation implementation."""

import os
import pathlib
import csv
import numpy as np
import random
import tempfile
import time
import warnings

from tqdm import tqdm
from typing import List
import torch
import lightning.pytorch as pl

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

try:
    from pesq import pesq as pesq_score
except Exception:
    pesq_score = None

from net.model_factory import build_fmoe
from data.audio_dataset import AudioTestDataset
from utils.audio_utils import (
    complex_to_channels,
    ensure_audio_channels,
    istft_from_channels,
    istft_from_mag_phase,
    pad_spectrogram_for_fmoe,
    stft_complex,
    stft_mag_phase,
    unpad_spectrogram,
)
from utils.checkpoint import load_strict_model_checkpoint, resolve_checkpoint_path
from utils.inference_utils import crossfade_audio_chunks, inference_chunk_starts


_METRIC_WARNINGS = set()


def _warn_once(key: str, message: str):
    if key in _METRIC_WARNINGS:
        return
    _METRIC_WARNINGS.add(key)
    print(message)


def _to_1d_numpy(wav: torch.Tensor) -> np.ndarray:
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    return wav.detach().cpu().float().numpy()



def _save_audio(path: str, wav: torch.Tensor, sr: int):
    """
    Save audio with soundfile.
    Input wav shape: [C, T] or [T]
    """
    if sf is None:
        raise ImportError("soundfile is required for audio saving. Please install soundfile.")

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    peak = float(wav.detach().abs().max().item()) if wav.numel() else 0.0
    if peak > 1.0:
        warnings.warn(
            f"Audio peak {peak:.6f} exceeds full scale; samples will be clipped "
            f"when saving {path!r}. No peak normalization was applied.",
            RuntimeWarning,
            stacklevel=2,
        )
    wav = torch.clamp(wav, -1.0, 1.0).detach().cpu().float()
    wav_np = wav.numpy().T  # [T, C]
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(6):
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
        )
        os.close(fd)
        temporary = pathlib.Path(temporary_name)
        try:
            sf.write(str(temporary), wav_np, sr)
            info = sf.info(str(temporary))
            if (
                info.samplerate != sr
                or info.frames != wav_np.shape[0]
                or info.channels != wav_np.shape[1]
            ):
                raise RuntimeError(f"invalid temporary audio header: {info}")
            os.replace(temporary, destination)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 5:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"failed to atomically write {path!r} after 6 attempts") from last_error


def _split_audio_chunks(wav: torch.Tensor, sr: int, segment_seconds: float) -> List[torch.Tensor]:
    """Split one waveform after the benchmark's explicit stereo-to-mono mix.

    SonicMaster's non-stereo metrics load audio with ``mono=True``.  Averaging
    channels here matches that contract and avoids the old behavior of
    concatenating left and right samples along time with ``reshape(-1)``.
    """
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    elif wav.ndim != 1:
        raise ValueError(
            "metric waveform must have shape [T] or [C,T], got "
            f"{tuple(wav.shape)}"
        )
    if wav.numel() == 0:
        return []
    seg_len = max(1, int(round(segment_seconds * sr)))
    chunks = []
    for start in range(0, wav.numel(), seg_len):
        end = min(start + seg_len, wav.numel())
        chunk = wav[start:end]
        if chunk.numel() > 0:
            chunks.append(chunk)
    return chunks


def _inference_chunk_starts(total_length: int, segment_length: int, overlap_ratio: float):
    return inference_chunk_starts(total_length, segment_length, overlap_ratio)


def _crossfade_audio_chunks(restored_chunks, starts, total_length):
    return crossfade_audio_chunks(restored_chunks, starts, total_length)


def _restore_waveform_by_segments(net, degraded_wav: torch.Tensor, sr: int, opts) -> torch.Tensor:
    """
    Split long degraded waveform into fixed segments, restore segment by segment,
    and stitch restored segments back to one waveform.
    """
    segment_seconds = getattr(opts, "inference_segment_seconds", None)
    if segment_seconds is None:
        segment_seconds = getattr(opts, "segment_seconds", 10.0)

    if degraded_wav.numel() == 0:
        return degraded_wav.clone()

    seg_len = max(1, int(round(segment_seconds * sr)))
    overlap_ratio = float(getattr(opts, "inference_overlap_ratio", 0.0))
    device = next(net.parameters()).device

    complex_representation = getattr(opts, "representation", "magnitude") == "complex"
    if complex_representation:
        audio_channels = 2 if getattr(opts, "stereo_mode", "mono") == "stereo" else 1
        degraded_wav = ensure_audio_channels(degraded_wav, audio_channels)
    else:
        degraded_wav = ensure_audio_channels(degraded_wav, 1)

    total_length = int(degraded_wav.size(-1))
    starts = _inference_chunk_starts(total_length, seg_len, overlap_ratio)
    restored_chunks = []
    for start in starts:
        chunk = degraded_wav[..., start:start + seg_len]
        chunk_len = int(chunk.size(-1))
        if chunk_len < seg_len:
            chunk = torch.nn.functional.pad(chunk, (0, seg_len - chunk_len))

        if complex_representation:
            chunk = chunk.to(device)
            degraded_spec = complex_to_channels(
                stft_complex(
                    chunk,
                    n_fft=opts.n_fft,
                    hop_length=opts.hop_length,
                    win_length=opts.win_length,
                    center=opts.stft_center,
                )
            )
            padded_spec, pad_info = pad_spectrogram_for_fmoe(degraded_spec, multiple=8)
            output = net(padded_spec.unsqueeze(0))
            if isinstance(output, dict):
                output = output["restored_spec"]
            elif isinstance(output, (list, tuple)):
                output = output[0]
            restored_spec = unpad_spectrogram(output.squeeze(0), pad_info)
            restored_chunk = istft_from_channels(
                restored_spec,
                length=seg_len,
                n_fft=opts.n_fft,
                hop_length=opts.hop_length,
                win_length=opts.win_length,
                center=opts.stft_center,
            )
            restored_chunks.append(restored_chunk[..., :chunk_len].cpu())
        else:
            chunk_wav = chunk.to(device)
            degraded_mag, degraded_phase, _ = stft_mag_phase(
                chunk_wav,
                n_fft=opts.n_fft,
                hop_length=opts.hop_length,
                win_length=opts.win_length,
                spec_log=opts.spec_log,
            )
            degraded_mag, _ = pad_spectrogram_for_fmoe(degraded_mag, multiple=8)
            pred_mag = net(degraded_mag.unsqueeze(0))
            if isinstance(pred_mag, dict):
                pred_mag = pred_mag["restored_spec"]
            elif isinstance(pred_mag, (list, tuple)):
                pred_mag = pred_mag[0]
            pred_mag = torch.clamp(pred_mag, min=0.0)

            target_f = degraded_phase.size(0)
            target_t = degraded_phase.size(1)
            pred_mag_unpadded = unpad_spectrogram(
                pred_mag.squeeze(0), target_f, target_t
            )
            restored_chunk = istft_from_mag_phase(
                pred_mag_unpadded,
                degraded_phase,
                length=seg_len,
                n_fft=opts.n_fft,
                hop_length=opts.hop_length,
                win_length=opts.win_length,
                spec_log=opts.spec_log,
            ).cpu().reshape(1, -1)
            restored_chunks.append(restored_chunk[..., :chunk_len])

    return _crossfade_audio_chunks(restored_chunks, starts, total_length)


def _sispnr(clean_spec: torch.Tensor, pred_spec: torch.Tensor, spec_log: bool = True) -> float:
    """Spectrum-domain SiSPNR with global projection as defined in the formula."""
    clean = clean_spec.reshape(-1)
    pred = pred_spec.reshape(-1)
    if clean.numel() == 0 or pred.numel() == 0:
        return float("nan")

    n = min(clean.numel(), pred.numel())
    clean = clean[:n]
    pred = pred[:n]

    if spec_log:
        clean = torch.expm1(torch.clamp(clean, min=0.0))
        pred = torch.expm1(torch.clamp(pred, min=0.0))

    alpha = torch.dot(pred, clean) / (torch.dot(clean, clean) + 1e-12)
    target = alpha * clean
    noise = pred - target
    ratio = (torch.sum(target * target) + 1e-12) / (torch.sum(noise * noise) + 1e-12)
    return float(10.0 * torch.log10(ratio).item())


def _sisnr(clean_wav: torch.Tensor, pred_wav: torch.Tensor) -> float:
    clean = clean_wav.reshape(-1)
    pred = pred_wav.reshape(-1)
    if clean.numel() == 0 or pred.numel() == 0:
        return float("nan")
    n = min(clean.numel(), pred.numel())
    clean = clean[:n]
    pred = pred[:n]

    clean_zm = clean - torch.mean(clean)
    pred_zm = pred - torch.mean(pred)
    alpha = torch.dot(pred_zm, clean_zm) / (torch.dot(clean_zm, clean_zm) + 1e-12)
    target = alpha * clean_zm
    noise = pred_zm - target
    ratio = (torch.sum(target * target) + 1e-12) / (torch.sum(noise * noise) + 1e-12)
    return float(10.0 * torch.log10(ratio).item())


def _lsd(ref: torch.Tensor, est: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> float:
    ref = ref.reshape(-1)
    est = est.reshape(-1)
    if ref.numel() == 0 or est.numel() == 0:
        return float("nan")
    n = min(ref.numel(), est.numel())
    ref = ref[:n]
    est = est[:n]

    window = torch.hann_window(win_length, device=ref.device)
    ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    est_spec = torch.stft(est, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

    ref_log = torch.log10(ref_spec.abs().pow(2) + 1e-12)
    est_log = torch.log10(est_spec.abs().pow(2) + 1e-12)
    diff = ref_log - est_log
    lsd_frames = torch.sqrt(torch.mean(diff * diff, dim=0))
    return float(torch.mean(lsd_frames).item())


def _resample_waveform(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav

    if resample_poly is None:
        raise ImportError("scipy is required for resampling. Please install scipy.")

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [T] -> [1, T]

    wav_np = wav.detach().cpu().numpy()  # [C, T]

    resampled_channels = []
    for ch in range(wav_np.shape[0]):
        ch_resampled = resample_poly(wav_np[ch], dst_sr, src_sr)
        resampled_channels.append(ch_resampled.astype(np.float32))

    min_len = min(x.shape[0] for x in resampled_channels)
    resampled_np = np.stack([x[:min_len] for x in resampled_channels], axis=0)  # [C, T']

    return torch.from_numpy(resampled_np)


def _pesq(ref: torch.Tensor, est: torch.Tensor, sr: int) -> float:
    if pesq_score is None:
        _warn_once("pesq_missing", "PESQ unavailable: install package 'pesq' to enable this metric.")
        return float("nan")

    def _prep_pair(target_sr: int):
        ref_rs = _resample_waveform(ref, sr, target_sr)
        est_rs = _resample_waveform(est, sr, target_sr)
        n = min(ref_rs.numel(), est_rs.numel())
        min_len = int(0.25 * target_sr)
        if n < min_len:
            return None, None

        ref_vec = ref_rs[:n].view(-1)
        est_vec = est_rs[:n].view(-1)

        ref_vec = ref_vec - torch.mean(ref_vec)
        est_vec = est_vec - torch.mean(est_vec)
        ref_peak = torch.max(torch.abs(ref_vec)).item() + 1e-9
        est_peak = torch.max(torch.abs(est_vec)).item() + 1e-9
        ref_vec = torch.clamp(ref_vec / ref_peak, -1.0, 1.0)
        est_vec = torch.clamp(est_vec / est_peak, -1.0, 1.0)

        ref_rms = torch.sqrt(torch.mean(ref_vec * ref_vec) + 1e-12).item()
        est_rms = torch.sqrt(torch.mean(est_vec * est_vec) + 1e-12).item()
        if ref_rms < 1e-4 or est_rms < 1e-4:
            return None, None

        return _to_1d_numpy(ref_vec), _to_1d_numpy(est_vec)

    try:
        ref_np, est_np = _prep_pair(16000)
        if ref_np is None or est_np is None:
            _warn_once("pesq_silence", "PESQ skipped for low-energy or too-short segments, returning NaN.")
            return float("nan")
        return float(pesq_score(16000, ref_np, est_np, "wb"))
    except Exception as exc:
        _warn_once("pesq_runtime", f"PESQ computation failed, returning NaN. Detail: {exc}")
        return float("nan")


def _compute_chunkwise_metrics(clean_wav: torch.Tensor, restored_wav: torch.Tensor, sr: int, opts) -> dict:
    segment_seconds = getattr(opts, "metric_segment_seconds", None)
    if segment_seconds is None:
        segment_seconds = getattr(opts, "segment_seconds", 10.0)

    clean_chunks = _split_audio_chunks(clean_wav, sr, segment_seconds)
    restored_chunks = _split_audio_chunks(restored_wav, sr, segment_seconds)
    n_chunks = min(len(clean_chunks), len(restored_chunks))
    if n_chunks == 0:
        return {
            "pesq": float("nan"),
            "lsd": float("nan"),
            "sispnr": float("nan"),
            "sisnr": float("nan"),
        }

    pesq_vals, lsd_vals, sispnr_vals, sisnr_vals = [], [], [], []
    for i in range(n_chunks):
        c = clean_chunks[i].unsqueeze(0)
        r = restored_chunks[i].unsqueeze(0)

        pesq_v = _pesq(c, r, sr)
        lsd_v = _lsd(c.squeeze(0), r.squeeze(0), opts.n_fft, opts.hop_length, opts.win_length)
        sisnr_v = _sisnr(c, r)

        clean_spec, _, _ = stft_mag_phase(
            c,
            n_fft=opts.n_fft,
            hop_length=opts.hop_length,
            win_length=opts.win_length,
            spec_log=opts.spec_log,
        )
        pred_spec, _, _ = stft_mag_phase(
            r,
            n_fft=opts.n_fft,
            hop_length=opts.hop_length,
            win_length=opts.win_length,
            spec_log=opts.spec_log,
        )
        sispnr_v = _sispnr(clean_spec.squeeze(0), pred_spec.squeeze(0), spec_log=opts.spec_log)

        pesq_vals.append(pesq_v)
        lsd_vals.append(lsd_v)
        sispnr_vals.append(sispnr_v)
        sisnr_vals.append(sisnr_v)

    return {
        "pesq": _mean_or_nan(pesq_vals),
        "lsd": _mean_or_nan(lsd_vals),
        "sispnr": _mean_or_nan(sispnr_vals),
        "sisnr": _mean_or_nan(sisnr_vals),
    }


def _mean_or_nan(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    arr = np.array(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _format_metric(value: float, ndigits: int = 6) -> str:
    if value is None or np.isnan(value):
        return "nan"
    return f"{value:.{ndigits}f}"


def _to_csv_value(value):
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except Exception:
        pass
    return float(value) if isinstance(value, (np.floating, float, int, np.integer)) else str(value)


def _append_metrics_csv(opts, row: dict, csv_name: str = "metrics.csv"):
    checkpoint_id = getattr(opts, "checkpoint_id", None) or "unknown_checkpoint"
    out_dir = os.path.join(os.getcwd(), f"results/{checkpoint_id}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)

    normalized_row = {k: _to_csv_value(v) for k, v in row.items()}
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fieldnames = list(normalized_row.keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.net = build_fmoe(opt, inference=True)

    def forward(self, x):
        return self.net(x)


####################################################################################################
def run_audio_test(opts, net, dataset):
    if opts.save_results:
        checkpoint_id = getattr(opts, "checkpoint_id", None) or (
            pathlib.Path(opts.checkpoint_path).stem
            if getattr(opts, "checkpoint_path", None)
            else "unknown_checkpoint"
        )
        out_dir = os.path.join(os.getcwd(), f"results/{checkpoint_id}/sonicmaster")
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    pesq_scores = []
    lsd_scores = []
    sispnr_scores = []
    sisnr_scores = []
    shard_count = int(getattr(opts, "eval_shard_count", 1))
    shard_index = int(getattr(opts, "eval_shard_index", 0))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"invalid evaluation shard: index={shard_index}, count={shard_count}"
        )
    indices = range(shard_index, len(dataset), shard_count)
    skip_internal_metrics = bool(getattr(opts, "skip_internal_metrics", False))
    print(
        f"Audio evaluation shard {shard_index + 1}/{shard_count}: "
        f"samples={len(indices)} total={len(dataset)} "
        f"skip_internal_metrics={skip_internal_metrics}"
    )
    with torch.no_grad():
        for idx in tqdm(indices):
            degraded_wav, clean_wav, sr, path, de_id = dataset.get_waveform_pair(idx)
            de_index = int(de_id.item()) if hasattr(de_id, "item") else int(de_id)
            de_name = dataset.de_dict_reverse.get(de_index, f"de{de_index}")
            base = os.path.splitext(os.path.basename(path))[0]
            restored_name = f"{base}_{de_name}_restored.wav"
            degraded_name = f"{base}_{de_name}_degraded.wav"
            clean_name = f"{base}_{de_name}_clean.wav"
            restored_wav = _restore_waveform_by_segments(net, degraded_wav, sr, opts)

            clean_wav_cpu = clean_wav.cpu()
            if not skip_internal_metrics:
                metrics = _compute_chunkwise_metrics(clean_wav_cpu, restored_wav, sr, opts)
                pesq_scores.append(metrics["pesq"])
                lsd_scores.append(metrics["lsd"])
                sispnr_scores.append(metrics["sispnr"])
                sisnr_scores.append(metrics["sisnr"])

            if out_dir is not None:
                _save_audio(os.path.join(out_dir, restored_name), restored_wav, sr)
                _save_audio(os.path.join(out_dir, degraded_name), degraded_wav.cpu(), sr)
                _save_audio(os.path.join(out_dir, clean_name), clean_wav_cpu, sr)

    if skip_internal_metrics:
        print(
            f"Saved shard {shard_index + 1}/{shard_count}: "
            f"samples={len(indices)} out_dir={out_dir}"
        )
        return

    mean_pesq = _mean_or_nan(pesq_scores)
    mean_lsd = _mean_or_nan(lsd_scores)
    mean_sispnr = _mean_or_nan(sispnr_scores)
    mean_sisnr = _mean_or_nan(sisnr_scores)

    print('Audio PESQ: {}'.format(_format_metric(mean_pesq)))
    print('Audio LSD: {}'.format(_format_metric(mean_lsd)))
    print('Audio SiSPNR: {}'.format(_format_metric(mean_sispnr)))
    print('Audio SiSNR: {}'.format(_format_metric(mean_sisnr)))

    _append_metrics_csv(
        opts,
        {
            "task": "audio",
            "benchmark": "sonicmaster",
            "de_type": "|".join(getattr(opts, "de_type", []) or []),
            "pesq": mean_pesq,
            "lsd": mean_lsd,
            "sispnr": mean_sispnr,
            "sisnr": mean_sisnr,
            "num_samples": len(pesq_scores),
        },
    )


def run_audio_single_random_test(opts, net, dataset):
    rng = random.Random(opts.audio_random_seed) if opts.audio_random_seed is not None else random

    if getattr(dataset, "use_predegraded_pairs", False):
        source_to_indices = {}
        for pair_index, (_, clean_path, degradation_name) in enumerate(dataset.pairs):
            source_to_indices.setdefault(clean_path, {})[degradation_name] = pair_index
        if not source_to_indices:
            raise ValueError("SonicMaster test dataset contains no paired sources.")
        picked_path = rng.choice(sorted(source_to_indices))
        available = source_to_indices[picked_path]
        selected_pairs = [
            (name, available[name]) for name in dataset.de_type if name in available
        ]
    else:
        if not dataset.files:
            raise ValueError("Audio test dataset is empty.")
        file_idx = rng.randrange(len(dataset.files))
        picked_path = dataset.files[file_idx]
        selected_pairs = [
            (de_name, de_idx * len(dataset.files) + file_idx)
            for de_idx, de_name in enumerate(dataset.de_type)
        ]

    if not selected_pairs:
        raise ValueError(f"No selected degradation pairs found for source {picked_path!r}")

    if opts.save_results:
        checkpoint_id = getattr(opts, "checkpoint_id", None) or (
            pathlib.Path(opts.checkpoint_path).stem
            if getattr(opts, "checkpoint_path", None)
            else "unknown_checkpoint"
        )
        out_dir = os.path.join(
            os.getcwd(), f"results/{checkpoint_id}/sonicmaster_single_random"
        )
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    print(f"Randomly selected SonicMaster source: {picked_path}")
    print(f"Evaluating available degradations: {[name for name, _ in selected_pairs]}")

    pesq_by_degradation = []
    lsd_by_degradation = []
    sispnr_by_degradation = []
    sisnr_by_degradation = []
    with torch.no_grad():
        for de_name, pair_index in selected_pairs:
            degraded_wav, clean_wav, sr, path, _ = dataset.get_waveform_pair(pair_index)
            restored_wav = _restore_waveform_by_segments(net, degraded_wav, sr, opts)

            clean_wav_cpu = clean_wav.cpu()
            metrics = _compute_chunkwise_metrics(clean_wav_cpu, restored_wav, sr, opts)
            pesq_value = metrics["pesq"]
            lsd_value = metrics["lsd"]
            sispnr_value = metrics["sispnr"]
            sisnr_value = metrics["sisnr"]

            pesq_by_degradation.append((de_name, pesq_value))
            lsd_by_degradation.append((de_name, lsd_value))
            sispnr_by_degradation.append((de_name, sispnr_value))
            sisnr_by_degradation.append((de_name, sisnr_value))

            print(
                f"{de_name}: "
                f"PESQ={_format_metric(pesq_value)} "
                f"LSD={_format_metric(lsd_value)} "
                f"SiSPNR={_format_metric(sispnr_value)} "
                f"SiSNR={_format_metric(sisnr_value)}"
            )

            if out_dir is not None:
                base = os.path.splitext(os.path.basename(path))[0]
                _save_audio(os.path.join(out_dir, f"{base}_{de_name}_restored.wav"), restored_wav, sr)
                _save_audio(os.path.join(out_dir, f"{base}_{de_name}_degraded.wav"), degraded_wav, sr)
                _save_audio(os.path.join(out_dir, f"{base}_{de_name}_clean.wav"), clean_wav_cpu, sr)

    mean_pesq = _mean_or_nan([value for _, value in pesq_by_degradation])
    mean_lsd = _mean_or_nan([value for _, value in lsd_by_degradation])
    mean_sispnr = _mean_or_nan([value for _, value in sispnr_by_degradation])
    mean_sisnr = _mean_or_nan([value for _, value in sisnr_by_degradation])
    print(f"Mean PESQ over selected source: {_format_metric(mean_pesq)}")
    print(f"Mean LSD over selected source: {_format_metric(mean_lsd)}")
    print(f"Mean SiSPNR over selected source: {_format_metric(mean_sispnr)}")
    print(f"Mean SiSNR over selected source: {_format_metric(mean_sisnr)}")

    _append_metrics_csv(
        opts,
        {
            "task": "audio_single_random",
            "benchmark": "sonicmaster_single_random",
            "song_path": picked_path,
            "pesq": mean_pesq,
            "lsd": mean_lsd,
            "sispnr": mean_sispnr,
            "sisnr": mean_sisnr,
            "num_degradations": len(selected_pairs),
        },
    )

    for de_name, pesq_value in pesq_by_degradation:
        _append_metrics_csv(
            opts,
            {
                "task": "audio_single_random_per_degradation",
                "benchmark": "sonicmaster_single_random",
                "song_path": picked_path,
                "degradation": de_name,
                "pesq": pesq_value,
                "lsd": dict(lsd_by_degradation).get(de_name, float("nan")),
                "sispnr": dict(sispnr_by_degradation).get(de_name, float("nan")),
                "sisnr": dict(sisnr_by_degradation).get(de_name, float("nan")),
            },
        )


####################################################################################################
## main
def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    print("model:", getattr(opt, "model", None))
    print("dim:", getattr(opt, "dim", None))
    print("num_blocks:", getattr(opt, "num_blocks", None))
    print("num_dec_blocks:", getattr(opt, "num_dec_blocks", None))
    print("heads:", getattr(opt, "heads", None))
    print("num_refinement_blocks:", getattr(opt, "num_refinement_blocks", None))
    print("num_exp_blocks:", getattr(opt, "num_exp_blocks", None))

    net = PLTestModel(opt)
    checkpoint_path = resolve_checkpoint_path(
        ckpt_dir=opt.ckpt_dir,
        reference=opt.checkpoint_id,
        explicit_path=opt.checkpoint_path,
        selection=opt.checkpoint_select,
    )
    checkpoint_report = load_strict_model_checkpoint(
        net,
        checkpoint_path,
    )
    print(
        "Loaded {} model tensors from {}".format(
            checkpoint_report["loaded_tensor_count"],
            checkpoint_report["checkpoint_path"],
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(f"Evaluation device: {device}")
    net.eval()

    if getattr(opt, "input_audio", None) or getattr(opt, "output_audio", None):
        raise ValueError(
            "Single-file restoration is handled by inference.py; "
            "evaluate.py sonicmaster is reserved for paired SonicMaster test-set evaluation."
        )

    if not opt.degradation_metadata_jsonl:
        raise ValueError("SonicMaster test-set evaluation requires --degradation_metadata_jsonl.")

    dataset = AudioTestDataset(opt)
    print("--------> Testing on the SonicMaster test split.")
    print("\n")
    if opt.audio_single_random:
        run_audio_single_random_test(opt, net, dataset)
    else:
        run_audio_test(opts=opt, net=net, dataset=dataset)


