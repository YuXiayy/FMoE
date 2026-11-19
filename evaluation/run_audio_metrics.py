#!/usr/bin/env python3
"""Reproducible additional metrics for the frozen SonicMaster 30 s protocol.

Signal metrics:
  * Multi-Mel-SNR: exact MSRKit/MSR Challenge definition.
  * LSD: AudioSR repository definition, including lower-band energy scaling.
  * SI-SDRi: nussl scale_bss_eval definition (Le Roux et al.).

ViSQOL:
  * Official Google ViSQOL Python API, audio mode at 48 kHz.
  * Each 30 s item is evaluated as six 5 s chunks, then averaged per item.

Outputs are resumable.  A JSONL journal is written first and atomically converted
to per-sample and group-summary CSV files when a model completes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = Path("evaluation_runs/default")
SIGNAL_PROGRESS = ".additional_signal_metrics.progress.jsonl"
SIGNAL_DETAIL = "additional_signal_metrics_per_sample.csv"
SIGNAL_SUMMARY = "additional_signal_metrics_comparison.csv"
VISQOL_PROGRESS = ".visqol_audio_5s.progress.jsonl"
VISQOL_DETAIL = "visqol_audio_5s_per_sample.csv"
VISQOL_SUMMARY = "visqol_audio_5s_comparison.csv"
COMBINED_SUMMARY = "additional_metrics_comparison.csv"
PROTOCOL_FILE = "additional_metrics_protocol.json"

SIGNAL_FIELDS = [
    "id", "group", "reference_path", "degraded_path", "prediction_path",
    "reference_channels", "prediction_channels", "channel_adapter",
    "num_samples_44100", "multi_mel_snr_db", "lsd_db", "si_sdr_db",
    "degraded_si_sdr_db", "si_sdri_db", "status", "error",
]
VISQOL_FIELDS = [
    "id", "group", "reference_path", "prediction_path", "num_chunks",
    "visqol_audio_mos_5s_mean", "visqol_audio_mos_5s_std", "chunk_scores",
    "status", "error",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-platform SonicMaster-compatible MM-SNR/LSD/SI-SDRi/ViSQOL evaluation"
    )
    p.add_argument("mode", choices=("signal", "visqol", "merge", "validate", "self-test"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--models", nargs="*", default=None,
                   help="Metric directory names. Default: all audited directories.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 means all.")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    ids = [x["id"] for x in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Manifest contains duplicate ids")
    return rows


def group_of(entry: Mapping[str, Any]) -> str:
    return "single" if len(entry.get("degradations", [])) == 1 else "double_plus_triple"


def discover_models(root: Path, requested: Optional[Sequence[str]]) -> List[Tuple[str, Path, Path]]:
    metrics_root = root / "metrics"
    names = list(requested) if requested else sorted(p.name for p in metrics_root.iterdir() if p.is_dir())
    if "degraded_input" in names:
        names.remove("degraded_input")
        names.insert(0, "degraded_input")
    found = []
    manifest_resolved = (root / "manifest.jsonl").resolve()
    expected_count = len(load_manifest(manifest_resolved))
    for name in names:
        metric_dir = metrics_root / name
        audit_path = metric_dir / "prediction_audit.json"
        if not audit_path.exists():
            raise FileNotFoundError(f"Missing audit: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if Path(audit["manifest"]).resolve() != manifest_resolved:
            raise RuntimeError(f"{name}: audit points to a different manifest")
        if int(audit["num_predictions"]) != expected_count:
            raise RuntimeError(
                f"{name}: audit count {audit['num_predictions']} does not match "
                f"manifest count {expected_count}"
            )
        prediction_dir = Path(audit["prediction_dir"])
        found.append((name, metric_dir, prediction_dir))
    return found


def prediction_path(prediction_dir: Path, entry: Mapping[str, Any]) -> Path:
    candidates = [prediction_dir / f"{entry['id']}{suffix}" for suffix in (".flac", ".wav")]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f"Ambiguous prediction formats for {entry['id']}: {existing}")
    return candidates[0]


def append_jsonl(path: Path, row: Mapping[str, Any], handle: Any, counter: int, flush_every: int) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if counter % flush_every == 0:
        handle.flush()
        os.fsync(handle.fileno())


def read_progress(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                rows[str(row["id"])] = row
            except Exception:
                print(f"[WARN] Ignoring incomplete journal line {path}:{line_no}", flush=True)
    return rows


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def finite_mean(values: Iterable[Any]) -> float:
    vals = np.asarray([float(x) for x in values if x not in (None, "")], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    vals = np.asarray([float(x) for x in values if x not in (None, "")], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.std()) if vals.size else float("nan")


def summarize_signal(model: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for group in ("single", "double_plus_triple", "all"):
        subset = list(rows) if group == "all" else [r for r in rows if r["group"] == group]
        valid = [r for r in subset if r.get("status") == "ok"]
        out.append({
            "model": model, "group": group, "num_pairs": len(valid),
            "num_failed": len(subset) - len(valid),
            "avg_multi_mel_snr_db": finite_mean(r.get("multi_mel_snr_db") for r in valid),
            "avg_lsd_db": finite_mean(r.get("lsd_db") for r in valid),
            "avg_si_sdr_db": finite_mean(r.get("si_sdr_db") for r in valid),
            "avg_degraded_si_sdr_db": finite_mean(r.get("degraded_si_sdr_db") for r in valid),
            "avg_si_sdri_db": finite_mean(r.get("si_sdri_db") for r in valid),
        })
    return out


def summarize_visqol(model: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for group in ("single", "double_plus_triple", "all"):
        subset = list(rows) if group == "all" else [r for r in rows if r["group"] == group]
        valid = [r for r in subset if r.get("status") == "ok"]
        out.append({
            "model": model, "group": group, "num_pairs": len(valid),
            "num_failed": len(subset) - len(valid),
            "avg_visqol_audio_mos_5s": finite_mean(r.get("visqol_audio_mos_5s_mean") for r in valid),
            "std_across_samples": finite_std(r.get("visqol_audio_mos_5s_mean") for r in valid),
        })
    return out


class SignalMetricEngine:
    def __init__(self, device_name: str):
        import torch
        import torchaudio

        self.torch = torch
        self.torchaudio = torchaudio
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = torch.device(device_name)
        self.analysis_sr = 48000
        self.dataset_sr = 44100
        self.resample = torchaudio.transforms.Resample(self.dataset_sr, self.analysis_sr).to(self.device)
        self.mels = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.analysis_sr, n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, f_min=0, f_max=24000, power=2.0,
            ).to(self.device)
            for n_fft, hop, n_mels in ((512, 256, 80), (1024, 512, 128), (2048, 1024, 192))
        ]
        self.lsd_n_fft = 2048
        self.lsd_hop = 512
        # AudioSR's public LSD demonstration uses EPS=1e-8 and scales the
        # estimate from the lowest 10% of magnitude bins before scoring.
        # Source: example/lsd_calculation_pitfall/demo_comparison.py
        self.lsd_eps = 1e-8
        self.lsd_lower_percentile = 10
        self.window = torch.hann_window(self.lsd_n_fft, device=self.device)
        self._cache_key: Optional[str] = None
        self._cache: Dict[str, Any] = {}

    def load_audio(self, path: Path) -> Tuple[Any, int]:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = self.torch.from_numpy(np.ascontiguousarray(wav.T)).to(self.device)
        if int(sr) != self.dataset_sr:
            x = self.torchaudio.functional.resample(x, int(sr), self.dataset_sr)
        return x, int(sr)

    def adapt_channels(self, prediction: Any, reference_channels: int) -> Tuple[Any, str]:
        pred_channels = int(prediction.shape[0])
        if pred_channels == reference_channels:
            return prediction, "none"
        if pred_channels == 1 and reference_channels > 1:
            return prediction.repeat(reference_channels, 1), f"repeat_mono_to_{reference_channels}ch"
        if reference_channels == 1 and pred_channels > 1:
            return prediction.mean(dim=0, keepdim=True), f"mean_{pred_channels}ch_to_mono"
        raise ValueError(f"Unsupported channel mapping {pred_channels}->{reference_channels}")

    def magnitude_spectrogram(self, x: Any) -> Any:
        spec = self.torch.stft(
            x, n_fft=self.lsd_n_fft, hop_length=self.lsd_hop,
            window=self.window, center=True, pad_mode="reflect", return_complex=True,
        )
        return spec.abs()

    def reference(self, path: Path) -> Dict[str, Any]:
        key = str(path)
        if key == self._cache_key:
            return self._cache
        ref44, source_sr = self.load_audio(path)
        ref48 = self.resample(ref44)
        cache = {
            "wave44": ref44,
            "wave48": ref48,
            "source_sr": source_sr,
            "mels": [m(ref48) for m in self.mels],
            "magnitude": self.magnitude_spectrogram(ref48),
        }
        self._cache_key = key
        self._cache = cache
        return cache

    def si_sdr(self, estimate: Any, reference: Any) -> float:
        # Faithful batched/torch port of nussl.evaluation.bss_eval's
        # _scale_bss_eval SI-SDR branch. BSSEvalScale zero-centers inputs
        # before calling the kernel; we do that explicitly per channel here.
        est = estimate.double() - estimate.double().mean(dim=-1, keepdim=True)
        ref = reference.double() - reference.double().mean(dim=-1, keepdim=True)
        ref_energy = ref.square().sum(dim=-1)
        if self.torch.any(ref_energy <= 1e-12):
            raise ValueError("Near-silent reference channel")
        alpha = (est * ref).sum(dim=-1) / ref_energy
        target = alpha[:, None] * ref
        noise = est - target
        eps = self.torch.finfo(self.torch.float64).eps
        value = 10.0 * self.torch.log10(
            target.square().sum(dim=-1).clamp_min(eps)
            / noise.square().sum(dim=-1).clamp_min(eps)
        )
        return float(value.mean().item())

    def multi_mel_snr(self, pred48: Any, ref: Mapping[str, Any]) -> float:
        ref48 = ref["wave48"]
        alpha = (ref48 * pred48).sum(dim=-1) / (pred48.square().sum(dim=-1) + 1e-8)
        scaled = alpha[:, None] * pred48
        values = []
        for mel, ref_mel in zip(self.mels, ref["mels"]):
            pred_mel = mel(scaled)
            numerator = ref_mel.square().sum(dim=(-2, -1))
            denominator = (ref_mel - pred_mel).square().sum(dim=(-2, -1)) + 1e-8
            values.append(10.0 * self.torch.log10(numerator / denominator))
        return float(self.torch.stack(values).mean().item())

    def lsd(self, pred48: Any, ref: Mapping[str, Any]) -> float:
        # Faithful batched/torch port of AudioSR's official public function:
        #   lsd(gt_magnitude, scaled_prediction_magnitude)
        # AudioSR intentionally has no leading factor of 10 in this formula.
        ref_mag = ref["magnitude"]
        pred_mag = self.magnitude_spectrogram(pred48)
        lower_bins = int(ref_mag.shape[-2] * self.lsd_lower_percentile / 100)
        if lower_bins <= 0:
            raise ValueError("AudioSR LSD lower-frequency band is empty")
        ref_low = ref_mag[..., :lower_bins, :].mean(dim=(-2, -1))
        pred_low = pred_mag[..., :lower_bins, :].mean(dim=(-2, -1))
        scale = self.torch.where(
            pred_low > 0, ref_low / pred_low, self.torch.ones_like(pred_low)
        )
        pred_mag = pred_mag * scale[:, None, None]
        ratio = pred_mag.square() / (ref_mag + self.lsd_eps).square()
        lsd_squared = self.torch.log10(ratio + self.lsd_eps).square()
        frame_lsd = lsd_squared.mean(dim=-2).sqrt()
        return float(frame_lsd.mean().item())

    def compute(self, reference_path: Path, prediction_path_: Path,
                baseline_si_sdr: Optional[float]) -> Dict[str, Any]:
        ref = self.reference(reference_path)
        pred44_raw, _ = self.load_audio(prediction_path_)
        raw_channels = int(pred44_raw.shape[0])
        pred44, adapter = self.adapt_channels(pred44_raw, int(ref["wave44"].shape[0]))
        n = min(int(ref["wave44"].shape[-1]), int(pred44.shape[-1]))
        if n <= self.lsd_n_fft:
            raise ValueError("Audio is too short")
        if n != int(ref["wave44"].shape[-1]):
            raise ValueError("Prediction/reference length mismatch violates frozen audit")
        pred44 = pred44[..., :n]
        pred48 = self.resample(pred44)
        n48 = min(int(ref["wave48"].shape[-1]), int(pred48.shape[-1]))
        pred48 = pred48[..., :n48]
        if n48 != int(ref["wave48"].shape[-1]):
            raise ValueError("Resampled prediction/reference length mismatch")
        pred_si = self.si_sdr(pred44, ref["wave44"])
        return {
            "reference_channels": int(ref["wave44"].shape[0]),
            "prediction_channels": raw_channels,
            "channel_adapter": adapter,
            "num_samples_44100": n,
            "multi_mel_snr_db": self.multi_mel_snr(pred48, ref),
            "lsd_db": self.lsd(pred48, ref),
            "si_sdr_db": pred_si,
            "degraded_si_sdr_db": pred_si if baseline_si_sdr is None else float(baseline_si_sdr),
            "si_sdri_db": 0.0 if baseline_si_sdr is None else pred_si - float(baseline_si_sdr),
        }


def write_protocol(metric_dir: Path, root: Path, mode: str) -> None:
    protocol_path = metric_dir / PROTOCOL_FILE
    old: Dict[str, Any] = {}
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
    old.update({
        "protocol_name": (
            "SonicMaster frozen paper protocol 30s/7000"
            if len(load_manifest(root / "manifest.jsonl")) == 7000
            else "SonicMaster-compatible evaluation subset"
        ),
        "manifest": str(root / "manifest.jsonl"),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "updated_unix": time.time(),
        "last_mode": mode,
        "multi_mel_snr": {
            "source": "https://github.com/yongyizang/MSRKit/blob/main/calculate_metrics.py",
            "sample_rate": 48000,
            "configs": [[512, 256, 80], [1024, 512, 128], [2048, 1024, 192]],
            "power": 2.0, "channel_reduction": "mean",
            "note": "44.1 kHz dataset audio is correctly resampled to the official 48 kHz domain",
        },
        "lsd": {
            "source_repository": "https://github.com/haoheliu/versatile_audio_super_resolution",
            "source_file": "example/lsd_calculation_pitfall/demo_comparison.py",
            "paper": "AudioSR: Versatile Audio Super-resolution at Scale, ICASSP 2024",
            "definition": "AudioSR lsd(ref_magnitude, lower-band-energy-scaled prediction_magnitude)",
            "sample_rate": 48000, "n_fft": 2048, "hop_length": 512,
            "window": "periodic Hann", "center": True, "eps": 1e-8,
            "lower_frequency_percentile_for_energy_scaling": 10,
            "numeric_convention": "log10 power ratio exactly as repository; no leading factor 10",
            "implementation": "faithful vectorized torch port for GPU throughput",
        },
        "si_sdri": {
            "source_repository": "https://github.com/nussl/nussl/blob/master/nussl/evaluation/bss_eval.py",
            "toolkit_paper": "The Northwestern University Source Separation Library, ISMIR 2018",
            "metric_paper": "SDR -- half-baked or well done?, ICASSP 2019",
            "definition": "SI-SDR(prediction, clean)-SI-SDR(degraded, clean)",
            "sample_rate": 44100, "zero_mean": True, "delay_search": False,
            "implementation": "faithful vectorized torch port of nussl scale_bss_eval",
        },
        "channels": {
            "reduction": "per-channel score then mean",
            "mono_prediction_for_stereo_reference": "repeat mono into both reference channels",
        },
        "visqol": {
            "source": "https://github.com/google/visqol",
            "mode": "audio", "sample_rate": 48000, "chunk_seconds": 5,
            "sample_reduction": "mean of six contiguous chunks",
            "channel_handling": "arithmetic mono mixdown before official API",
        },
    })
    tmp = protocol_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(old, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, protocol_path)


def finalize_signal(model: str, metric_dir: Path, manifest: Sequence[Mapping[str, Any]],
                    rows_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [rows_by_id[str(e["id"])] for e in manifest if str(e["id"]) in rows_by_id]
    atomic_csv(metric_dir / SIGNAL_DETAIL, rows, SIGNAL_FIELDS)
    summary = summarize_signal(model, rows)
    atomic_csv(metric_dir / SIGNAL_SUMMARY, summary, list(summary[0].keys()))


def run_signal(args: argparse.Namespace) -> None:
    import torch

    root = args.root.resolve()
    manifest = load_manifest(root / "manifest.jsonl")
    selected = manifest[:args.limit] if args.limit else manifest
    models = discover_models(root, args.models)
    if models[0][0] != "degraded_input":
        baseline_detail = root / "metrics" / "degraded_input" / SIGNAL_DETAIL
        if not baseline_detail.exists():
            raise RuntimeError("Run degraded_input first so SI-SDRi has the frozen baseline")
    torch.set_grad_enabled(False)
    engine = SignalMetricEngine(args.device)
    baseline: Dict[str, float] = {}
    baseline_detail = root / "metrics" / "degraded_input" / SIGNAL_DETAIL
    if baseline_detail.exists():
        with baseline_detail.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    baseline[row["id"]] = float(row["si_sdr_db"])

    for model, metric_dir, pred_dir in models:
        progress_path = metric_dir / SIGNAL_PROGRESS
        if args.force:
            progress_path.unlink(missing_ok=True)
        done = read_progress(progress_path)
        print(f"[SIGNAL] {model}: resume={len(done)}/{len(selected)}", flush=True)
        with progress_path.open("a", encoding="utf-8", buffering=1) as journal:
            completed_now = 0
            for index, entry in enumerate(selected, 1):
                item_id = str(entry["id"])
                if item_id in done and done[item_id].get("status") == "ok":
                    continue
                ref_path = Path(entry["original_location"])
                deg_path = Path(entry["location"])
                pred_path = deg_path if model == "degraded_input" else prediction_path(pred_dir, entry)
                base = None if model == "degraded_input" else baseline.get(item_id)
                common = {
                    "id": item_id, "group": group_of(entry),
                    "reference_path": str(ref_path), "degraded_path": str(deg_path),
                    "prediction_path": str(pred_path),
                }
                try:
                    if not ref_path.exists() or not deg_path.exists() or not pred_path.exists():
                        raise FileNotFoundError("reference, degraded, or prediction file is missing")
                    if model != "degraded_input" and base is None:
                        raise RuntimeError("Missing degraded SI-SDR baseline for this id")
                    values = engine.compute(ref_path, pred_path, base)
                    row = {**common, **values, "status": "ok", "error": ""}
                except Exception as exc:
                    row = {**common, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                done[item_id] = row
                completed_now += 1
                append_jsonl(progress_path, row, journal, completed_now, args.flush_every)
                if model == "degraded_input" and row["status"] == "ok":
                    baseline[item_id] = float(row["si_sdr_db"])
                if index % 100 == 0:
                    print(f"[SIGNAL] {model}: {index}/{len(selected)}", flush=True)
        finalize_signal(model, metric_dir, selected, done)
        if model == "degraded_input":
            # Also cover successful rows restored from the resume journal.
            baseline = {
                item_id: float(row["si_sdr_db"])
                for item_id, row in done.items() if row.get("status") == "ok"
            }
        write_protocol(metric_dir, root, "signal")
        print(f"[SIGNAL] {model}: finalized {SIGNAL_SUMMARY}", flush=True)


_VISQOL_API: Any = None


def init_visqol_worker() -> None:
    global _VISQOL_API
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    # Required by the official example even though it is not referenced directly:
    # importing it registers SimilarityResultMsg with protobuf before pybind returns it.
    from visqol.pb2 import similarity_result_pb2  # noqa: F401

    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 48000
    config.options.use_speech_scoring = False
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", "libsvm_nu_svr_model.txt"
    )
    api = visqol_lib_py.VisqolApi()
    status = api.Create(config)
    if hasattr(status, "ok") and not status.ok():
        raise RuntimeError(f"ViSQOL API creation failed: {status}")
    _VISQOL_API = api


def load_mono_48k(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = np.mean(wav, axis=1)
    if int(sr) != 48000:
        g = math.gcd(int(sr), 48000)
        mono = resample_poly(mono, 48000 // g, int(sr) // g)
    return np.ascontiguousarray(mono, dtype=np.float64)


def visqol_task(task: Tuple[str, str, str, str]) -> Dict[str, Any]:
    item_id, group, ref_path, pred_path = task
    common = {
        "id": item_id, "group": group, "reference_path": ref_path,
        "prediction_path": pred_path,
    }
    try:
        ref = load_mono_48k(ref_path)
        pred = load_mono_48k(pred_path)
        n = min(len(ref), len(pred))
        chunk = 5 * 48000
        n_chunks = n // chunk
        if n_chunks < 1:
            raise ValueError("Less than one complete 5 s chunk")
        scores = []
        for i in range(n_chunks):
            lo, hi = i * chunk, (i + 1) * chunk
            result = _VISQOL_API.Measure(ref[lo:hi], pred[lo:hi])
            score = float(result.moslqo)
            if not np.isfinite(score):
                raise ValueError(f"Non-finite ViSQOL score at chunk {i}")
            scores.append(score)
        return {
            **common, "num_chunks": len(scores),
            "visqol_audio_mos_5s_mean": float(np.mean(scores)),
            "visqol_audio_mos_5s_std": float(np.std(scores)),
            "chunk_scores": json.dumps(scores, separators=(",", ":")),
            "status": "ok", "error": "",
        }
    except Exception as exc:
        return {**common, "num_chunks": 0, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}


def finalize_visqol(model: str, metric_dir: Path, manifest: Sequence[Mapping[str, Any]],
                    rows_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [rows_by_id[str(e["id"])] for e in manifest if str(e["id"]) in rows_by_id]
    atomic_csv(metric_dir / VISQOL_DETAIL, rows, VISQOL_FIELDS)
    summary = summarize_visqol(model, rows)
    atomic_csv(metric_dir / VISQOL_SUMMARY, summary, list(summary[0].keys()))


def run_visqol(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.workers > 24:
        raise ValueError("--workers must be in [1, 24]")
    root = args.root.resolve()
    manifest = load_manifest(root / "manifest.jsonl")
    selected = manifest[:args.limit] if args.limit else manifest
    models = discover_models(root, args.models)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_visqol_worker) as pool:
        for model, metric_dir, pred_dir in models:
            progress_path = metric_dir / VISQOL_PROGRESS
            if args.force:
                progress_path.unlink(missing_ok=True)
            done = read_progress(progress_path)
            tasks = [
                (str(e["id"]), group_of(e), str(e["original_location"]),
                 str(Path(e["location"]) if model == "degraded_input" else prediction_path(pred_dir, e)))
                for e in selected
                if str(e["id"]) not in done or done[str(e["id"])].get("status") != "ok"
            ]
            print(f"[VISQOL] {model}: resume={len(done)}/{len(selected)}, workers={args.workers}", flush=True)
            with progress_path.open("a", encoding="utf-8", buffering=1) as journal:
                for index, row in enumerate(pool.imap_unordered(visqol_task, tasks, chunksize=1), 1):
                    done[str(row["id"])] = row
                    append_jsonl(progress_path, row, journal, index, args.flush_every)
                    if index % 100 == 0:
                        print(f"[VISQOL] {model}: new={index}/{len(tasks)}", flush=True)
            finalize_visqol(model, metric_dir, selected, done)
            write_protocol(metric_dir, root, "visqol")
            print(f"[VISQOL] {model}: finalized {VISQOL_SUMMARY}", flush=True)


def run_merge(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    for model, metric_dir, _ in discover_models(root, args.models):
        signal_path = metric_dir / SIGNAL_SUMMARY
        visqol_path = metric_dir / VISQOL_SUMMARY
        if not signal_path.exists() or not visqol_path.exists():
            print(f"[MERGE] {model}: waiting for both summaries", flush=True)
            continue
        with signal_path.open("r", encoding="utf-8") as f:
            signal = {r["group"]: r for r in csv.DictReader(f)}
        with visqol_path.open("r", encoding="utf-8") as f:
            visqol = {r["group"]: r for r in csv.DictReader(f)}
        rows = []
        for group in ("single", "double_plus_triple", "all"):
            s, v = signal[group], visqol[group]
            rows.append({
                "model": model, "group": group,
                "num_pairs": s["num_pairs"], "num_failed": s["num_failed"],
                "multi_mel_snr_db": s["avg_multi_mel_snr_db"],
                "lsd_db": s["avg_lsd_db"], "si_sdr_db": s["avg_si_sdr_db"],
                "si_sdri_db": s["avg_si_sdri_db"],
                "visqol_audio_mos_5s": v["avg_visqol_audio_mos_5s"],
                "visqol_num_pairs": v["num_pairs"], "visqol_num_failed": v["num_failed"],
            })
        atomic_csv(metric_dir / COMBINED_SUMMARY, rows, list(rows[0].keys()))
        print(f"[MERGE] {model}: wrote {COMBINED_SUMMARY}", flush=True)


def run_validate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = load_manifest(root / "manifest.jsonl")
    expected_ids = [str(e["id"]) for e in manifest]
    expected_set = set(expected_ids)
    report: Dict[str, Any] = {
        "manifest": str(root / "manifest.jsonl"),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "expected_samples": len(manifest),
        "models": {},
        "ok": True,
    }
    errors: List[str] = []
    for model, metric_dir, _ in discover_models(root, args.models):
        model_report: Dict[str, Any] = {}
        for kind, filename in (("signal", SIGNAL_DETAIL), ("visqol", VISQOL_DETAIL)):
            path = metric_dir / filename
            if not path.exists():
                errors.append(f"{model}: missing {filename}")
                continue
            with path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ids = [r["id"] for r in rows]
            failed = [r for r in rows if r.get("status") != "ok"]
            item_report: Dict[str, Any] = {
                "rows": len(rows), "unique_ids": len(set(ids)), "failed": len(failed),
                "sha256": sha256_file(path),
            }
            if kind == "visqol":
                bad_chunks = sum(int(r.get("num_chunks", 0)) != 6 for r in rows if r.get("status") == "ok")
                item_report["bad_chunk_count"] = bad_chunks
                if bad_chunks:
                    errors.append(f"{model}: {bad_chunks} ViSQOL rows do not have six chunks")
            if len(rows) != len(manifest) or set(ids) != expected_set or len(ids) != len(set(ids)):
                errors.append(f"{model}: {kind} id/count mismatch")
            if failed:
                errors.append(f"{model}: {kind} has {len(failed)} failed rows")
            model_report[kind] = item_report
        combined = metric_dir / COMBINED_SUMMARY
        if not combined.exists():
            errors.append(f"{model}: missing {COMBINED_SUMMARY}")
        else:
            with combined.open("r", encoding="utf-8") as f:
                combined_rows = list(csv.DictReader(f))
            model_report["combined_groups"] = [r["group"] for r in combined_rows]
            model_report["combined_sha256"] = sha256_file(combined)
            if model_report["combined_groups"] != ["single", "double_plus_triple", "all"]:
                errors.append(f"{model}: combined summary groups are invalid")
        report["models"][model] = model_report
    report["errors"] = errors
    report["ok"] = not errors
    report["validated_unix"] = time.time()
    out_path = root / "additional_metrics_validation.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    print(json.dumps({"ok": report["ok"], "errors": errors, "report": str(out_path)}, indent=2))
    if errors:
        raise RuntimeError(f"Additional-metrics validation failed with {len(errors)} error(s)")


def self_test(args: argparse.Namespace) -> None:
    import soundfile as sf
    import tempfile
    import torch

    engine = SignalMetricEngine(args.device)
    rng = np.random.default_rng(1234)
    t = np.arange(44100 * 6, dtype=np.float32) / 44100.0
    clean = (0.25 * np.sin(2 * np.pi * 440 * t)
             + 0.15 * np.sin(2 * np.pi * 1703 * t)).astype(np.float32)
    noisy = clean + 0.02 * rng.standard_normal(clean.shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        clean_path, noisy_path = Path(td) / "clean.wav", Path(td) / "noisy.wav"
        sf.write(clean_path, clean, 44100, subtype="FLOAT")
        sf.write(noisy_path, noisy, 44100, subtype="FLOAT")
        identical = engine.compute(clean_path, clean_path, None)
        degraded = engine.compute(clean_path, noisy_path, None)
        # Resampling/STFT backends can introduce sub-millidecibel numerical
        # differences across operating systems even for identical inputs.
        assert identical["lsd_db"] < 2e-3, identical
        assert identical["multi_mel_snr_db"] > degraded["multi_mel_snr_db"], (identical, degraded)
        assert degraded["lsd_db"] > identical["lsd_db"], (identical, degraded)
        print(json.dumps({"signal_identical": identical, "signal_noisy": degraded}, indent=2))
        try:
            init_visqol_worker()
            result = visqol_task(("self-test", "single", str(clean_path), str(noisy_path)))
            assert result["status"] == "ok", result
            print(json.dumps({"visqol_noisy": result}, indent=2))
        except ModuleNotFoundError:
            print("[SELF-TEST] ViSQOL is not installed; signal tests passed.")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "signal":
        run_signal(args)
    elif args.mode == "visqol":
        run_visqol(args)
    elif args.mode == "merge":
        run_merge(args)
    elif args.mode == "validate":
        run_validate(args)
    else:
        self_test(args)


