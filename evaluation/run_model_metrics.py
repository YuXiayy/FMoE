#!/usr/bin/env python3
"""Portable FAD, KL, log-mel SSIM, and PQ evaluation on one frozen manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .run_audio_metrics import discover_models, group_of, load_manifest, prediction_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SonicMaster-compatible model-based and log-mel metrics"
    )
    parser.add_argument("metric", choices=("kl-ssim", "fad", "pq"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--clap-checkpoint", type=Path)
    parser.add_argument("--audiobox-checkpoint", type=Path)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    fields = fields or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def group_rows(rows: Iterable[dict], group: str) -> list[dict]:
    rows = list(rows)
    return rows if group == "all" else [row for row in rows if row["group"] == group]


def require_complete(rows: list[dict], model: str, allow_failures: bool) -> None:
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed and not allow_failures:
        first = failed[0]
        raise RuntimeError(
            f"{model}: {len(failed)} failed item(s); first={first.get('id')}: {first.get('error')}"
        )


def log_mel(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=44100, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=44100, n_fft=2048, hop_length=512, n_mels=128
    )
    return librosa.power_to_db(mel, ref=np.max)


def kl_value(first: np.ndarray, second: np.ndarray) -> float:
    p = np.mean(first, axis=1)
    q = np.mean(second, axis=1)
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


def ssim_value(first: np.ndarray, second: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    first = first[:height, :width]
    second = second[:height, :width]
    first = (first - first.min()) / (first.max() - first.min() + 1e-8)
    second = (second - second.min()) / (second.max() - second.min() + 1e-8)
    return float(structural_similarity(first, second, data_range=1.0))


def run_kl_ssim(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = load_manifest(root / "manifest.jsonl")
    for model, metric_dir, prediction_dir in discover_models(root, args.models):
        rows = []
        for entry in manifest:
            item_id = str(entry["id"])
            reference = Path(entry["original_location"])
            prediction = Path(entry["location"]) if model == "degraded_input" else prediction_path(prediction_dir, entry)
            common = {
                "id": item_id, "group": group_of(entry),
                "reference_path": str(reference), "prediction_path": str(prediction),
            }
            try:
                pred_mel = log_mel(prediction)
                ref_mel = log_mel(reference)
                row = {
                    **common, "kl": kl_value(pred_mel, ref_mel),
                    "ssim": ssim_value(pred_mel, ref_mel), "status": "ok", "error": "",
                }
            except Exception as exc:
                row = {**common, "kl": "", "ssim": "", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
        require_complete(rows, model, args.allow_failures)
        write_csv(metric_dir / "kl_ssim_per_sample.csv", rows)
        summary = []
        for group in ("single", "double_plus_triple", "all"):
            subset = group_rows(rows, group)
            valid = [row for row in subset if row["status"] == "ok"]
            summary.append({
                "folder": str(prediction_dir), "group": group,
                "num_pairs": len(valid), "num_missing_or_failed": len(subset) - len(valid),
                "avg_kl": float(np.mean([row["kl"] for row in valid])),
                "avg_ssim": float(np.mean([row["ssim"] for row in valid])),
            })
        write_csv(metric_dir / "kl_ssim_comparison.csv", summary)
        print(f"[KL-SSIM] {model}: {len(rows)} rows, output={metric_dir}")


def load_clap(checkpoint: Path):
    from laion_clap import CLAP_Module

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt=str(checkpoint), verbose=True)
    model.eval()
    return model


def clap_embedding(model, path: Path) -> np.ndarray:
    value = model.get_audio_embedding_from_filelist([str(path)], use_tensor=False)
    if len(value) != 1:
        raise RuntimeError(f"CLAP returned {len(value)} embeddings for {path}")
    return np.asarray(value[0], dtype=np.float64)


def frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    mu_first, mu_second = first.mean(axis=0), second.mean(axis=0)
    cov_first = np.cov(first, rowvar=False)
    cov_second = np.cov(second, rowvar=False)
    cov_mean = sqrtm(cov_first @ cov_second)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    diff = mu_first - mu_second
    return float(diff @ diff + np.trace(cov_first + cov_second - 2 * cov_mean))


def run_fad(args: argparse.Namespace) -> None:
    if args.clap_checkpoint is None:
        raise ValueError("fad requires --clap-checkpoint")
    root = args.root.resolve()
    manifest = load_manifest(root / "manifest.jsonl")
    clap = load_clap(args.clap_checkpoint.expanduser().resolve())
    clean_cache: dict[str, np.ndarray] = {}
    for entry in manifest:
        path = str(Path(entry["original_location"]))
        if path not in clean_cache:
            clean_cache[path] = clap_embedding(clap, Path(path))
    for model, metric_dir, prediction_dir in discover_models(root, args.models):
        records = []
        for entry in manifest:
            item_id = str(entry["id"])
            reference = Path(entry["original_location"])
            prediction = Path(entry["location"]) if model == "degraded_input" else prediction_path(prediction_dir, entry)
            try:
                records.append({
                    "id": item_id, "group": group_of(entry), "status": "ok", "error": "",
                    "clean": clean_cache[str(reference)], "prediction": clap_embedding(clap, prediction),
                })
            except Exception as exc:
                records.append({
                    "id": item_id, "group": group_of(entry), "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        require_complete(records, model, args.allow_failures)
        summary = []
        for group in ("single", "double_plus_triple", "all"):
            valid = [row for row in group_rows(records, group) if row["status"] == "ok"]
            if len(valid) < 2:
                raise RuntimeError(f"{model}/{group}: FAD requires at least two valid pairs")
            clean = np.stack([row["clean"] for row in valid])
            prediction = np.stack([row["prediction"] for row in valid])
            summary.append({
                "folder": str(prediction_dir), "group": group,
                "num_pairs": len(valid), "fad": frechet_distance(clean, prediction),
            })
        write_csv(metric_dir / "fad_comparison.csv", summary)
        print(f"[FAD] {model}: {len(records)} rows, output={metric_dir}")


def load_pq_predictor(checkpoint: Path):
    from audiobox_aesthetics.infer import initialize_predictor

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return initialize_predictor(ckpt=str(checkpoint))


def run_pq(args: argparse.Namespace) -> None:
    if args.audiobox_checkpoint is None:
        raise ValueError("pq requires --audiobox-checkpoint")
    root = args.root.resolve()
    manifest = load_manifest(root / "manifest.jsonl")
    predictor = load_pq_predictor(args.audiobox_checkpoint.expanduser().resolve())
    for model, metric_dir, prediction_dir in discover_models(root, args.models):
        rows = []
        axes: set[str] = set()
        for entry in manifest:
            item_id = str(entry["id"])
            prediction = Path(entry["location"]) if model == "degraded_input" else prediction_path(prediction_dir, entry)
            common = {"id": item_id, "group": group_of(entry), "prediction_path": str(prediction)}
            try:
                scores = predictor.forward([{"path": str(prediction)}])[0]
                axes.update(scores)
                rows.append({**common, **{key: float(value) for key, value in scores.items()}, "status": "ok", "error": ""})
            except Exception as exc:
                rows.append({**common, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        require_complete(rows, model, args.allow_failures)
        if "PQ" not in axes:
            raise RuntimeError(f"{model}: Audiobox output does not contain PQ; axes={sorted(axes)}")
        fields = ["id", "group", "prediction_path", *sorted(axes), "status", "error"]
        write_csv(metric_dir / "pq_per_sample.csv", rows, fields)
        summary = []
        for group in ("single", "double_plus_triple", "all"):
            valid = [row for row in group_rows(rows, group) if row["status"] == "ok"]
            summary.append({
                "folder": str(prediction_dir), "group": group, "pq_key": "PQ",
                "avg_pq": float(np.mean([row["PQ"] for row in valid])),
            })
        write_csv(metric_dir / "pq_comparison.csv", summary)
        print(f"[PQ] {model}: {len(rows)} rows, output={metric_dir}")


def main(argv=None) -> None:
    args = parse_args(argv)
    runners: dict[str, Callable[[argparse.Namespace], None]] = {
        "kl-ssim": run_kl_ssim,
        "fad": run_fad,
        "pq": run_pq,
    }
    runners[args.metric](args)


