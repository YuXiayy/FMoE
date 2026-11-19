#!/usr/bin/env python3
"""Export the official SonicMaster Parquet release to F-MoE's paired layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq


DATASET_REPO = "amaai-lab/SonicMasterDataset"
PAIR_ID = re.compile(r"^(?P<source>.+)_(?:deg|dt)(?P<index>[1-9]\d*)$", re.IGNORECASE)
REQUIRED_METADATA = {
    "degradations",
    "degradations_specifics",
    "degradation_tracking",
    "hidden_clipping",
}


def _safe_identifier(value: Any, field: str) -> str:
    identifier = str(value).strip()
    if not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError(f"invalid {field}: {value!r}")
    return identifier


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"meta must be a JSON object, got {type(value).__name__}")
    return dict(value)


def _audio_bytes(value: Any, field: str) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        if payload is not None:
            return _audio_bytes(payload, field)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    raise TypeError(f"{field} must contain FLAC bytes or an audio mapping")


def _write_verified(path: Path, payload: bytes, overwrite: bool) -> None:
    if not payload:
        raise ValueError(f"refusing to write empty audio payload: {path}")
    if path.exists() and not overwrite:
        existing_hash = hashlib.sha256(path.read_bytes()).digest()
        payload_hash = hashlib.sha256(payload).digest()
        if existing_hash != payload_hash:
            raise FileExistsError(
                f"existing file differs from the dataset row: {path}; use --overwrite"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _infer_split(meta: Mapping[str, Any], parquet_path: Path) -> str:
    split = str(meta.get("split", "")).strip().lower()
    if not split:
        split = parquet_path.stem.split("-", 1)[0].lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported or missing dataset split: {split!r}")
    return split


def _download_parquet(destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=["*.parquet", "**/*.parquet"],
    )
    return Path(snapshot)


def export_dataset(
    parquet_root: Path,
    output_dir: Path,
    overwrite: bool,
    limit: int,
) -> int:
    parquet_files = sorted(parquet_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no Parquet files found under {parquet_root}")

    rows_by_id: dict[str, dict[str, Any]] = {}
    processed = 0
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        required_columns = {"id", "input_flac", "gt_flac", "meta"}
        missing_columns = required_columns - set(parquet.schema.names)
        if missing_columns:
            raise ValueError(
                f"{parquet_path} is missing columns: {sorted(missing_columns)}"
            )
        for batch in parquet.iter_batches(columns=sorted(required_columns), batch_size=64):
            for row in batch.to_pylist():
                row_id = _safe_identifier(row["id"], "row id")
                match = PAIR_ID.fullmatch(row_id)
                if match is None:
                    raise ValueError(
                        f"row id must end in _deg<number> or _DT<number>: {row_id!r}"
                    )
                source_id = _safe_identifier(match.group("source"), "source id")
                degradation_index = int(match.group("index"))
                canonical_id = f"{source_id}_deg{degradation_index}"
                meta = _metadata(row["meta"])
                split = _infer_split(meta, parquet_path)
                missing_metadata = REQUIRED_METADATA - set(meta)
                if missing_metadata:
                    raise ValueError(
                        f"{row_id} metadata is missing: {sorted(missing_metadata)}"
                    )

                split_dir = output_dir / "pairs" / split
                _write_verified(
                    split_dir / f"{source_id}.flac",
                    _audio_bytes(row["gt_flac"], "gt_flac"),
                    overwrite,
                )
                _write_verified(
                    split_dir / f"{canonical_id}.flac",
                    _audio_bytes(row["input_flac"], "input_flac"),
                    overwrite,
                )

                metadata_row = dict(meta)
                metadata_row.update({"id": canonical_id, "split": split})
                previous = rows_by_id.get(canonical_id)
                if previous is not None and previous != metadata_row:
                    raise ValueError(f"conflicting duplicate metadata id: {canonical_id}")
                rows_by_id[canonical_id] = metadata_row
                processed += 1
                if limit and processed >= limit:
                    break
            if limit and processed >= limit:
                break
        if limit and processed >= limit:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    temporary = metadata_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row_id in sorted(rows_by_id):
            handle.write(json.dumps(rows_by_id[row_id], ensure_ascii=False) + "\n")
    temporary.replace(metadata_path)
    return len(rows_by_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--parquet-dir",
        type=Path,
        help="Existing SonicMaster Parquet directory; skips downloading.",
    )
    source.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/sonicmaster"),
        help="Directory used for the Hugging Face dataset snapshot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test row limit; 0 exports all rows.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    parquet_root = (
        args.parquet_dir
        if args.parquet_dir is not None
        else _download_parquet(args.download_dir)
    )
    count = export_dataset(parquet_root, args.output_dir, args.overwrite, args.limit)
    print(f"Exported {count} paired rows to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
