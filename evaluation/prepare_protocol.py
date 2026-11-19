#!/usr/bin/env python3
"""Prepare a portable evaluation run without copying the audio files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATH_KEYS = {
    "original_location": ("original_location", "reference_path", "clean_path"),
    "location": ("location", "degraded_path", "input_path"),
}
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the directory layout consumed by run_audio_metrics.py"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", default=[], metavar="NAME=DIR",
        help="Prediction directory; repeat for multiple systems. Files must be ID.flac or ID.wav.",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest IDs must be non-empty and unique")
    return rows


def normalize_audio_path(row: dict, canonical: str, base: Path) -> str:
    source_key = next((key for key in PATH_KEYS[canonical] if row.get(key)), None)
    if source_key is None:
        raise ValueError(f"Entry {row.get('id')} lacks one of {PATH_KEYS[canonical]}")
    path = Path(str(row[source_key])).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Entry {row.get('id')}: audio file not found: {path}")
    return str(path)


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--prediction must be NAME=DIR, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not MODEL_NAME.fullmatch(name):
        raise ValueError(f"Invalid model name {name!r}")
    directory = Path(raw_path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return name, directory


def find_prediction(directory: Path, item_id: str) -> Path:
    found = [directory / f"{item_id}{suffix}" for suffix in (".flac", ".wav")]
    found = [path for path in found if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one prediction for {item_id} in {directory} "
            f"(ID.flac or ID.wav); found {len(found)}"
        )
    return found[0]


def write_audit(root: Path, name: str, prediction_dir: Path, count: int) -> None:
    metric_dir = root / "metrics" / name
    metric_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "manifest": str((root / "manifest.jsonl").resolve()),
        "prediction_dir": str(prediction_dir),
        "num_predictions": count,
    }
    (metric_dir / "prediction_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    root = args.root.expanduser().resolve()
    rows = read_jsonl(manifest_path)
    normalized = []
    for row in rows:
        item = dict(row)
        item["id"] = str(item["id"])
        item["original_location"] = normalize_audio_path(item, "original_location", manifest_path.parent)
        item["location"] = normalize_audio_path(item, "location", manifest_path.parent)
        degradations = item.get("degradations", [])
        if not isinstance(degradations, list) or not degradations:
            raise ValueError(f"Entry {item['id']}: degradations must be a non-empty list")
        normalized.append(item)

    predictions = [parse_prediction(value) for value in args.prediction]
    names = [name for name, _ in predictions]
    if len(names) != len(set(names)) or "degraded_input" in names:
        raise ValueError("Prediction names must be unique and cannot be 'degraded_input'")
    for name, directory in predictions:
        for row in normalized:
            find_prediction(directory, row["id"])

    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    degraded_parent = Path(normalized[0]["location"]).parent
    write_audit(root, "degraded_input", degraded_parent, len(normalized))
    for name, directory in predictions:
        write_audit(root, name, directory, len(normalized))
    print(f"Prepared {len(normalized)} pairs and {len(predictions)} model(s) under {root}")


