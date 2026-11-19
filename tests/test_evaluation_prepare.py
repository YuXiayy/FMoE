import json
from pathlib import Path

from evaluation.prepare_protocol import find_prediction, normalize_audio_path, read_jsonl


def test_manifest_and_prediction_resolution(tmp_path: Path):
    audio = tmp_path / "audio"
    predictions = tmp_path / "predictions"
    audio.mkdir()
    predictions.mkdir()
    (audio / "clean.wav").touch()
    (audio / "degraded.wav").touch()
    (predictions / "item.wav").touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "id": "item",
            "original_location": "audio/clean.wav",
            "location": "audio/degraded.wav",
            "degradations": ["compression"],
        }) + "\n",
        encoding="utf-8",
    )

    rows = read_jsonl(manifest)
    assert normalize_audio_path(rows[0], "original_location", tmp_path) == str((audio / "clean.wav").resolve())
    assert normalize_audio_path(rows[0], "location", tmp_path) == str((audio / "degraded.wav").resolve())
    assert find_prediction(predictions, "item") == predictions / "item.wav"


def test_duplicate_ids_are_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    row = {"id": "same"}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    try:
        read_jsonl(manifest)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate IDs must be rejected")
