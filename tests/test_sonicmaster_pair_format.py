"""SonicMaster official/legacy audio-pair naming regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.audio_dataset import AudioTrainDataset
from data.data_preflight import audit_sonicmaster_data
from data.degradation_metadata import pair_metadata_key


def _metadata_rows():
    return [
        {
            "id": "track_001_deg1",
            "split": "train",
            "degradations": ["EQ"],
            "degradations_specifics": ["bright"],
            "degradation_tracking": {"EQ": ["bright", [6]]},
            "hidden_clipping": [False, 0],
        },
        {
            "id": "track_001_deg2",
            "split": "train",
            "degradations": ["Dynamics"],
            "degradations_specifics": ["comp"],
            "degradation_tracking": {"Dynamics": ["comp", [2, 4, 5]]},
            "hidden_clipping": [False, 0],
        },
    ]


def _write_metadata(path: Path):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _metadata_rows()),
        encoding="utf-8",
    )


def _dataset_args(root: Path, metadata_path: Path):
    return SimpleNamespace(
        audio_data_dir=str(root),
        audio_ext=".flac",
        audio_use_predegraded=True,
        audio_clean_dt="dt0",
        audio_excluded_pair_ids=[],
        audio_pair_length_tolerance_samples=0,
        degradation_metadata_jsonl=str(metadata_path),
        degradation_ignored_classes=[],
        degradation_ignore_hidden_clipping_labels=False,
        de_type=["dt1", "dt2"],
        representation="complex",
        stereo_mode="stereo",
        segment_seconds=5.0,
        sample_rate=44100,
        seed=3407,
        audio_random_seed=None,
    )


@pytest.mark.parametrize("naming", ["official", "legacy_dt"])
def test_audio_dataset_scans_official_and_legacy_sonicmaster_pairs(
    tmp_path: Path, naming: str
):
    split_root = tmp_path / "pairs" / "train"
    split_root.mkdir(parents=True)
    metadata_path = tmp_path / "metadata.jsonl"
    _write_metadata(metadata_path)

    if naming == "official":
        names = ["track_001.flac", "track_001_deg1.flac", "track_001_deg2.flac"]
    else:
        names = [
            "track_001_DT0.flac",
            "track_001_DT1.flac",
            "track_001_DT2.flac",
        ]
    for name in names:
        (split_root / name).touch()

    dataset = AudioTrainDataset(
        _dataset_args(tmp_path / "pairs", metadata_path), split="train"
    )

    assert len(dataset.pairs) == 2
    assert {pair[2] for pair in dataset.pairs} == {"dt1", "dt2"}
    assert {Path(pair[1]).name for pair in dataset.pairs} == {names[0]}


@pytest.mark.parametrize("naming", ["official", "legacy_dt"])
def test_preflight_matches_official_and_legacy_sonicmaster_pairs(
    tmp_path: Path, naming: str
):
    audio_root = tmp_path / "pairs" / "train"
    audio_root.mkdir(parents=True)
    metadata_path = tmp_path / "metadata.jsonl"
    _write_metadata(metadata_path)

    if naming == "official":
        names = ["track_001.flac", "track_001_deg1.flac", "track_001_deg2.flac"]
    else:
        names = [
            "track_001_DT0.flac",
            "track_001_DT1.flac",
            "track_001_DT2.flac",
        ]
    for name in names:
        (audio_root / name).touch()

    report = audit_sonicmaster_data(
        audio_root=str(audio_root),
        metadata_jsonl=str(metadata_path),
        degradation_names=["dt1", "dt2"],
        audio_ext=".flac",
        verify_audio_headers=False,
    )

    assert report.clean_files == 1
    assert report.degraded_files_selected == 2
    assert report.matched_pair_ids == 2
    assert report.missing_pair_ids_count == 0
    assert report.extra_pair_ids_count == 0
    assert report.degraded_without_clean_count == 0


def test_pair_metadata_key_accepts_official_and_legacy_clean_names():
    assert pair_metadata_key("track_001.flac", "dt1") == "track_001_deg1"
    assert pair_metadata_key("track_001_DT0.flac", "dt2") == "track_001_deg2"


def test_pair_metadata_key_rejects_a_degraded_file_as_the_clean_target():
    with pytest.raises(ValueError, match="clean source"):
        pair_metadata_key("track_001_deg1.flac", "dt2")
