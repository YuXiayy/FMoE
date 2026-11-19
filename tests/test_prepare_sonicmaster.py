import json
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from scripts.prepare_sonicmaster import export_dataset


def test_export_dataset_builds_official_pair_layout(tmp_path: Path):
    parquet_root = tmp_path / "raw"
    parquet_root.mkdir()
    metadata = {
        "split": "train",
        "degradations": ["EQ"],
        "degradations_specifics": ["bright"],
        "degradation_tracking": {"EQ": ["bright", {}]},
        "hidden_clipping": [False, 0],
    }
    table = pyarrow.table(
        {
            "id": ["song_001_deg1"],
            "input_flac": [b"degraded-flac"],
            "gt_flac": [b"clean-flac"],
            "meta": [json.dumps(metadata)],
        }
    )
    pq.write_table(table, parquet_root / "train-00000.parquet")

    output = tmp_path / "prepared"
    count = export_dataset(parquet_root, output, overwrite=False, limit=0)

    assert count == 1
    assert (output / "pairs/train/song_001.flac").read_bytes() == b"clean-flac"
    assert (output / "pairs/train/song_001_deg1.flac").read_bytes() == b"degraded-flac"
    row = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))
    assert row["id"] == "song_001_deg1"
    assert row["split"] == "train"


def test_export_dataset_rejects_path_traversal_ids(tmp_path: Path):
    parquet_root = tmp_path / "raw"
    parquet_root.mkdir()
    table = pyarrow.table(
        {
            "id": ["../song_deg1"],
            "input_flac": [b"degraded"],
            "gt_flac": [b"clean"],
            "meta": [
                json.dumps(
                    {
                        "split": "train",
                        "degradations": ["EQ"],
                        "degradations_specifics": ["bright"],
                        "degradation_tracking": {"EQ": ["bright", {}]},
                        "hidden_clipping": [False, 0],
                    }
                )
            ],
        }
    )
    pq.write_table(table, parquet_root / "train-00000.parquet")

    with pytest.raises(ValueError, match="invalid row id"):
        export_dataset(parquet_root, tmp_path / "prepared", overwrite=False, limit=0)
