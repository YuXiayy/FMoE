#!/usr/bin/env python3
"""Download the released F-MoE checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "Xianyyy/F-MoE"
REMOTE_FILENAME = "fmoce_5s_step265000_fp32.pth"
LOCAL_FILENAME = "fmoe_5s.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=REMOTE_FILENAME,
        local_dir=args.output_dir,
    )
    source = Path(path)
    destination = args.output_dir / LOCAL_FILENAME
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
