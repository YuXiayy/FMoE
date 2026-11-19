#!/usr/bin/env python3
"""Unified evaluation entry point for F-MoE and SonicMaster experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


COMMANDS = {
    "sonicmaster": "Run paired SonicMaster test-set restoration with a Lightning checkpoint.",
    "prepare": "Validate a manifest and prepare prediction folders for metric evaluation.",
    "signal": "Compute supplementary MM-SNR, LSD, and SI-SDRi metrics.",
    "visqol": "Compute optional ViSQOL audio scores.",
    "merge": "Merge signal and ViSQOL metric summaries.",
    "validate": "Validate evaluation outputs and prediction alignment.",
    "self-test": "Run the signal-metric smoke test.",
    "kl-ssim": "Compute log-mel KL divergence and SSIM.",
    "fad": "Compute FAD using a supplied CLAP checkpoint.",
    "pq": "Compute Production Quality using a supplied Audiobox checkpoint.",
    "verify-kernels": "Check numerical equivalence of the LSD and SI-SDRi kernels.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified F-MoE evaluation interface. Put the selected workflow first; "
            "all following arguments are forwarded to that workflow."
        ),
        epilog=(
            "Examples:\n"
            "  python evaluate.py sonicmaster --config configs/fmoe_train.yaml --checkpoint_path path/to/model.ckpt\n"
            "  python evaluate.py signal --root evaluation_runs/paper_test --models fmoe"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=tuple(COMMANDS), metavar="COMMAND")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = parser.parse_args([arguments[0]]).command
    forwarded = arguments[1:]

    if command == "sonicmaster":
        from options import train_options
        from evaluate_sonicmaster import main as run_sonicmaster

        options = train_options(forwarded)
        if options.input_audio is not None or options.output_audio is not None:
            raise SystemExit(
                "Single-file restoration belongs to inference.py. "
                "Use 'evaluate.py sonicmaster' only for paired test-set evaluation."
            )
        run_sonicmaster(options)
        return 0

    if command == "prepare":
        from evaluation.prepare_protocol import main as run_prepare

        run_prepare(forwarded)
        return 0

    if command in {"signal", "visqol", "merge", "validate", "self-test"}:
        from evaluation.run_audio_metrics import main as run_audio_metrics

        run_audio_metrics([command, *forwarded])
        return 0

    if command in {"kl-ssim", "fad", "pq"}:
        from evaluation.run_model_metrics import main as run_model_metrics

        run_model_metrics([command, *forwarded])
        return 0

    if command == "verify-kernels":
        if forwarded in (["--help"], ["-h"]):
            print("Usage: python evaluate.py verify-kernels")
            return 0
        if forwarded:
            parser.error("verify-kernels does not accept additional arguments")
        from evaluation.verify_metric_kernels import main as run_verify_kernels

        run_verify_kernels()
        return 0

    parser.error(f"unsupported evaluation command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
