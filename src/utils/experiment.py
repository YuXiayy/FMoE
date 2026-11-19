"""Reproducible experiment metadata helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

import torch
import torch.nn as nn


Config = Union[Namespace, Mapping[str, Any]]


def runtime_versions() -> Dict[str, Any]:
    """Capture the runtime contract needed to reproduce a checkpoint."""

    packages = {}
    for name in ("lightning", "numpy", "torchaudio", "einops", "fvcore"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": packages,
    }


def serializable_config(config: Config) -> Dict[str, Any]:
    """Return a JSON-serializable shallow copy of a run configuration."""
    values = vars(config) if isinstance(config, Namespace) else dict(config)
    # Round-trip through JSON to fail before training if a new option is not
    # serializable. This also normalizes pathlib values if present.
    normalized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }
    json.dumps(normalized)
    return normalized


def current_git_commit(repo_root: Path) -> str:
    """Return the checked-out commit, or an explicit unavailable marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable (source tree is not a readable Git checkout)"
    return result.stdout.strip()


def model_summary_text(model: nn.Module) -> str:
    """Build a deterministic text summary with total/trainable parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    return "\n".join(
        [
            f"model_class: {model.__class__.__name__}",
            f"total_parameters: {total}",
            f"trainable_parameters: {trainable}",
            f"buffers: {buffer_count}",
            "",
            str(model),
            "",
        ]
    )


def write_experiment_artifacts(
    directories: Iterable[Union[str, Path]],
    config: Config,
    model: nn.Module,
    repo_root: Union[str, Path],
) -> Dict[str, Any]:
    """Write ``config.json``, ``git_commit.txt`` and ``model_summary.txt``.

    ``run_config.json`` mirrors ``config.json`` for external experiment tools.
    """
    config_data = serializable_config(config)
    config_data.setdefault("runtime_versions", runtime_versions())
    commit = current_git_commit(Path(repo_root))
    summary = model_summary_text(model)
    for directory_value in directories:
        directory = Path(directory_value)
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "run_config.json"):
            with (directory / filename).open("w", encoding="utf-8") as output:
                json.dump(config_data, output, indent=2, sort_keys=True)
                output.write("\n")
        (directory / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        (directory / "model_summary.txt").write_text(summary, encoding="utf-8")
    return config_data
