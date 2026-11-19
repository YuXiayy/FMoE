"""Auditable optimizer, schedule, distributed, and launch-time contracts."""

from __future__ import annotations

import json
import math
import os
import pathlib
import time
from datetime import datetime

import torch.optim as optim

from data.data_preflight import (
    SonicMasterPreflightReport,
    run_sonicmaster_preflight,
)


def build_adamw(parameters, opt):
    """Build AdamW from the fully serialized optimizer contract."""

    return optim.AdamW(
        parameters,
        lr=float(opt.lr),
        betas=(float(opt.adam_beta1), float(opt.adam_beta2)),
        eps=float(opt.adam_eps),
        weight_decay=float(opt.weight_decay),
    )


def warmup_cosine_multiplier(
    current_step: int,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_ratio: float,
) -> float:
    """Linear warmup followed by cosine decay, expressed in optimizer steps."""

    total_steps = max(1, int(total_steps))
    warmup_steps = int(total_steps * float(warmup_ratio))
    if warmup_ratio > 0.0:
        warmup_steps = max(1, warmup_steps)
    if current_step < warmup_steps:
        return float(current_step + 1) / float(warmup_steps)
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio) + (1.0 - float(min_ratio)) * cosine


def build_training_contract(opt, training_budget, *, data_preflight_report=None):
    """Serialize every setting that controls exact optimizer-step continuity."""

    contract = {
        "dataset_size": int(training_budget.dataset_size),
        "world_size": int(training_budget.world_size),
        "micro_batch_size_per_device": int(
            training_budget.micro_batch_size_per_device
        ),
        "accumulate_grad_batches": int(training_budget.accumulate_grad_batches),
        "max_epochs": int(training_budget.max_epochs),
        "max_steps": int(training_budget.max_steps),
        "planned_optimizer_steps": int(training_budget.planned_optimizer_steps),
        "optimizer": {
            "name": "AdamW",
            "lr": float(opt.lr),
            "weight_decay": float(opt.weight_decay),
            "betas": [float(opt.adam_beta1), float(opt.adam_beta2)],
            "eps": float(opt.adam_eps),
        },
        "scheduler": {
            "name": "linear_warmup_cosine",
            "warmup_ratio": float(opt.lr_warmup_ratio),
            "min_ratio": float(opt.lr_min_ratio),
            "total_steps": int(training_budget.planned_optimizer_steps),
        },
        "objective": {
            "name": "fmoe_5loss_v1",
            "weights": {
                "complex": float(opt.lambda_complex),
                "wave": float(opt.lambda_wave),
                "mrstft": float(opt.lambda_mrstft),
                "degradation": float(opt.lambda_degradation),
                "balance": float(opt.lambda_balance),
            },
            "stft": {
                "n_fft": int(opt.n_fft),
                "hop_length": int(opt.hop_length),
                "win_length": int(opt.win_length),
                "center": bool(opt.stft_center),
            },
            "mrstft": {
                "fft_sizes": [int(value) for value in opt.mrstft_fft_sizes],
                "hop_sizes": [int(value) for value in opt.mrstft_hop_sizes],
                "win_lengths": [int(value) for value in opt.mrstft_win_lengths],
                "spectral_convergence_normalization": str(
                    opt.mrstft_sc_normalization
                ),
            },
            "check_finite": bool(opt.loss_check_finite),
        },
        "audio": {
            "sample_rate": int(opt.sample_rate),
            "segment_seconds": float(opt.segment_seconds),
            "representation": str(opt.representation),
            "stereo_mode": str(opt.stereo_mode),
            "spec_log": bool(opt.spec_log),
        },
    }
    if data_preflight_report is not None:
        contract["data_identity"] = {
            "audio_root": data_preflight_report.audio_root,
            "metadata_jsonl": data_preflight_report.metadata_jsonl,
            "metadata_sha256": data_preflight_report.metadata_sha256,
            "matched_pair_ids": int(data_preflight_report.matched_pair_ids),
            "audio_headers_checked": int(
                data_preflight_report.audio_headers_checked
            ),
            "maximum_pair_length_delta_samples": int(
                data_preflight_report.maximum_pair_length_delta_samples
            ),
            "preflight_schema_version": int(
                data_preflight_report.preflight_schema_version
            ),
            "audio_decoder_tails_checked": int(
                data_preflight_report.audio_decoder_tails_checked
            ),
            "declared_decoded_length_mismatches_count": int(
                data_preflight_report.declared_decoded_length_mismatches_count
            ),
            "excluded_pair_ids": list(data_preflight_report.excluded_pair_ids),
        }
    return contract


def resolve_run_id(opt, environ=None):
    """Return one filesystem-safe run ID shared by every launched process."""

    environ = os.environ if environ is None else environ
    world_size = int(environ.get("WORLD_SIZE", "1"))
    inherited = environ.get("FMOE_RUN_ID")
    if opt.run_id:
        run_id = str(opt.run_id)
    elif inherited:
        run_id = inherited
    elif world_size > 1:
        raise RuntimeError(
            "External distributed launches require --run_id (or FMOE_RUN_ID) "
            "so ranks cannot write to different timestamped directories."
        )
    else:
        run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in run_id
    ):
        raise ValueError("run_id contains unsafe filesystem characters")
    if environ is os.environ:
        os.environ["FMOE_RUN_ID"] = run_id
    return run_id


def resolve_distributed_topology(*, requested_gpus, visible_gpus, environ=None):
    """Resolve Lightning's local devices separately from global world size."""

    environ = os.environ if environ is None else environ
    world_size = int(environ.get("WORLD_SIZE", "1"))
    rank = int(environ.get("RANK", "0"))
    local_rank = int(environ.get("LOCAL_RANK", "0"))
    if visible_gpus < 1:
        raise RuntimeError("No CUDA device is visible to this process.")
    if world_size > 1:
        local_world_size = int(environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if local_world_size < 1 or world_size % local_world_size:
            raise RuntimeError(
                f"Invalid torchrun topology WORLD_SIZE={world_size}, "
                f"LOCAL_WORLD_SIZE={local_world_size}"
            )
        if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
            raise RuntimeError(
                f"Invalid distributed ranks RANK={rank}, LOCAL_RANK={local_rank}"
            )
        if visible_gpus < local_world_size:
            raise RuntimeError(
                "torchrun exposes fewer local CUDA devices than LOCAL_WORLD_SIZE: "
                f"visible={visible_gpus}, local_world_size={local_world_size}"
            )
        return {
            "external": True,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "devices": local_world_size,
            "num_nodes": world_size // local_world_size,
        }

    devices = min(int(requested_gpus), int(visible_gpus))
    if devices < 1:
        raise RuntimeError("num_gpus must request at least one visible CUDA device")
    return {
        "external": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": devices,
        "devices": devices,
        "num_nodes": 1,
    }


def _write_preflight_status(path, payload):
    path = pathlib.Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _preflight_satisfies_request(opt, report):
    """Reject stale schema-v1 cache entries when deep decoder checks are required."""

    if opt.data_preflight_verify_audio_headers and report.audio_headers_checked < 1:
        return False
    if opt.data_preflight_verify_audio_decoder_tails:
        return (
            report.preflight_schema_version >= 2
            and report.audio_decoder_tails_checked == report.audio_headers_checked
            and report.audio_decoder_errors_count == 0
        )
    return True


def coordinated_data_preflight(opt, *, log_dir, checkpoint_path, rank):
    """Run the expensive inventory once and gate all external ranks on it."""

    status_path = pathlib.Path(checkpoint_path) / "data_preflight_status.json"
    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing and existing.get("status") == "ok":
            report = SonicMasterPreflightReport(**existing["report"])
            if _preflight_satisfies_request(opt, report):
                return report
        if existing and existing.get("status") == "error":
            raise RuntimeError(
                "SonicMaster preflight already failed for this run: "
                + str(existing.get("error", "unknown error"))
            )
    if rank == 0:
        report_paths = (
            os.path.join(log_dir, "data_preflight.json"),
            os.path.join(checkpoint_path, "data_preflight.json"),
        )
        try:
            report = run_sonicmaster_preflight(
                audio_root=opt.audio_data_dir,
                metadata_jsonl=opt.degradation_metadata_jsonl,
                degradation_names=opt.de_type,
                audio_ext=opt.audio_ext,
                example_limit=opt.data_preflight_example_limit,
                required_splits=opt.data_preflight_required_splits,
                require_source_disjoint=opt.data_preflight_require_source_disjoint,
                max_extra_pairs=opt.data_preflight_max_extra_pairs,
                ignored_degradation_classes=opt.degradation_ignored_classes,
                ignore_hidden_clipping_labels=(
                    opt.degradation_ignore_hidden_clipping_labels
                ),
                verify_audio_headers=opt.data_preflight_verify_audio_headers,
                verify_audio_decoder_tails=(
                    opt.data_preflight_verify_audio_decoder_tails
                ),
                excluded_pair_ids=opt.audio_excluded_pair_ids,
                target_sample_rate=opt.sample_rate,
                pair_length_tolerance_samples=(
                    opt.audio_pair_length_tolerance_samples
                ),
                audio_header_workers=opt.data_preflight_header_workers,
                header_progress_interval=(
                    opt.data_preflight_header_log_interval
                ),
                report_paths=report_paths,
            )
        except Exception as exc:
            _write_preflight_status(
                status_path,
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        _write_preflight_status(
            status_path, {"status": "ok", "report": report.as_dict()}
        )
        return report

    deadline = time.monotonic() + float(
        opt.distributed_preflight_timeout_seconds
    )
    while time.monotonic() < deadline:
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.2)
                continue
            if payload.get("status") == "error":
                raise RuntimeError(
                    "Rank-zero SonicMaster preflight failed: "
                    + str(payload.get("error", "unknown error"))
                )
            if payload.get("status") == "ok":
                report = SonicMasterPreflightReport(**payload["report"])
                if _preflight_satisfies_request(opt, report):
                    return report
        time.sleep(0.2)
    raise TimeoutError(
        "Timed out waiting for rank-zero SonicMaster data preflight after "
        f"{opt.distributed_preflight_timeout_seconds:g} seconds"
    )


__all__ = [
    "build_adamw",
    "build_training_contract",
    "coordinated_data_preflight",
    "resolve_distributed_topology",
    "resolve_run_id",
    "warmup_cosine_multiplier",
]
