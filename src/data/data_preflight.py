"""Auditable SonicMaster metadata/audio inventory checks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from data.degradation_metadata import (
    SONICMASTER_DEGRADATION_LABELS,
    normalize_sonicmaster_record,
)

try:
    import soundfile as _soundfile
except Exception:  # pragma: no cover - exercised on dependency-poor hosts
    _soundfile = None

try:
    import torchaudio as _torchaudio
except Exception:  # pragma: no cover - exercised on dependency-poor hosts
    _torchaudio = None


_AUDIO_STEM = re.compile(
    r"^(?P<source>.+)_(?P<kind>dt|deg)(?P<version>\d+)$", re.IGNORECASE
)
_METADATA_ID = re.compile(
    r"^(?P<source>.+)_deg(?P<version>\d+)$", re.IGNORECASE
)


def _examples(values: Iterable[str], limit: int) -> Tuple[str, ...]:
    return tuple(sorted(values)[:limit])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def degradation_versions(names: Sequence[str]) -> Tuple[int, ...]:
    versions = []
    for name in names:
        match = re.fullmatch(r"dt(?P<version>\d+)", str(name), re.IGNORECASE)
        if match is None or int(match.group("version")) == 0:
            raise ValueError(f"Expected non-clean degradation name dt<number>, got {name!r}")
        versions.append(int(match.group("version")))
    if len(versions) != len(set(versions)):
        raise ValueError("degradation versions must not contain duplicates")
    return tuple(sorted(versions))


@dataclass(frozen=True)
class SonicMasterPreflightReport:
    audio_root: str
    metadata_jsonl: str
    metadata_sha256: str
    selected_degradation_versions: Tuple[int, ...]
    metadata_records: int
    metadata_records_by_split: Mapping[str, int]
    metadata_sources_by_split: Mapping[str, int]
    metadata_source_groups_by_split: Mapping[str, int]
    label_rows: Mapping[str, int]
    compound_rows: Mapping[str, int]
    vocal_rows: int
    hidden_clipping_rows: int
    audio_files_scanned: int
    clean_files: int
    degraded_files_selected: int
    matched_pair_ids: int
    missing_pair_ids_count: int
    missing_pair_ids_examples: Tuple[str, ...]
    extra_pair_ids_count: int
    extra_pair_ids_examples: Tuple[str, ...]
    duplicate_clean_sources_count: int
    duplicate_clean_sources_examples: Tuple[str, ...]
    duplicate_degraded_pair_ids_count: int
    duplicate_degraded_pair_ids_examples: Tuple[str, ...]
    degraded_without_clean_count: int
    degraded_without_clean_examples: Tuple[str, ...]
    source_overlap: Mapping[str, Mapping[str, object]]
    source_group_overlap: Mapping[str, Mapping[str, object]]
    audio_headers_checked: int
    audio_header_errors_count: int
    audio_header_errors_examples: Tuple[str, ...]
    pair_length_mismatches_count: int
    pair_length_mismatches_examples: Tuple[str, ...]
    maximum_pair_length_delta_samples: int
    # Defaults preserve readability of schema-v1 reports while ensuring that
    # callers can tell whether runtime-decoder tail integrity was actually
    # checked.  New reports always write schema version 2.
    preflight_schema_version: int = 1
    audio_decoder_tails_checked: int = 0
    audio_decoder_errors_count: int = 0
    audio_decoder_errors_examples: Tuple[str, ...] = ()
    declared_decoded_length_mismatches_count: int = 0
    declared_decoded_length_mismatches_examples: Tuple[str, ...] = ()
    excluded_pair_ids_count: int = 0
    excluded_pair_ids: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def audit_sonicmaster_data(
    *,
    audio_root: str,
    metadata_jsonl: str,
    degradation_names: Sequence[str],
    audio_ext: str = ".flac",
    clean_version: int = 0,
    example_limit: int = 10,
    verify_audio_headers: bool = False,
    verify_audio_decoder_tails: bool = False,
    excluded_pair_ids: Sequence[str] = (),
    target_sample_rate: int = 44100,
    pair_length_tolerance_samples: int = 0,
    audio_header_workers: int = 1,
    header_progress_interval: int = 0,
) -> SonicMasterPreflightReport:
    """Scan metadata plus optional bounded header/decoder integrity probes."""

    root = Path(audio_root)
    metadata_path = Path(metadata_jsonl)
    if not root.is_dir():
        raise FileNotFoundError(f"SonicMaster audio root not found: {root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"SonicMaster metadata JSONL not found: {metadata_path}")
    if example_limit < 1:
        raise ValueError("example_limit must be positive")
    if target_sample_rate < 1:
        raise ValueError("target_sample_rate must be positive")
    if pair_length_tolerance_samples < 0:
        raise ValueError("pair_length_tolerance_samples must be non-negative")
    if audio_header_workers < 1:
        raise ValueError("audio_header_workers must be positive")
    if header_progress_interval < 0:
        raise ValueError("header_progress_interval must be non-negative")
    if verify_audio_decoder_tails and not verify_audio_headers:
        raise ValueError(
            "verify_audio_decoder_tails requires verify_audio_headers"
        )
    selected_versions = degradation_versions(degradation_names)
    selected_version_set = set(selected_versions)

    excluded_pair_id_set = set()
    for raw_pair_id in excluded_pair_ids:
        pair_id = str(raw_pair_id).casefold()
        match = _METADATA_ID.fullmatch(pair_id)
        if match is None or int(match.group("version")) not in selected_version_set:
            raise ValueError(
                "excluded_pair_ids must contain selected metadata IDs such as "
                f"source_deg1, got {raw_pair_id!r}"
            )
        if pair_id in excluded_pair_id_set:
            raise ValueError(f"duplicate excluded pair ID {pair_id!r}")
        excluded_pair_id_set.add(pair_id)

    metadata_ids = set()
    seen_metadata_ids = set()
    excluded_pair_ids_found = set()
    sources_by_split = defaultdict(set)
    source_groups_by_split = defaultdict(set)
    records_by_split = Counter()
    label_rows = Counter({name: 0 for name in SONICMASTER_DEGRADATION_LABELS})
    compound_rows = Counter()
    hidden_clipping_rows = 0

    with metadata_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id", "")).lower()
            match = _METADATA_ID.fullmatch(record_id)
            if match is None:
                raise ValueError(
                    f"Malformed SonicMaster id at {metadata_path}:{line_number}: {record_id!r}"
                )
            version = int(match.group("version"))
            if version not in selected_version_set:
                continue
            if record_id in seen_metadata_ids:
                raise ValueError(f"Duplicate SonicMaster id {record_id!r}")
            seen_metadata_ids.add(record_id)
            if record_id in excluded_pair_id_set:
                excluded_pair_ids_found.add(record_id)
                continue
            split = str(record.get("split", "")).lower()
            if not split:
                raise ValueError(
                    f"Missing split at {metadata_path}:{line_number} for {record_id!r}"
                )
            normalized = normalize_sonicmaster_record(record)
            metadata_ids.add(record_id)
            source_id = match.group("source").lower()
            source_group_id = source_id.split("_", maxsplit=1)[0]
            sources_by_split[split].add(source_id)
            source_groups_by_split[split].add(source_group_id)
            records_by_split[split] += 1
            compound_rows[str(normalized["compound_level"])] += 1
            for name in normalized["names"]:
                label_rows[name] += 1
            if normalized["hidden_clipping"]["enabled"]:
                hidden_clipping_rows += 1

    unknown_exclusions = excluded_pair_id_set - excluded_pair_ids_found
    if unknown_exclusions:
        raise ValueError(
            "excluded pair IDs are absent from selected metadata: "
            f"{sorted(unknown_exclusions)}"
        )

    metadata_sources = {
        match.group("source").lower()
        for record_id in metadata_ids
        for match in [_METADATA_ID.fullmatch(record_id)]
        if match is not None
    }

    extension = audio_ext.lower()
    if not extension.startswith("."):
        extension = "." + extension
    clean_first: Dict[str, str] = {}
    degraded_first: Dict[str, str] = {}
    duplicate_clean_sources = set()
    duplicate_degraded_ids = set()
    audio_files_scanned = 0
    clean_files = 0
    degraded_files_selected = 0

    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            if not filename.lower().endswith(extension):
                continue
            audio_files_scanned += 1
            stem = filename[: -len(extension)]
            match = _AUDIO_STEM.fullmatch(stem)
            path = os.path.join(directory, filename)
            if match is None:
                # Official SonicMaster clean files use the unsuffixed source ID.
                source_id = stem.lower()
                if source_id not in metadata_sources:
                    continue
                is_clean = True
                is_selected_degradation = False
                version = clean_version
            else:
                source_id = match.group("source").lower()
                version = int(match.group("version"))
                kind = match.group("kind").lower()
                is_clean = kind == "dt" and version == clean_version
                is_selected_degradation = (
                    kind in {"dt", "deg"} and version in selected_version_set
                )

            if is_clean:
                clean_files += 1
                if source_id in clean_first:
                    duplicate_clean_sources.add(source_id)
                else:
                    clean_first[source_id] = path
            elif is_selected_degradation:
                degraded_files_selected += 1
                pair_id = f"{source_id}_deg{version}"
                if pair_id in degraded_first:
                    duplicate_degraded_ids.add(pair_id)
                else:
                    degraded_first[pair_id] = path

    audio_pair_ids = set(degraded_first)
    active_audio_pair_ids = audio_pair_ids - excluded_pair_id_set
    missing_pair_ids = metadata_ids - active_audio_pair_ids
    extra_pair_ids = active_audio_pair_ids - metadata_ids
    degraded_without_clean = {
        pair_id
        for pair_id in active_audio_pair_ids
        if _METADATA_ID.fullmatch(pair_id).group("source").lower()
        not in clean_first
    }

    audio_header_errors = set()
    audio_decoder_errors = set()
    declared_decoded_length_mismatches = set()
    pair_length_mismatches = set()
    maximum_pair_length_delta = 0
    header_cache = {}
    header_paths_checked = set()
    decoder_tail_paths_checked = set()
    if verify_audio_headers:
        if _soundfile is None:
            raise RuntimeError(
                "soundfile is required for data_preflight_verify_audio_headers"
            )
        if verify_audio_decoder_tails and _torchaudio is None:
            raise RuntimeError(
                "torchaudio is required for data_preflight_verify_audio_decoder_tails"
            )

        matched_paths = []
        for pair_id in sorted(metadata_ids & active_audio_pair_ids):
            match = _METADATA_ID.fullmatch(pair_id)
            source_id = match.group("source").lower()
            clean_path = clean_first.get(source_id)
            degraded_path = degraded_first[pair_id]
            if clean_path is not None:
                matched_paths.append((pair_id, clean_path, degraded_path))

        unique_paths = sorted(
            {
                path
                for _, clean_path, degraded_path in matched_paths
                for path in (clean_path, degraded_path)
            }
        )

        def read_audio_probe(path: str):
            try:
                info = _soundfile.info(path)
                frames = int(info.frames)
                sample_rate = int(info.samplerate)
                if frames < 1 or sample_rate < 1:
                    raise ValueError(
                        f"invalid frames/sample-rate metadata: {frames}/{sample_rate}"
                    )
            except Exception as exc:
                return (
                    path,
                    None,
                    f"{path}: {type(exc).__name__}: {exc}",
                    None,
                    None,
                    False,
                )

            decoded_frames = frames
            decoder_error: Optional[str] = None
            declared_mismatch: Optional[str] = None
            decoder_checked = False
            if verify_audio_decoder_tails:
                try:
                    tail, decoded_sample_rate = _torchaudio.load(
                        path,
                        frame_offset=frames - 1,
                        num_frames=1,
                    )
                    decoder_checked = True
                    if int(decoded_sample_rate) != sample_rate:
                        raise ValueError(
                            "decoder/header sample-rate mismatch: "
                            f"{decoded_sample_rate}/{sample_rate}"
                        )
                    if int(tail.shape[-1]) != 1:
                        waveform, decoded_sample_rate = _torchaudio.load(path)
                        if int(decoded_sample_rate) != sample_rate:
                            raise ValueError(
                                "full decoder/header sample-rate mismatch: "
                                f"{decoded_sample_rate}/{sample_rate}"
                            )
                        decoded_frames = int(waveform.shape[-1])
                        if decoded_frames < 1:
                            raise ValueError("runtime decoder returned empty audio")
                        if decoded_frames != frames:
                            declared_mismatch = (
                                f"{path}: declared={frames}, decoded={decoded_frames}, "
                                f"delta={abs(frames - decoded_frames)}"
                            )
                except Exception as exc:
                    decoder_error = f"{path}: {type(exc).__name__}: {exc}"

            normalized = int(
                math.ceil(
                    decoded_frames * float(target_sample_rate) / sample_rate
                )
            )
            return (
                path,
                normalized,
                None,
                decoder_error,
                declared_mismatch,
                decoder_checked,
            )

        # Submit bounded batches rather than all ~200k paths at once. This
        # keeps future objects bounded while allowing storage I/O to overlap.
        batch_size = 4096
        checked = 0
        next_progress = header_progress_interval
        workers = min(audio_header_workers, max(1, len(unique_paths)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch_start in range(0, len(unique_paths), batch_size):
                batch = unique_paths[batch_start : batch_start + batch_size]
                for (
                    path,
                    normalized,
                    header_error,
                    decoder_error,
                    declared_mismatch,
                    decoder_checked,
                ) in executor.map(
                    read_audio_probe, batch
                ):
                    header_cache[path] = normalized
                    if header_error is not None:
                        audio_header_errors.add(header_error)
                    else:
                        header_paths_checked.add(path)
                    if decoder_error is not None:
                        audio_decoder_errors.add(decoder_error)
                        header_cache[path] = None
                    if declared_mismatch is not None:
                        declared_decoded_length_mismatches.add(declared_mismatch)
                    if decoder_checked:
                        decoder_tail_paths_checked.add(path)
                checked += len(batch)
                if header_progress_interval and checked >= next_progress:
                    print(
                        "[Data preflight headers] "
                        f"files={checked:,}/{len(unique_paths):,}, "
                        f"header_errors={len(audio_header_errors)}, "
                        f"decoder_errors={len(audio_decoder_errors)}, "
                        "declared_decoded_mismatches="
                        f"{len(declared_decoded_length_mismatches)}",
                        flush=True,
                    )
                    while next_progress <= checked:
                        next_progress += header_progress_interval
        if header_progress_interval and checked and checked < next_progress:
            print(
                "[Data preflight headers] "
                f"files={checked:,}/{len(unique_paths):,}, "
                f"header_errors={len(audio_header_errors)}, "
                f"decoder_errors={len(audio_decoder_errors)}, "
                "declared_decoded_mismatches="
                f"{len(declared_decoded_length_mismatches)} (complete)",
                flush=True,
            )

        for pair_id, clean_path, degraded_path in matched_paths:
            clean_length = header_cache[clean_path]
            degraded_length = header_cache[degraded_path]
            if clean_length is None or degraded_length is None:
                continue
            delta = abs(int(clean_length) - int(degraded_length))
            maximum_pair_length_delta = max(maximum_pair_length_delta, delta)
            if delta > pair_length_tolerance_samples:
                pair_length_mismatches.add(
                    f"{pair_id}: clean={clean_length}, degraded={degraded_length}, "
                    f"delta={delta}"
                )

    source_overlap = {}
    source_group_overlap = {}
    for first_split, second_split in itertools.combinations(
        sorted(sources_by_split), 2
    ):
        overlap = sources_by_split[first_split] & sources_by_split[second_split]
        key = f"{first_split}__{second_split}"
        source_overlap[key] = {
            "count": len(overlap),
            "examples": list(_examples(overlap, example_limit)),
        }
        group_overlap = (
            source_groups_by_split[first_split]
            & source_groups_by_split[second_split]
        )
        source_group_overlap[key] = {
            "count": len(group_overlap),
            "examples": list(_examples(group_overlap, example_limit)),
        }

    return SonicMasterPreflightReport(
        audio_root=str(root.resolve()),
        metadata_jsonl=str(metadata_path.resolve()),
        metadata_sha256=_sha256(metadata_path),
        selected_degradation_versions=selected_versions,
        metadata_records=len(metadata_ids),
        metadata_records_by_split=dict(sorted(records_by_split.items())),
        metadata_sources_by_split={
            split: len(sources) for split, sources in sorted(sources_by_split.items())
        },
        metadata_source_groups_by_split={
            split: len(groups)
            for split, groups in sorted(source_groups_by_split.items())
        },
        label_rows={name: int(label_rows[name]) for name in SONICMASTER_DEGRADATION_LABELS},
        compound_rows=dict(sorted(compound_rows.items())),
        vocal_rows=int(label_rows["vocal"]),
        hidden_clipping_rows=hidden_clipping_rows,
        audio_files_scanned=audio_files_scanned,
        clean_files=clean_files,
        degraded_files_selected=degraded_files_selected,
        matched_pair_ids=len(metadata_ids & active_audio_pair_ids),
        missing_pair_ids_count=len(missing_pair_ids),
        missing_pair_ids_examples=_examples(missing_pair_ids, example_limit),
        extra_pair_ids_count=len(extra_pair_ids),
        extra_pair_ids_examples=_examples(extra_pair_ids, example_limit),
        duplicate_clean_sources_count=len(duplicate_clean_sources),
        duplicate_clean_sources_examples=_examples(
            duplicate_clean_sources, example_limit
        ),
        duplicate_degraded_pair_ids_count=len(duplicate_degraded_ids),
        duplicate_degraded_pair_ids_examples=_examples(
            duplicate_degraded_ids, example_limit
        ),
        degraded_without_clean_count=len(degraded_without_clean),
        degraded_without_clean_examples=_examples(
            degraded_without_clean, example_limit
        ),
        source_overlap=source_overlap,
        source_group_overlap=source_group_overlap,
        audio_headers_checked=len(header_paths_checked),
        audio_header_errors_count=len(audio_header_errors),
        audio_header_errors_examples=_examples(audio_header_errors, example_limit),
        pair_length_mismatches_count=len(pair_length_mismatches),
        pair_length_mismatches_examples=_examples(
            pair_length_mismatches, example_limit
        ),
        maximum_pair_length_delta_samples=maximum_pair_length_delta,
        preflight_schema_version=2,
        audio_decoder_tails_checked=len(decoder_tail_paths_checked),
        audio_decoder_errors_count=len(audio_decoder_errors),
        audio_decoder_errors_examples=_examples(
            audio_decoder_errors, example_limit
        ),
        declared_decoded_length_mismatches_count=len(
            declared_decoded_length_mismatches
        ),
        declared_decoded_length_mismatches_examples=_examples(
            declared_decoded_length_mismatches, example_limit
        ),
        excluded_pair_ids_count=len(excluded_pair_id_set),
        excluded_pair_ids=tuple(sorted(excluded_pair_id_set)),
    )


def enforce_sonicmaster_preflight(
    report: SonicMasterPreflightReport,
    *,
    required_splits: Sequence[str],
    require_source_disjoint: bool,
    max_extra_pairs: int,
    ignored_degradation_classes: Sequence[str],
    ignore_hidden_clipping_labels: bool,
) -> None:
    """Turn integrity and known-label risks into explicit launch blockers."""

    if max_extra_pairs < 0:
        raise ValueError("max_extra_pairs must be non-negative")
    errors = []
    available_splits = set(report.metadata_records_by_split)
    missing_splits = sorted(
        {str(split).lower() for split in required_splits} - available_splits
    )
    if missing_splits:
        errors.append(f"missing metadata splits: {missing_splits}")
    if report.missing_pair_ids_count:
        errors.append(
            f"{report.missing_pair_ids_count} metadata rows have no degraded audio"
        )
    if report.extra_pair_ids_count > max_extra_pairs:
        errors.append(
            f"{report.extra_pair_ids_count} audio pairs lack metadata "
            f"(allowed {max_extra_pairs})"
        )
    if report.duplicate_clean_sources_count:
        errors.append(
            f"{report.duplicate_clean_sources_count} source IDs have duplicate clean files"
        )
    if report.duplicate_degraded_pair_ids_count:
        errors.append(
            f"{report.duplicate_degraded_pair_ids_count} pair IDs have duplicate degraded files"
        )
    if report.degraded_without_clean_count:
        errors.append(
            f"{report.degraded_without_clean_count} degraded pairs have no clean source"
        )
    if report.audio_header_errors_count:
        errors.append(
            f"{report.audio_header_errors_count} audio files have unreadable headers"
        )
    if report.audio_decoder_errors_count:
        errors.append(
            f"{report.audio_decoder_errors_count} audio files fail runtime-decoder "
            "tail integrity"
        )
    if report.pair_length_mismatches_count:
        errors.append(
            f"{report.pair_length_mismatches_count} decoded pairs exceed the "
            "configured length tolerance"
        )
    if require_source_disjoint:
        leaking_clips = {
            key: int(value["count"])
            for key, value in report.source_overlap.items()
            if int(value["count"]) > 0
        }
        leaking_groups = {
            key: int(value["count"])
            for key, value in report.source_group_overlap.items()
            if int(value["count"]) > 0
        }
        if leaking_clips or leaking_groups:
            errors.append(
                "source-level split overlap detected: "
                f"clips={leaking_clips}, source_groups={leaking_groups}"
            )

    if errors:
        raise RuntimeError(
            "SonicMaster data preflight failed:\n- " + "\n- ".join(errors)
        )


def write_sonicmaster_preflight_report(
    report: SonicMasterPreflightReport,
    paths: Sequence[str],
) -> None:
    payload = report.as_dict()
    for raw_path in paths:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")


def format_sonicmaster_preflight(report: SonicMasterPreflightReport) -> str:
    clip_overlap_count = sum(
        int(value["count"]) for value in report.source_overlap.values()
    )
    group_overlap_count = sum(
        int(value["count"]) for value in report.source_group_overlap.values()
    )
    return (
        "[Data preflight] "
        f"metadata={report.metadata_records:,}, matched={report.matched_pair_ids:,}, "
        f"missing={report.missing_pair_ids_count}, extra={report.extra_pair_ids_count}, "
        f"clip_overlap={clip_overlap_count}, source_group_overlap={group_overlap_count}, "
        f"vocal={report.vocal_rows:,}, "
        f"hidden_clipping={report.hidden_clipping_rows:,}, "
        f"excluded={report.excluded_pair_ids_count}, "
        f"header_errors={report.audio_header_errors_count}, "
        f"decoder_errors={report.audio_decoder_errors_count}, "
        "declared_decoded_mismatches="
        f"{report.declared_decoded_length_mismatches_count}, "
        f"length_mismatches={report.pair_length_mismatches_count}"
    )


def run_sonicmaster_preflight(
    *,
    audio_root: str,
    metadata_jsonl: str,
    degradation_names: Sequence[str],
    audio_ext: str,
    example_limit: int,
    required_splits: Sequence[str],
    require_source_disjoint: bool,
    max_extra_pairs: int,
    ignored_degradation_classes: Sequence[str],
    ignore_hidden_clipping_labels: bool,
    verify_audio_headers: bool = False,
    verify_audio_decoder_tails: bool = False,
    excluded_pair_ids: Sequence[str] = (),
    target_sample_rate: int = 44100,
    pair_length_tolerance_samples: int = 0,
    audio_header_workers: int = 1,
    header_progress_interval: int = 0,
    report_paths: Sequence[str] = (),
) -> SonicMasterPreflightReport:
    """Run the complete audit/enforcement contract and optionally persist it."""

    report = audit_sonicmaster_data(
        audio_root=audio_root,
        metadata_jsonl=metadata_jsonl,
        degradation_names=degradation_names,
        audio_ext=audio_ext,
        example_limit=example_limit,
        verify_audio_headers=verify_audio_headers,
        verify_audio_decoder_tails=verify_audio_decoder_tails,
        excluded_pair_ids=excluded_pair_ids,
        target_sample_rate=target_sample_rate,
        pair_length_tolerance_samples=pair_length_tolerance_samples,
        audio_header_workers=audio_header_workers,
        header_progress_interval=header_progress_interval,
    )
    # Persist the immutable inventory before enforcement so a failed launch
    # leaves evidence describing the exact dataset that was rejected.
    if report_paths:
        write_sonicmaster_preflight_report(report, report_paths)
    enforce_sonicmaster_preflight(
        report,
        required_splits=required_splits,
        require_source_disjoint=require_source_disjoint,
        max_extra_pairs=max_extra_pairs,
        ignored_degradation_classes=ignored_degradation_classes,
        ignore_hidden_clipping_labels=ignore_hidden_clipping_labels,
    )
    return report


__all__ = [
    "SonicMasterPreflightReport",
    "audit_sonicmaster_data",
    "degradation_versions",
    "enforce_sonicmaster_preflight",
    "format_sonicmaster_preflight",
    "run_sonicmaster_preflight",
    "write_sonicmaster_preflight_report",
]
