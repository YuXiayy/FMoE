"""Canonical SonicMaster degradation labels and JSONL metadata parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch


# Exact order from SonicMaster's five-group data generator.  The order is a
# model/data contract and must be stored with every checkpoint configuration.
SONICMASTER_DEGRADATION_LABELS = (
    "xband",
    "mic",
    "bright",
    "dark",
    "airy",
    "boom",
    "clarity",
    "mud",
    "warm",
    "vocal",
    "comp",
    "punch",
    "small",
    "big",
    "mix",
    "real",
    "clip",
    "volume",
    "stereo",
)

SONICMASTER_DEGRADATION_GROUPS = {
    "EQ": SONICMASTER_DEGRADATION_LABELS[:10],
    "Dynamics": SONICMASTER_DEGRADATION_LABELS[10:12],
    "Reverb": SONICMASTER_DEGRADATION_LABELS[12:16],
    "Amplitude": SONICMASTER_DEGRADATION_LABELS[16:18],
    "Stereo": SONICMASTER_DEGRADATION_LABELS[18:],
}

# ``degrade_final.py`` samples groups through sets, so the order serialized in
# ``degradations`` and ``degradations_specifics`` is not an application order.
# The effects are actually executed by these five consecutive code blocks.
SONICMASTER_APPLICATION_GROUP_ORDER = (
    "Dynamics",
    "EQ",
    "Reverb",
    "Stereo",
    "Amplitude",
)

_LABEL_TO_ID = {
    name: index for index, name in enumerate(SONICMASTER_DEGRADATION_LABELS)
}
_LABEL_TO_GROUP = {
    name: group
    for group, labels in SONICMASTER_DEGRADATION_GROUPS.items()
    for name in labels
}
_GROUP_BY_LOWER_NAME = {
    group.lower(): group for group in SONICMASTER_DEGRADATION_GROUPS
}
_PAIR_PATTERN = re.compile(r"^(?P<source>.+)_dt(?P<version>\d+)$", re.IGNORECASE)


def degradation_multi_hot(
    names: Sequence[str],
    labels: Sequence[str] = SONICMASTER_DEGRADATION_LABELS,
) -> torch.Tensor:
    """Encode one SonicMaster single/double/triple label list as ``[19]``."""
    label_to_id = {name.lower(): index for index, name in enumerate(labels)}
    result = torch.zeros(len(labels), dtype=torch.float32)
    for raw_name in names:
        name = str(raw_name).lower()
        if name not in label_to_id:
            raise ValueError(f"Unknown SonicMaster degradation label: {raw_name!r}")
        result[label_to_id[name]] = 1.0
    return result


def normalize_sonicmaster_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one official JSONL row to the auditable application chain.

    SonicMaster's generator converts the selected group and label sets to
    independent lists before serializing them.  Their positions therefore
    cannot be zipped and neither list represents execution order.  We derive
    each label's group from the official 19-label taxonomy and then order the
    chain by the generator's actual Dynamics -> EQ -> Reverb -> Stereo ->
    Amplitude execution sequence.
    """
    raw_names = [
        str(name).lower() for name in record.get("degradations_specifics", [])
    ]
    if not 1 <= len(raw_names) <= 3:
        raise ValueError(
            f"Expected one to three degradation specifics for {record.get('id')!r}, "
            f"got {raw_names!r}"
        )
    unknown = [name for name in raw_names if name not in _LABEL_TO_ID]
    if unknown:
        raise ValueError(f"Unknown SonicMaster degradation specifics: {unknown}")

    names_by_group: Dict[str, str] = {}
    for name in raw_names:
        group = _LABEL_TO_GROUP[name]
        if group in names_by_group:
            raise ValueError(
                "SonicMaster records must contain at most one degradation per "
                f"group; got {names_by_group[group]!r} and {name!r} in {group}"
            )
        names_by_group[group] = name

    raw_groups = list(record.get("degradations", []))
    canonical_source_groups = []
    for raw_group in raw_groups:
        normalized_group = str(raw_group).lower()
        if normalized_group not in _GROUP_BY_LOWER_NAME:
            raise ValueError(f"Unknown SonicMaster degradation group: {raw_group!r}")
        canonical_source_groups.append(_GROUP_BY_LOWER_NAME[normalized_group])
    if len(canonical_source_groups) != len(set(canonical_source_groups)):
        raise ValueError(f"Duplicate SonicMaster degradation groups: {raw_groups!r}")
    if canonical_source_groups and set(canonical_source_groups) != set(names_by_group):
        raise ValueError(
            "SonicMaster degradation groups do not match their specific labels: "
            f"groups={raw_groups!r}, specifics={raw_names!r}"
        )

    groups = [
        group
        for group in SONICMASTER_APPLICATION_GROUP_ORDER
        if group in names_by_group
    ]
    names = [names_by_group[group] for group in groups]
    ids = [_LABEL_TO_ID[name] for name in names]

    tracking_value = record.get("degradation_tracking", {})
    if not isinstance(tracking_value, Mapping):
        raise ValueError("degradation_tracking must be a JSON object")
    tracking = dict(tracking_value)
    tracked_names = []
    for value in tracking.values():
        if isinstance(value, list) and value and isinstance(value[0], str):
            tracked_names.append(value[0].lower())
    if tracked_names and set(tracked_names) != set(raw_names):
        raise ValueError(
            "SonicMaster degradation_tracking does not match specific labels: "
            f"tracked={tracked_names!r}, specifics={raw_names!r}"
        )

    hidden_value = record.get("hidden_clipping", [False, 0])
    if not isinstance(hidden_value, (list, tuple)) or len(hidden_value) != 2:
        raise ValueError("hidden_clipping must be [enabled, level]")
    hidden_clipping = {
        "enabled": bool(hidden_value[0]),
        "level": hidden_value[1],
    }
    return {
        "ids": ids,
        "names": names,
        "params": tracking,
        "order": ids.copy(),
        "groups": groups,
        "compound_level": len(ids),
        # Hidden clipping is deliberately absent from the official prompt and
        # 1-3 label count, but it is still an applied transformation and must
        # remain available for auditing or a future class-mask policy.
        "hidden_clipping": hidden_clipping,
    }


def load_sonicmaster_metadata(
    jsonl_path: str,
    split: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Stream a SonicMaster JSONL file into a compact ``id -> chain`` index."""
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"SonicMaster metadata JSONL not found: {path}")
    expected_split = split.lower() if split else None
    index: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_split = str(record.get("split", "")).lower()
            if expected_split and record_split != expected_split:
                continue
            record_id = str(record.get("id", "")).lower()
            if not record_id:
                raise ValueError(f"Missing id at {path}:{line_number}")
            if record_id in index:
                raise ValueError(f"Duplicate SonicMaster id {record_id!r}")
            index[record_id] = normalize_sonicmaster_record(record)
    if not index:
        raise ValueError(f"No matching SonicMaster records in {path}")
    return index


def pair_metadata_key(clean_path: str, degradation_name: str) -> str:
    """Map official ``source.flac`` or legacy ``source_DT0.flac`` to ``source_degK``."""
    clean_stem = Path(clean_path).stem
    match = _PAIR_PATTERN.fullmatch(clean_stem)
    if match is not None:
        if int(match.group("version")) != 0:
            raise ValueError(f"Expected a _DT0 clean pair, got: {clean_path}")
        source_id = match.group("source")
    else:
        if re.search(r"_(?:dt|deg)\d+$", clean_stem, flags=re.IGNORECASE):
            raise ValueError(f"Expected a clean source file, got: {clean_path}")
        source_id = clean_stem

    degradation_match = re.fullmatch(r"dt(?P<version>\d+)", degradation_name.lower())
    if degradation_match is None:
        raise ValueError(f"Expected degradation name dt<number>, got {degradation_name!r}")
    degradation_version = int(degradation_match.group("version"))
    if degradation_version == 0:
        raise ValueError("The degraded member of a SonicMaster pair cannot be dt0")
    return f"{source_id}_deg{degradation_version}".lower()


__all__ = [
    "SONICMASTER_APPLICATION_GROUP_ORDER",
    "SONICMASTER_DEGRADATION_GROUPS",
    "SONICMASTER_DEGRADATION_LABELS",
    "degradation_multi_hot",
    "load_sonicmaster_metadata",
    "normalize_sonicmaster_record",
    "pair_metadata_key",
]
