from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -----------------------------
# Canonical Taxonomy Schema
# -----------------------------
# Keep taxonomy as JSON on disk so we can edit it by hand.
# Also provide a small dataclass wrapper so code stays readable.


@dataclass(frozen=True)
class EventTaxonomy:
    """
    A minimal event taxonomy.

    - version: allows future changes without breaking old label files
    - labels: list of allowed event label strings (e.g. "pick", "place", "open_drawer")
    - descriptions: optional label -> human-readable description
    """
    version: str
    labels: List[str]
    descriptions: Dict[str, str]


# -----------------------------
# Canonical Label File Schema
# -----------------------------
# Store labels next to episode.json as event_labels.json.
#
# Important choice:
# - Label in *frame indices*, not timestamps.
#   Because labeling workflow is frame-based today (inspect frames -> decide start/end).
#   If FPS/timestamps matter later, we can derive them from Episode metadata.


@dataclass(frozen=True)
class LabeledEventSpan:
    """
    One labeled event span in frame coordinates (inclusive indices).

    start_frame_idx, end_frame_idx:
      - inclusive bounds
      - must satisfy 0 <= start <= end < num_frames

    label:
      - must be present in taxonomy.labels

    notes:
      - free text; optional
    """
    start_frame_idx: int
    end_frame_idx: int
    label: str
    notes: Optional[str] = None


@dataclass(frozen=True)
class EventLabelsFile:
    """
    Whole label file stored as JSON.

    We store:
    - schema_version: for format evolution
    - taxonomy_version: which taxonomy version this label file expects
    - episode_id: helpful for sanity checking (folder name)
    - labeled_events: list of LabeledEventSpan
    """
    schema_version: str
    taxonomy_version: str
    episode_id: str
    labeled_events: List[LabeledEventSpan]


# -----------------------------
# Tiny validation helpers
# -----------------------------


def validate_taxonomy(tax: EventTaxonomy) -> None:
    # Basic sanity checks. Keep it simple.
    if not tax.version:
        raise ValueError("taxonomy.version must be non-empty")
    if not tax.labels:
        raise ValueError("taxonomy.labels must be non-empty")
    if len(set(tax.labels)) != len(tax.labels):
        raise ValueError("taxonomy.labels must not contain duplicates")


def validate_labels_file(labels_file: EventLabelsFile, tax: EventTaxonomy, num_frames: int) -> None:
    # Ensure taxonomy versions match (prevents silent mismatches).
    if labels_file.taxonomy_version != tax.version:
        raise ValueError(
            f"Taxonomy version mismatch: labels_file={labels_file.taxonomy_version} taxonomy={tax.version}"
        )

    # Validate each span.
    allowed = set(tax.labels)
    for e in labels_file.labeled_events:
        if e.label not in allowed:
            raise ValueError(f"Unknown label '{e.label}'. Allowed labels: {sorted(allowed)}")
        if e.start_frame_idx < 0 or e.end_frame_idx < 0:
            raise ValueError("Frame indices must be >= 0")
        if e.start_frame_idx > e.end_frame_idx:
            raise ValueError("start_frame_idx must be <= end_frame_idx")
        if e.end_frame_idx >= num_frames:
            raise ValueError(f"end_frame_idx={e.end_frame_idx} out of range for num_frames={num_frames}")