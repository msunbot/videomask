from __future__ import annotations

import json
from pathlib import Path
from typing import List

from conceptops.types import EventRecord  # <-- uses existing canonical schema
from conceptops.labeling.schemas import (
    EventTaxonomy,
    EventLabelsFile,
    LabeledEventSpan,
    validate_taxonomy,
    validate_labels_file,
)


DEFAULT_SCHEMA_VERSION = "1.0"


# -----------------------------
# Taxonomy I/O
# -----------------------------


def load_taxonomy(taxonomy_path: Path) -> EventTaxonomy:
    """
    Load event taxonomy JSON from disk.

    We keep this strict-ish so labeling doesn't silently drift.
    """
    data = json.loads(taxonomy_path.read_text())

    tax = EventTaxonomy(
        version=str(data["version"]),
        labels=list(data["labels"]),
        descriptions=dict(data.get("descriptions", {})),
    )
    validate_taxonomy(tax)
    return tax


def save_taxonomy(taxonomy_path: Path, tax: EventTaxonomy) -> None:
    """
    Save taxonomy in a stable, human-editable format.
    """
    validate_taxonomy(tax)
    payload = {
        "version": tax.version,
        "labels": tax.labels,
        "descriptions": tax.descriptions,
    }
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_path.write_text(json.dumps(payload, indent=2, sort_keys=False))


# -----------------------------
# Label file I/O
# -----------------------------


def load_event_labels(labels_path: Path) -> EventLabelsFile:
    """
    Load event_labels.json into an EventLabelsFile dataclass.
    """
    data = json.loads(labels_path.read_text())

    labeled = []
    for item in data.get("labeled_events", []):
        labeled.append(
            LabeledEventSpan(
                start_frame_idx=int(item["start_frame_idx"]),
                end_frame_idx=int(item["end_frame_idx"]),
                label=str(item["label"]),
                notes=item.get("notes", None),
            )
        )

    return EventLabelsFile(
        schema_version=str(data["schema_version"]),
        taxonomy_version=str(data["taxonomy_version"]),
        episode_id=str(data["episode_id"]),
        labeled_events=labeled,
    )


def save_event_labels(labels_path: Path, labels_file: EventLabelsFile) -> None:
    """
    Save label file as JSON.

    Keep JSON pretty-printed so we can inspect diffs in git later (Phase 5–6),
    but without *building* a full SoR system yet.
    """
    payload = {
        "schema_version": labels_file.schema_version,
        "taxonomy_version": labels_file.taxonomy_version,
        "episode_id": labels_file.episode_id,
        "labeled_events": [
            {
                "start_frame_idx": e.start_frame_idx,
                "end_frame_idx": e.end_frame_idx,
                "label": e.label,
                "notes": e.notes,
            }
            for e in labels_file.labeled_events
        ],
    }

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def init_empty_labels_file(
    labels_path: Path,
    episode_id: str,
    taxonomy: EventTaxonomy,
) -> EventLabelsFile:
    """
    Create an empty event_labels.json if it doesn't exist yet.

    This is intentionally explicit: the first time labeling a clip, this bootstraps the file.
    """
    labels_file = EventLabelsFile(
        schema_version=DEFAULT_SCHEMA_VERSION,
        taxonomy_version=taxonomy.version,
        episode_id=episode_id,
        labeled_events=[],
    )
    save_event_labels(labels_path, labels_file)
    return labels_file


# -----------------------------
# Conversion: labels -> EventRecord
# -----------------------------


def labeled_spans_to_event_records(
    spans: List[LabeledEventSpan],
    *,
    source: str = "manual",
    starting_event_id: int = 0,
) -> List[EventRecord]:
    """
    Convert labeled frame spans into the canonical EventRecord objects.

    Your real EventRecord schema is:

      event_id: int
      label: str
      start_frame: int   (inclusive)
      end_frame: int     (inclusive)
      score: Optional[float]
      metadata: Dict[str, Any]

    Design choice:
    - Generate event_id deterministically in order (0..N-1) by default.
      This is good enough for Phase 3. Later, Phase 5–6 can assign stable IDs.

    - Place provenance info (manual vs model) in metadata, not a separate field,
      because your EventRecord doesn't have `source`.
    """
    records: List[EventRecord] = []

    for i, s in enumerate(spans):
        event_id = starting_event_id + i

        # Store extra info in metadata so we don't mutate the core schema.
        md = {
            "source": source,       # manual labeling provenance
            "notes": s.notes,       # keep notes, but don't force a top-level field
        }

        records.append(
            EventRecord(
                event_id=event_id,
                label=s.label,
                start_frame=s.start_frame_idx,  # inclusive
                end_frame=s.end_frame_idx,      # inclusive 
                score=None,                     # manual labels have no confidence score
                metadata=md,
            )
        )

    return records


def load_labeled_event_records(
    episode_dir: Path,
    taxonomy_path: Path,
    *,
    num_frames: int,
    labels_filename: str = "event_labels.json",
) -> List[EventRecord]:
    """
    High-level helper used later by training/eval:
      episode_dir/
        episode.json
        frames_raw/
        event_labels.json   <-- this file

    Returns EventRecord[] in the canonical schema.
    """
    tax = load_taxonomy(taxonomy_path)

    labels_path = episode_dir / labels_filename
    if not labels_path.exists():
        # Not every episode is labeled yet; this is normal.
        return []

    labels_file = load_event_labels(labels_path)

    # Validate spans vs taxonomy + frame bounds.
    validate_labels_file(labels_file, tax, num_frames=num_frames)

    return labeled_spans_to_event_records(labels_file.labeled_events, source="manual")