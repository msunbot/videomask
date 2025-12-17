from pathlib import Path
import json

from conceptops.labeling.io import (
    load_taxonomy,
    load_event_labels,
    save_event_labels,
)
from conceptops.labeling.schemas import EventLabelsFile, LabeledEventSpan


def test_event_labels_roundtrip(tmp_path: Path):
    # Create a fake taxonomy file
    taxonomy_path = tmp_path / "event_taxonomy.json"
    taxonomy_path.write_text(json.dumps({
        "version": "1.0",
        "labels": ["pick", "place"],
        "descriptions": {"pick": "pick up", "place": "put down"},
    }))

    tax = load_taxonomy(taxonomy_path)
    assert tax.version == "1.0"
    assert "pick" in tax.labels

    labels_path = tmp_path / "event_labels.json"

    labels_file = EventLabelsFile(
        schema_version="1.0",
        taxonomy_version=tax.version,
        episode_id="clip_001",
        labeled_events=[
            LabeledEventSpan(start_frame_idx=10, end_frame_idx=20, label="pick", notes="test"),
        ],
    )

    save_event_labels(labels_path, labels_file)

    loaded = load_event_labels(labels_path)
    assert loaded.episode_id == "clip_001"
    assert len(loaded.labeled_events) == 1
    assert loaded.labeled_events[0].label == "pick"
    assert loaded.labeled_events[0].start_frame_idx == 10
    assert loaded.labeled_events[0].end_frame_idx == 20