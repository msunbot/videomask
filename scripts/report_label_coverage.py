# scripts/report_label_coverage.py
"""
Label coverage report for ConceptOps Phase 3.

Your ground-truth schema (per screenshot) is:
event_labels.json:
{
  "episode_id": "...",
  "labeled_events": [
     {"start_frame_idx": 1, "end_frame_idx": 2, "label": "move", "notes": null},
     ...
  ]
}

This script scans episode subfolders under --episodes_root and aggregates:
- counts per label
- span length stats (in frames)
- overlap frequency within an episode
- recommended next labels to annotate (lowest coverage)

Usage:
  python scripts/report_label_coverage.py --episodes_root data/episodes
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_int(d: Dict[str, Any], keys: List[str]) -> Optional[int]:
    """
    Try a list of possible keys and return the first that exists as int.
    """
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                pass
    return None


def normalize_gt_spans(label_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize ground-truth event labels into:
      {"start": int, "end": int, "label": str}
    Accepts multiple possible formats:
    - labeled_events with start_frame_idx/end_frame_idx
    - spans/events with start/end
    """
    raw = (
        label_json.get("labeled_events")
        or label_json.get("spans")
        or label_json.get("events")
        or []
    )

    spans: List[Dict[str, Any]] = []
    for s in raw:
        start = _get_int(s, ["start_frame_idx", "start_frame", "start"])
        end = _get_int(s, ["end_frame_idx", "end_frame", "end"])
        label = s.get("label") or s.get("name") or "UNKNOWN"
        if start is None or end is None:
            # Skip malformed spans, but do not crash the report.
            continue
        spans.append({"start": start, "end": end, "label": str(label)})
    return spans


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    # Strict overlap (not just touching)
    return (a_start < b_end) and (b_start < a_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes_root",
        type=str,
        required=True,
        help="Root directory containing episode subfolders (each contains event_labels.json).",
    )
    args = ap.parse_args()

    root = Path(args.episodes_root)
    if not root.exists():
        raise FileNotFoundError(f"episodes_root not found: {root}")

    label_files = sorted(root.rglob("event_labels.json"))
    if not label_files:
        print(f"No event_labels.json found under: {root}")
        return

    label_counts: Dict[str, int] = {}
    span_lengths: List[float] = []  # in frames
    per_episode_overlap_counts: List[int] = []
    total_spans = 0
    total_episodes_with_labels = 0

    for lf in label_files:
        data = load_json(lf)
        spans = normalize_gt_spans(data)

        if not spans:
            continue

        total_episodes_with_labels += 1
        total_spans += len(spans)

        # counts + lengths
        for s in spans:
            label = s["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            span_lengths.append(max(0.0, float(s["end"]) - float(s["start"])))

        # overlap frequency within the episode
        overlaps = 0
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if overlap(spans[i]["start"], spans[i]["end"], spans[j]["start"], spans[j]["end"]):
                    overlaps += 1
        per_episode_overlap_counts.append(overlaps)

    # ---- Print report ----
    print("\n=== LABEL COVERAGE REPORT ===")
    print(f"Episodes root: {root}")
    print(f"Labeled episodes: {total_episodes_with_labels}")
    print(f"Total labeled spans: {total_spans}")

    print("\n-- Counts per label --")
    if not label_counts:
        print("(none found)")
    else:
        for k in sorted(label_counts.keys()):
            print(f"{k:>12}: {label_counts[k]}")

    if span_lengths:
        print("\n-- Span length stats (frames) --")
        print(f"mean   : {mean(span_lengths):.3f}")
        print(f"median : {median(span_lengths):.3f}")
        print(f"min/max: {min(span_lengths):.3f} / {max(span_lengths):.3f}")

    if per_episode_overlap_counts:
        overlap_eps = sum(1 for x in per_episode_overlap_counts if x > 0)
        print("\n-- Overlap frequency --")
        print(f"Episodes with >=1 overlap pair: {overlap_eps} / {len(per_episode_overlap_counts)}")
        print(f"Total overlap pairs (sum): {sum(per_episode_overlap_counts)}")

    # recommend next labels: lowest counts
    print("\n-- Recommended next labels to annotate (lowest coverage) --")
    if not label_counts:
        print("(none)")
    else:
        items = sorted(label_counts.items(), key=lambda kv: kv[1])
        for label, cnt in items[:6]:
            print(f"{label:>12}: {cnt}  <-- label more of these")

    print("\nDone.\n")


if __name__ == "__main__":
    main()