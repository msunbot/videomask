# scripts/batch_predict_events.py
"""
Batch re-generate predictions for episodes under a root directory.

Writes pred_events.json for each episode directory containing episode.json
that parses as conceptops.types.Episode.

New: if inference_profile is demo_clean/demo_clean_v2, print a short summary:
- events per episode (avg/median/min/max)
- empty episodes count
- overlap violations (should be 0)

Usage:
  python scripts/batch_predict_events.py \
    --episodes_root data/episodes \
    --model_dir data/models/event_model_demo3_wt \
    --inference_profile demo_clean_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from conceptops.core.events import ModelEventDetector


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def overlaps_strict(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    # touching endpoints is OK
    return (a_s < b_e) and (b_s < a_e)


def count_overlap_violations(events: List[Dict[str, Any]]) -> int:
    """
    Count overlapping span pairs within a single episode.
    Assumes events have start_frame/end_frame.
    """
    v = 0
    spans = [(int(e["start_frame"]), int(e["end_frame"])) for e in events if "start_frame" in e and "end_frame" in e]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if overlaps_strict(spans[i][0], spans[i][1], spans[j][0], spans[j][1]):
                v += 1
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_root", type=str, default="data/episodes")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--inference_profile", type=str, default="demo_clean_v2")
    ap.add_argument("--window_size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--topk", type=int, default=5)         # base; demo profiles override inside detector
    ap.add_argument("--min_score", type=float, default=0.55)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.episodes_root)
    if not root.exists():
        raise FileNotFoundError(f"episodes_root not found: {root}")

    from conceptops.types import Episode  # local import

    detector = ModelEventDetector(
        model_dir=str(args.model_dir),
        window_size=int(args.window_size),
        stride=int(args.stride),
        topk=int(args.topk),
        min_score=float(args.min_score),
        nms_iou=float(args.nms_iou),
        inference_profile=str(args.inference_profile),
    )

    episode_paths = sorted(root.rglob("episode.json"))
    if not episode_paths:
        print(f"No episode.json found under: {root}")
        return

    events_per_ep: List[int] = []
    empty_eps = 0
    overlap_violations_total = 0

    written = 0
    skipped = 0

    for ep_path in episode_paths:
        ep_dir = ep_path.parent

        try:
            episode = Episode.from_json(ep_path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        events = detector.detect(frame_records=episode.frames, episode_dir=str(ep_dir))

        out: List[Dict[str, Any]] = []
        for ev in events:
            if hasattr(ev, "to_dict"):
                out.append(ev.to_dict())
            else:
                out.append(dict(ev.__dict__))

        save_json(ep_dir / "pred_events.json", out)
        written += 1

        events_per_ep.append(len(out))
        if len(out) == 0:
            empty_eps += 1

        overlap_violations_total += count_overlap_violations(out)

    print(f"\nWrote pred_events.json for {written} episodes (skipped {skipped} non-ConceptOps schemas).")

    # ---- NEW: demo summary ----
    prof = (args.inference_profile or "").lower().strip()
    if prof in ("demo", "demo_clean", "clean", "demo_clean_v2", "demo2", "clean2"):
        if events_per_ep:
            sorted_counts = sorted(events_per_ep)
            avg = sum(events_per_ep) / len(events_per_ep)
            median = sorted_counts[len(sorted_counts) // 2]
            mn = sorted_counts[0]
            mx = sorted_counts[-1]
            total_events = sum(events_per_ep)

            print("\n=== DEMO PREDICTION SUMMARY ===")
            print(f"profile            : {args.inference_profile}")
            print(f"episodes predicted : {len(events_per_ep)}")
            print(f"total events       : {total_events}")
            print(f"events/episode avg : {avg:.2f}")
            print(f"events/episode med : {median}")
            print(f"events/episode min : {mn}")
            print(f"events/episode max : {mx}")
            print(f"empty episodes     : {empty_eps}")
            print(f"overlap violations : {overlap_violations_total}  (should be 0 for demo profiles)")
            print("=== END SUMMARY ===\n")


if __name__ == "__main__":
    main()