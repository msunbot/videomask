from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptops.eval.metrics import match_spans_one_to_one
from conceptops.labeling.io import load_event_labels, load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pred_events.json against event_labels.json for one episode.")
    parser.add_argument("--episode-dir", type=str, required=True)
    parser.add_argument("--taxonomy", type=str, default="conceptops/config/event_taxonomy.json")
    parser.add_argument("--pred", type=str, default="pred_events.json")
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--ignore_label", action="store_true", help="If set, do not require label match (span-only).")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    pred_path = episode_dir / args.pred
    labels_path = episode_dir / "event_labels.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    # Load predictions produced by run_event_model_inference.py
    pred_payload = json.loads(pred_path.read_text())
    preds = []
    for item in pred_payload:
        preds.append((
            int(item["start_frame"]),
            int(item["end_frame"]),
            str(item["label"]),
            float(item.get("score", 0.0)),
        ))

    # Load GT labels
    # (We load taxonomy mainly to ensure label vocabulary is stable; optional for now)
    tax = load_taxonomy(Path(args.taxonomy))
    gt_file = load_event_labels(labels_path)
    gts = [(e.start_frame_idx, e.end_frame_idx, e.label) for e in gt_file.labeled_events]

    res = match_spans_one_to_one(
        preds=preds,
        gts=gts,
        iou_thresh=float(args.iou),
        require_label_match=not args.ignore_label,
    )

    print(f"Episode: {episode_dir.name}")
    print(f"Pred spans: {len(preds)} | GT spans: {len(gts)}")
    print(f"IoU thresh: {args.iou} | label_match: {not args.ignore_label}")
    print(f"TP={res.tp} FP={res.fp} FN={res.fn}")
    print(f"Precision={res.precision:.3f} Recall={res.recall:.3f} F1={res.f1:.3f}")


if __name__ == "__main__":
    main()