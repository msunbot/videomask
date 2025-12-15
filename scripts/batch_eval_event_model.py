from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from conceptops.types import Episode
from conceptops.labeling.io import load_event_labels
from conceptops.eval.metrics import match_spans_one_to_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate model-backed episode.json events vs event_labels.json.")
    parser.add_argument("--episodes-root", type=str, default="data/episodes")
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--ignore_label", action="store_true")
    args = parser.parse_args()

    root = Path(args.episodes_root)
    episode_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    total_tp = total_fp = total_fn = 0
    n_eval = 0

    for ep_dir in episode_dirs:
        labels_path = ep_dir / "event_labels.json"
        episode_path = ep_dir / "episode.json"

        # Evaluate only labeled clips
        if not labels_path.exists():
            continue
        if not episode_path.exists():
            print(f"[WARN] Missing episode.json in {ep_dir}, skipping")
            continue

        # Load GT spans
        gt_file = load_event_labels(labels_path)
        gts: List[Tuple[int, int, str]] = [(e.start_frame_idx, e.end_frame_idx, e.label) for e in gt_file.labeled_events]

        # Load predicted events from episode.json
        txt = episode_path.read_text(encoding="utf-8").strip()
        if not txt:
            print(f"[WARN] Empty episode.json in {ep_dir}, skipping")
            continue
        ep = Episode.from_json(txt)

        # If your Episode schema stores events differently, adjust here only.
        preds: List[Tuple[int, int, str, float]] = []
        for ev in getattr(ep, "events", []) or []:
            preds.append((ev.start_frame, ev.end_frame, ev.label, float(ev.score or 0.0)))

        res = match_spans_one_to_one(
            preds=preds,
            gts=gts,
            iou_thresh=float(args.iou),
            require_label_match=not args.ignore_label,
        )

        total_tp += res.tp
        total_fp += res.fp
        total_fn += res.fn
        n_eval += 1

        print(
            f"{ep_dir.name}: TP={res.tp} FP={res.fp} FN={res.fn} "
            f"P={res.precision:.2f} R={res.recall:.2f} F1={res.f1:.2f}"
        )

    # Micro-average across clips (summing TP/FP/FN first)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print("\n=== Batch summary ===")
    print(f"Evaluated clips: {n_eval}")
    print(f"Total TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Micro Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()