import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


DEFAULT_IOU_THRESHOLD = 0.90
MIN_EVENT_LENGTH_FRAMES = 2


@dataclass
class EventConfig:
    out_dir: Path
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    min_event_length: int = MIN_EVENT_LENGTH_FRAMES

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "EventConfig":
        return cls(
            out_dir=Path(args.out),
            iou_threshold=float(args.iou_threshold),
            min_event_length=int(args.min_event_length),
        )


def _load_manifest(out_dir: Path) -> Dict:
    manifest_path = out_dir / "conceptops_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"conceptops_manifest.json not found under {out_dir}. "
            "Run the mask stage first."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(out_dir: Path, manifest: Dict) -> None:
    manifest_path = out_dir / "conceptops_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _load_metadata(metadata_path: Path) -> Dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mask(mask_path: Path) -> np.ndarray:
    """
    Load a binary mask as a boolean numpy array.
    Assumes VideoMask exports 0/255 grayscale PNGs.
    """
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    return arr > 0  # bool mask


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Intersection-over-Union between two boolean masks.
    If union is zero (no foreground in both), treat as IoU = 1.0 (no change).
    """
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)

def _build_events(
    mask_paths: List[Path],
    fps: float,
    iou_threshold: float,
    min_event_length: int,
) -> List[Dict]:
    """
    Robust temporal segmentation:

      - Always returns at least 1 event if there are masks.
      - Starts event 0 at frame 0.
      - For each frame t, compare IoU(mask[t-1], mask[t]).
      - If IoU < threshold => close current event at t-1 and start new one at t.
      - After loop, close the final event at last frame.
      - Filter out very short events; if all are filtered, fall back to [0..N-1].

    """
    if not mask_paths:
        return []

    masks = [_load_mask(p) for p in mask_paths]
    n = len(masks)

    raw_events = []
    current_start = 0

    def make_event(start_idx: int, end_idx: int, event_id: int) -> Dict:
        length = end_idx - start_idx + 1
        mid_idx = (start_idx + end_idx) // 2
        return {
            "event_id": event_id,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "num_frames": length,
            "start_time_sec": start_idx / fps,
            "end_time_sec": (end_idx + 1) / fps,
            "key_frame_index": mid_idx,
            "key_frame_path": str(mask_paths[mid_idx]),
        }

    # Build raw segments
    event_id = 0
    for idx in range(1, n):
        iou = _mask_iou(masks[idx - 1], masks[idx])
        if iou < iou_threshold:
            # close current event at idx-1
            raw_events.append(make_event(current_start, idx - 1, event_id))
            event_id += 1
            current_start = idx

    # close final event
    raw_events.append(make_event(current_start, n - 1, event_id))

    # Filter by min_event_length
    events = [
        ev for ev in raw_events
        if ev["num_frames"] >= min_event_length
    ]

    # Fallback: if everything got filtered out, keep one single full-length event
    if not events:
        events = [make_event(0, n - 1, 0)]

    return events


def run_event_stage(cfg: EventConfig) -> Path:
    """
    Phase 2: temporal event extraction.
    Reads conceptops_manifest.json and metadata.json, writes events.json,
    and updates the manifest.
    """
    out_dir = cfg.out_dir
    manifest = _load_manifest(out_dir)

    metadata_path = Path(manifest["metadata_path"])
    metadata = _load_metadata(metadata_path)

    # Assume VideoMask metadata lists mask paths in order.
    mask_paths = [Path(p) for p in metadata.get("masks", [])]
    if not mask_paths:
        raise ValueError(
            f"No mask paths found in metadata.json at {metadata_path}. "
            "Expected key 'masks'."
        )

    fps = float(manifest.get("fps", manifest.get("config", {}).get("fps", 1.0)))

    print("[ConceptOps] Event stage: computing temporal events")
    print(f"  - masks: {len(mask_paths)} frames")
    print(f"  - fps: {fps}")
    print(f"  - IoU threshold: {cfg.iou_threshold}")
    print(f"  - min_event_length: {cfg.min_event_length}")

    events = _build_events(
        mask_paths=mask_paths,
        fps=fps,
        iou_threshold=cfg.iou_threshold,
        min_event_length=cfg.min_event_length,
    )

    events_path = out_dir / "events.json"
    with events_path.open("w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=2)

    # Update manifest
    manifest.setdefault("stages", {})
    manifest["stages"]["events"] = "completed"
    manifest["events_path"] = str(events_path)
    _save_manifest(out_dir, manifest)

    print(f"[ConceptOps] Wrote events → {events_path}")
    print(f"[ConceptOps] Found {len(events)} events.")
    return events_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptops-events",
        description="ConceptOps temporal event extraction (Phase 2).",
    )
    parser.add_argument(
        "out",
        type=str,
        help="Output directory from the mask stage (where conceptops_manifest.json lives).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for starting a new event (default: 0.90).",
    )
    parser.add_argument(
        "--min-event-length",
        type=int,
        default=MIN_EVENT_LENGTH_FRAMES,
        help="Minimum number of frames per event (default: 2).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = EventConfig.from_args(args)
    run_event_stage(cfg)


if __name__ == "__main__":
    main()
