from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2  # OpenCV is the simplest dependency for frame reading.

from conceptops.labeling.io import (
    init_empty_labels_file,
    load_event_labels,
    load_taxonomy,
    save_event_labels,
)
from conceptops.labeling.schemas import LabeledEventSpan


def _frames_dir(episode_dir: Path) -> Path:
    # Your pipeline outputs frames_raw/ in the episode dir.
    return episode_dir / "frames_raw"


def _labels_path(episode_dir: Path) -> Path:
    return episode_dir / "event_labels.json"


def _read_frame_bgr(frame_path: Path):
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {frame_path}")
    return img


def _print_usage() -> None:
    print("\nCommands:")
    print("  n                -> next frame")
    print("  p                -> previous frame")
    print("  j <k>            -> jump to frame index k (e.g. j 120)")
    print("  mark_start       -> mark current frame as span start")
    print("  mark_end         -> mark current frame as span end")
    print("  add <label>      -> add labeled span using marked start/end")
    print("  list             -> list current labeled spans")
    print("  save             -> write event_labels.json to disk")
    print("  q                -> quit\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal terminal labeling for ConceptOps episodes.")
    parser.add_argument("--episode-dir", type=str, required=True, help="Episode directory (e.g. data/episodes/clip_001).")
    parser.add_argument("--taxonomy", type=str, default="config/event_taxonomy.json", help="Path to taxonomy JSON.")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    taxonomy_path = Path(args.taxonomy)

    frames_dir = _frames_dir(episode_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_raw/ not found in {episode_dir}. Did you run batch_build_episodes.py?")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        # If your pipeline writes png, extend this list.
        frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} (expected .jpg or .png).")

    num_frames = len(frame_paths)
    print(f"Loaded {num_frames} frames from {frames_dir}")

    tax = load_taxonomy(taxonomy_path)
    print(f"Loaded taxonomy v{tax.version} with {len(tax.labels)} labels.")
    print("Allowed labels:", ", ".join(tax.labels))

    labels_path = episode_dir / "event_labels.json"
    episode_id = episode_dir.name

    # Create label file if missing.
    if not labels_path.exists():
        labels_file = init_empty_labels_file(labels_path, episode_id=episode_id, taxonomy=tax)
        print(f"Initialized empty labels file: {labels_path}")
    else:
        labels_file = load_event_labels(labels_path)
        print(f"Loaded existing labels file: {labels_path} ({len(labels_file.labeled_events)} spans)")

    idx = 0
    marked_start: Optional[int] = None
    marked_end: Optional[int] = None

    # We use OpenCV imshow for simplicity: it's not pretty, but it works everywhere.
    # This is intentionally NOT a full notebook UI yet.
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    _print_usage()

    while True:
        # Show the current frame.
        fp = frame_paths[idx]
        img = _read_frame_bgr(fp)

        # Overlay text with frame index + filename for quick sanity.
        overlay = f"frame_idx={idx}  file={fp.name}"
        cv2.putText(img, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show current marks if set.
        mark_txt = f"start={marked_start} end={marked_end}"
        cv2.putText(img, mark_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("frame", img)
        cv2.waitKey(1)  # Let OpenCV update the window.

        cmd = input(f"[{episode_id}] idx={idx} > ").strip()

        if cmd in {"q", "quit", "exit"}:
            break

        if cmd == "n":
            idx = min(idx + 1, num_frames - 1)
            continue

        if cmd == "p":
            idx = max(idx - 1, 0)
            continue

        if cmd.startswith("j "):
            try:
                k = int(cmd.split(" ", 1)[1])
                idx = max(0, min(k, num_frames - 1))
            except ValueError:
                print("Invalid jump. Use: j <int>")
            continue

        if cmd == "mark_start":
            marked_start = idx
            print(f"Marked start at frame {marked_start}")
            continue

        if cmd == "mark_end":
            marked_end = idx
            print(f"Marked end at frame {marked_end}")
            continue

        if cmd.startswith("add "):
            label = cmd.split(" ", 1)[1].strip()
            if label not in set(tax.labels):
                print(f"Unknown label '{label}'. Allowed: {tax.labels}")
                continue
            if marked_start is None or marked_end is None:
                print("You must set both marked_start and marked_end before adding.")
                continue

            s = min(marked_start, marked_end)
            e = max(marked_start, marked_end)

            # Append span.
            new_span = LabeledEventSpan(
                start_frame_idx=s,
                end_frame_idx=e,
                label=label,
                notes=None,
            )
            labels_file = labels_file.__class__(
                schema_version=labels_file.schema_version,
                taxonomy_version=labels_file.taxonomy_version,
                episode_id=labels_file.episode_id,
                labeled_events=[*labels_file.labeled_events, new_span],
            )
            print(f"Added span: [{s}, {e}] label={label}")
            continue

        if cmd == "list":
            if not labels_file.labeled_events:
                print("(no spans yet)")
            for i, sp in enumerate(labels_file.labeled_events):
                print(f"{i:02d}: [{sp.start_frame_idx}, {sp.end_frame_idx}] {sp.label}  notes={sp.notes}")
            continue

        if cmd == "save":
            save_event_labels(labels_path, labels_file)
            print(f"Saved: {labels_path} ({len(labels_file.labeled_events)} spans)")
            continue

        print("Unknown command.")
        _print_usage()

    # On exit, don't force-save: explicit is better (prevents accidental overwrites).
    cv2.destroyAllWindows()
    print("Exited labeling tool (no auto-save).")


if __name__ == "__main__":
    main()