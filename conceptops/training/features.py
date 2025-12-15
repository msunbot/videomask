from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

from conceptops.types import Episode


def _safe_get_area_ratio(frame) -> float:
    """
    Pull a per-frame area_ratio signal from frame.metadata["mask_quality"] if present.

    We keep this resilient: missing fields -> 0.0.
    """
    mq = (frame.metadata or {}).get("mask_quality", {})
    # Prefer "area_ratio" if present; else fall back to mean_area_ratio; else 0.
    if "area_ratio" in mq:
        return float(mq["area_ratio"])
    if "mean_area_ratio" in mq:
        return float(mq["mean_area_ratio"])
    return 0.0


def _frame_diff_signal(frames_paths: List[Path]) -> List[float]:
    """
    Compute a simple motion-ish signal from raw frames:
    mean absolute pixel difference between consecutive frames (grayscale).

    This is not optical flow. It's a cheap baseline feature.
    """
    diffs: List[float] = []
    prev = None

    for p in frames_paths:
        img = cv2.imread(str(p))
        if img is None:
            # If a frame can't be read, treat it as no motion.
            diffs.append(0.0)
            prev = None
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev is None:
            diffs.append(0.0)
        else:
            d = cv2.absdiff(gray, prev)
            diffs.append(float(np.mean(d)) / 255.0)  # normalize to ~[0,1]

        prev = gray

    return diffs


def extract_window_features(
    episode: Episode,
    episode_dir: Path,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    """
    Convert a labeled span [start_frame, end_frame] into a fixed-size feature vector.

    Current feature set (simple but useful):
    - area_ratio mean/std/max/min over the span
    - area_ratio mean absolute derivative (how much mask size changes)
    - frame-diff mean/max over the span (cheap motion proxy)

    This is intentionally "handcrafted baseline"—good enough to validate training plumbing.
    """
    # Defensive bounds
    start = max(0, start_frame)
    end = min(len(episode.frames) - 1, end_frame)
    if end < start:
        end = start

    # 1) area_ratio time series from episode metadata
    areas = np.array([_safe_get_area_ratio(episode.frames[i]) for i in range(start, end + 1)], dtype=np.float32)

    # 2) frame diff signal computed from actual frame images
    frames_dir = episode_dir / "frames_raw"
    # We assume images are named in sorted order; use episode frame paths if present.
    # If your FrameRecord.image_path points to real files, we can prefer that later.
    frame_paths = sorted(list(frames_dir.glob("*.jpg")))
    if not frame_paths:
        frame_paths = sorted(list(frames_dir.glob("*.png")))

    # Only compute diffs for the span
    span_paths = frame_paths[start : end + 1] if frame_paths else []
    diffs = np.array(_frame_diff_signal(span_paths), dtype=np.float32) if span_paths else np.zeros((len(areas),), dtype=np.float32)

    # Aggregate stats
    area_mean = float(np.mean(areas)) if len(areas) else 0.0
    area_std = float(np.std(areas)) if len(areas) else 0.0
    area_min = float(np.min(areas)) if len(areas) else 0.0
    area_max = float(np.max(areas)) if len(areas) else 0.0

    # Mean absolute derivative (captures changes over time)
    if len(areas) >= 2:
        area_mad = float(np.mean(np.abs(np.diff(areas))))
    else:
        area_mad = 0.0

    diff_mean = float(np.mean(diffs)) if len(diffs) else 0.0
    diff_max = float(np.max(diffs)) if len(diffs) else 0.0

    # Fixed feature vector
    feats = np.array([area_mean, area_std, area_min, area_max, area_mad, diff_mean, diff_max], dtype=np.float32)
    return feats