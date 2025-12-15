from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SpanProposal:
    """
    A candidate event span over frames.

    Convention:
    - start_frame and end_frame are inclusive indices.
    """
    start_frame: int
    end_frame: int


def sliding_window_proposals(
    num_frames: int,
    window_size: int,
    stride: int,
) -> List[SpanProposal]:
    """
    Generate simple sliding-window proposals across the episode.

    This is intentionally dumb but reliable:
    - It guarantees coverage.
    - It’s enough to wire up inference end-to-end.

    Later we can replace proposals with:
    - motion peaks
    - change-point detection
    - learned proposal networks
    """
    if num_frames <= 0:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    props: List[SpanProposal] = []

    # Ensure the last window doesn’t go out of bounds.
    max_start = max(0, num_frames - window_size)
    s = 0
    while s <= max_start:
        e = s + window_size - 1  # inclusive
        props.append(SpanProposal(start_frame=s, end_frame=e))
        s += stride

    # If episode is shorter than window_size, create a single proposal over all frames.
    if not props:
        props.append(SpanProposal(start_frame=0, end_frame=num_frames - 1))

    return props

# Phase 3: motion-guided proposals
def motion_guided_proposals_from_area(
    area_series: list[float],
    window_size: int,
    stride: int,
    topk: int = 10,
) -> List[SpanProposal]:
    """
    Generate proposals centered around peaks in mask area change.

    Why:
    - Many actions correlate with object motion / mask size change.
    - This is much better than uniform sliding windows when data is small.

    Method (simple + explainable):
    1) Compute abs derivative of area_series.
    2) Pick top-K peak indices.
    3) Around each peak, create a window [peak - window/2, peak + window/2].
    4) Clamp to valid range.
    """
    n = len(area_series)
    if n == 0:
        return []
    if n == 1:
        return [SpanProposal(0, 0)]

    # abs derivative
    diffs = [abs(area_series[i] - area_series[i - 1]) for i in range(1, n)]
    # diffs index i corresponds to change ending at frame i
    peak_idxs = sorted(range(1, n), key=lambda i: diffs[i - 1], reverse=True)[: max(1, topk)]

    props: List[SpanProposal] = []
    half = window_size // 2

    for p in peak_idxs:
        s = max(0, p - half)
        e = min(n - 1, s + window_size - 1)
        s = max(0, e - window_size + 1)  # ensure size
        props.append(SpanProposal(start_frame=s, end_frame=e))

    # Optional: also include a few uniform windows for coverage
    props.extend(sliding_window_proposals(num_frames=n, window_size=window_size, stride=stride))

    # Dedup exact duplicates
    uniq = {}
    for sp in props:
        uniq[(sp.start_frame, sp.end_frame)] = sp
    return list(uniq.values())