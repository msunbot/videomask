from __future__ import annotations

"""
Simple temporal smoothing utilities for mask sequences.
This is intentionally minimal: it just drops noisy masks based on IoU.
"""

from typing import List

import numpy as np


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.
    """
    assert mask_a.shape == mask_b.shape
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def smooth_masks_sequence(
    masks: List[np.ndarray],
    min_iou_to_keep: float = 0.2,
) -> List[np.ndarray]:
    """
    Very simple temporal smoothing over a sequence of masks.

    If the IoU between current and previous mask is below a threshold,
    we treat the current mask as noise and zero it out.

    Args:
        masks: List of binary masks for consecutive frames.
        min_iou_to_keep: IoU threshold below which a mask is dropped.

    Returns:
        List of smoothed masks (same length as input).
    """
    if not masks:
        return masks

    smoothed: List[np.ndarray] = [masks[0]]

    for i in range(1, len(masks)):
        prev = smoothed[-1]
        curr = masks[i]
        score = iou(prev, curr)
        if score < min_iou_to_keep:
            # Drop transient blip: replace with all-zeros.
            smoothed.append(np.zeros_like(curr))
        else:
            smoothed.append(curr)

    return smoothed