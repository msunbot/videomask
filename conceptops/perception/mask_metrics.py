# conceptops/perception/mask_metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image


@dataclass
class MaskStats:
    """
    Simple quality stats for a binary (or soft) mask image.

    For now we focus on:
      - area_px: number of non-zero pixels.
      - area_ratio: fraction of pixels that are non-zero (0..1).

    Later we can extend this with:
      - bounding boxes
      - connected component counts
      - shape descriptors
    """
    area_px: int
    area_ratio: float


def compute_mask_stats(mask_path: str) -> MaskStats:
    """
    Compute simple quality metrics for a single mask image.

    We treat any non-zero pixel as foreground.

    Args:
        mask_path: Path to a mask image (binary or grayscale).

    Returns:
        MaskStats with area_px and area_ratio.

    Raises:
        FileNotFoundError: if the mask image cannot be opened.
    """
    img = Image.open(mask_path).convert("L")  # grayscale
    arr = np.array(img)

    total_px = arr.size
    if total_px == 0:
        return MaskStats(area_px=0, area_ratio=0.0)

    # Consider any pixel >0 as foreground.
    fg = arr > 0
    area_px = int(fg.sum())
    area_ratio = float(area_px) / float(total_px)

    return MaskStats(area_px=area_px, area_ratio=area_ratio)