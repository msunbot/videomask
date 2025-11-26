"""
Simple dummy backend for pipeline development and testing.
Produces a central rectangle mask regardless of input.
"""
from __future__ import annotations
import numpy as np
from .base import BaseSegmentationBackend

class DummyBackend(BaseSegmentationBackend):
    """Central rectangle mask, useful for testing the pipeline without a real model."""

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._is_loaded:
            self.load()

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Simple central rectangle as "object"
        h0, h1 = int(0.3 * h), int(0.7 * h)
        w0, w1 = int(0.3 * w), int(0.7 * w)
        mask[h0:h1, w0:w1] = 1
        return mask