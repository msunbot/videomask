"""
Base class for segmentation backends.
A backend takes an image (H, W, 3) and returns a binary mask (H, W),
where mask values are 0/1.
"""

from __future__ import annotations
from typing import Any
import numpy as np

class BaseSegmentationBackend:
    """
    Common interface for all segmentation backends.

    Subclasses should:
      - Implement `load` to lazily load heavy models.
      - Implement `segment_frame` for a single image.
    """

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._is_loaded = False

    def load(self) -> None:
        """
        Load model weights or perform any heavy initialization.

        Called lazily on first use.
        """
        self._is_loaded = True

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment a single frame.

        Args:
            frame: Input RGB image as numpy array (H, W, 3).

        Returns:
            Binary mask as numpy array (H, W) with dtype uint8 and values {0, 1}.
        """
        raise NotImplementedError