from __future__ import annotations
"""
SAM-3 backend implementation.

This wraps the SAM-3 image model and processor into the BaseSegmentationBackend
interface used by the VideoMask pipeline.

Notes:
- Requires CUDA-enabled PyTorch and the `sam3` library.
- Intended to be run in a GPU environment (e.g. Colab, remote GPU).
"""
from typing import Any, Optional, Tuple

import numpy as np

from .base import BaseSegmentationBackend


class SAM3NotAvailableError(RuntimeError):
    """Raised when SAM-3 or its dependencies are not available."""


def _safe_select_binary_mask(
    masks_tensor,
    scores_tensor,
    frame_shape: Tuple[int, int, int],
    score_threshold: float = 0.0,
) -> np.ndarray:
    """
    Core mask-selection logic.

    Converts SAM-3 outputs into a binary mask (H, W) with values {0,1}.
    Handles empty outputs safely by returning an all-zero mask.

    Args:
        masks_tensor: torch.Tensor of shape (N, H, W) or (N, 1, H, W)
        scores_tensor: torch.Tensor of shape (N,)
        frame_shape: shape of the input frame (H, W, C)
        score_threshold: optional min score to accept a mask

    Returns:
        binary_mask: np.ndarray of shape (H, W), dtype uint8, values {0,1}.
    """
    import torch  # local import to avoid hard dependency at import time

    scores_np = scores_tensor.detach().cpu().numpy()

    # Case 1: no masks returned
    if scores_np.size == 0:
        h, w, _ = frame_shape
        return np.zeros((h, w), dtype=np.uint8)

    # Pick the highest-scoring mask
    best_idx = int(scores_np.argmax())
    best_score = float(scores_np[best_idx])

    if best_score < score_threshold:
        # If we care about thresholding, we can drop low-confidence masks.
        h, w, _ = frame_shape
        return np.zeros((h, w), dtype=np.uint8)

    # Handle possible shapes: (N, 1, H, W) or (N, H, W)
    if masks_tensor.dim() == 4:
        # (N, 1, H, W) -> (N, H, W)
        masks_2d = masks_tensor[:, 0]
    else:
        masks_2d = masks_tensor

    mask_2d = masks_2d[best_idx].detach().cpu().numpy()  # (H, W)

    # Threshold to binary
    binary_mask = (mask_2d > 0.5).astype(np.uint8)
    return binary_mask


class SAM3Backend(BaseSegmentationBackend):
    """
    SAM-3 segmentation backend.

    Usage:
        backend = SAM3Backend(
            device="cuda",
            text_prompt="person",
            score_threshold=0.0,
        )

        backend.load()
        mask = backend.segment_frame(frame_np)

    Args:
        device: "cuda", "cpu", or "mps" (GPU strongly recommended).
        text_prompt: text description for what to segment (e.g. "person").
        score_threshold: optional minimum score for accepting a mask.
    """

    def __init__(
        self,
        device: str = "cuda",
        text_prompt: str = "person",
        score_threshold: float = 0.0,
        **config: Any,
    ) -> None:
        super().__init__(
            device=device,
            text_prompt=text_prompt,
            score_threshold=score_threshold,
            **config,
        )
        self.device = device
        self.text_prompt = text_prompt
        self.score_threshold = score_threshold

        self.model = None
        self.processor = None

    def load(self) -> None:
        """
        Lazily load the SAM-3 model and processor.

        This method:
        - Imports the SAM-3 library
        - Builds the image model
        - Creates an associated processor
        - Moves the model to the requested device (if applicable)
        """
        if self._is_loaded:
            return

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import torch
        except ImportError as e:
            raise SAM3NotAvailableError(
                "SAM-3 backend requires `sam3` and `torch` to be installed. "
                "Install them in a GPU environment (e.g. Colab) before using this backend."
            ) from e

        # Build model + processor
        self.model = build_sam3_image_model()

        # Move model to device if possible
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        elif self.device in ("cuda", "mps") and not torch.cuda.is_available():
            print(f"[SAM3Backend] Requested device '{self.device}' is not available. "
                  "Falling back to CPU. This may be slow.")
        # NOTE: for "mps", SAM-3 may or may not support it; CPU fallback is safest.

        self.processor = Sam3Processor(self.model)
        self._is_loaded = True

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run SAM-3 on a single RGB frame and return a binary mask.

        Args:
            frame: numpy array (H, W, 3), dtype uint8

        Returns:
            binary_mask: numpy array (H, W), dtype uint8, values {0,1}
        """
        if not self._is_loaded:
            self.load()

        if self.processor is None:
            raise SAM3NotAvailableError(
                "SAM-3 processor is not initialized. Did `load()` fail?"
            )

        from PIL import Image

        # Convert frame to PIL.Image
        frame_img = Image.fromarray(frame.astype("uint8"), mode="RGB")

        # Set image state in processor
        state = self.processor.set_image(frame_img)

        # Apply text prompt
        output = self.processor.set_text_prompt(
            state=state,
            prompt=self.text_prompt,
        )

        masks = output["masks"]   # torch.Tensor
        scores = output["scores"] # torch.Tensor

        # Use shared selection helper
        binary_mask = _safe_select_binary_mask(
            masks_tensor=masks,
            scores_tensor=scores,
            frame_shape=frame.shape,
            score_threshold=self.score_threshold,
        )

        return binary_mask