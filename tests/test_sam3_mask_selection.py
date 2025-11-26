"""
tests/test_sam3_mask_selection.py

Tests for the SAM-3 mask selection logic, independent of real SAM-3 models.
"""

from typing import Tuple

import numpy as np
import torch

from videomask.backends.sam3_backend import _safe_select_binary_mask


def _fake_frame_shape() -> Tuple[int, int, int]:
    # arbitrary frame shape (H, W, C)
    return (100, 200, 3)


def test_empty_scores_returns_zero_mask() -> None:
    h, w, _ = _fake_frame_shape()
    # No masks at all
    masks = torch.zeros((0, 1, h, w))
    scores = torch.zeros((0,))

    binary_mask = _safe_select_binary_mask(
        masks_tensor=masks,
        scores_tensor=scores,
        frame_shape=(h, w, 3),
        score_threshold=0.0,
    )

    assert binary_mask.shape == (h, w)
    assert binary_mask.dtype == np.uint8
    assert binary_mask.sum() == 0  # all zeros


def test_highest_score_mask_selected() -> None:
    h, w, _ = _fake_frame_shape()

    # Two fake masks: first is zeros, second is ones
    mask1 = torch.zeros((h, w))
    mask2 = torch.ones((h, w))

    masks = torch.stack([mask1, mask2], dim=0)  # shape (2, H, W)
    scores = torch.tensor([0.1, 0.9])

    binary_mask = _safe_select_binary_mask(
        masks_tensor=masks,
        scores_tensor=scores,
        frame_shape=(h, w, 3),
        score_threshold=0.0,
    )

    assert binary_mask.shape == (h, w)
    # Second mask has higher score; after thresholding it should be all ones.
    assert binary_mask.sum() == h * w


def test_score_threshold_can_zero_out_masks() -> None:
    h, w, _ = _fake_frame_shape()

    # One mask that would otherwise be selected
    mask = torch.ones((1, h, w))
    scores = torch.tensor([0.2])

    binary_mask = _safe_select_binary_mask(
        masks_tensor=mask,
        scores_tensor=scores,
        frame_shape=(h, w, 3),
        score_threshold=0.5,  # higher than 0.2
    )

    # Because score is below threshold, we expect all zeros
    assert binary_mask.shape == (h, w)
    assert binary_mask.sum() == 0