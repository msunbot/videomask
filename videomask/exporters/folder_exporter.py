from __future__ import annotations

"""
Exporter for a simple "folder dataset" format.

Structure:
    out_dir/
        frames_raw/
        masks/
        metadata.json
"""

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
from PIL import Image


def save_masks(
    masks: List[np.ndarray],
    frame_paths: List[str],
    masks_dir: str,
) -> List[str]:
    """
    Save binary masks to disk as PNG images.

    Args:
        masks: List of masks corresponding to each input frame.
        frame_paths: List of frame image paths (same length as masks).
        masks_dir: Directory to write mask PNGs into.

    Returns:
        List of mask file paths.
    """
    masks_dir_path = Path(masks_dir)
    masks_dir_path.mkdir(parents=True, exist_ok=True)

    mask_paths: List[str] = []
    for frame_path, mask in zip(frame_paths, masks):
        mask_img = Image.fromarray((mask * 255).astype("uint8"))
        mask_name = Path(frame_path).name.replace("frame_", "mask_")
        mask_path = masks_dir_path / mask_name
        mask_img.save(mask_path)
        mask_paths.append(str(mask_path))

    return mask_paths


def write_metadata(
    out_dir: str,
    frame_paths: List[str],
    mask_paths: List[str],
    config: Dict[str, Any],
) -> None:
    """
    Write a simple metadata.json file describing the dataset.

    Args:
        out_dir: Root output directory.
        frame_paths: List of all frame file paths.
        mask_paths: List of all corresponding mask file paths.
        config: Configuration used for this run (backend, fps, etc.).
    """
    out_dir_path = Path(out_dir)
    meta = {
        "frames": frame_paths,
        "masks": mask_paths,
        "config": config,
    }
    with open(out_dir_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)