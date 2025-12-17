# conceptops/export/coco.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from conceptops.types import Episode, FrameRecord, InstanceMask


def _compute_bbox_from_mask(mask_path: str) -> Optional[List[float]]:
    """
    Compute COCO-style bounding box [x_min, y_min, width, height]
    from a binary mask.

    Returns None if the mask has no foreground pixels.
    """
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    fg = arr > 0
    if not fg.any():
        return None

    ys, xs = np.where(fg)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [float(x_min), float(y_min), float(width), float(height)]


def export_coco_from_episode(
    episode: Episode,
    out_path: Union[str, Path],
    category_name: str = "object",
) -> str:
    """
    Export a single Episode into a COCO-style annotations JSON.

    Structure:

      {
        "images": [...],
        "annotations": [...],
        "categories": [...],
        "info": {...},
        "licenses": [...],
      }

    We treat each InstanceMask as a COCO annotation with a bbox.
    Segmentation polygons are omitted for now; bbox-only annotations
    are still valid COCO.

    Args:
        episode: Episode to export.
        out_path: Where to write the JSON file.
        category_name: Name of the single category (e.g. "object").

    Returns:
        The path to the written JSON file (string).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []

    category_id = 1
    categories = [
        {
            "id": category_id,
            "name": category_name,
            "supercategory": "object",
        }
    ]

    # Map from frame index to image_id
    image_id = 1
    ann_id = 1

    for frame in episode.frames:
        # Use the basename as COCO image file_name.
        file_name = os.path.basename(frame.image_path)

        # Get image size by opening once. If this is expensive, you can
        # add width/height into FrameRecord later.
        img = Image.open(frame.image_path)
        width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for inst in frame.instances:
            bbox = _compute_bbox_from_mask(inst.mask_path)
            if bbox is None:
                continue

            x_min, y_min, w, h = bbox
            area = float(w * h)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, w, h],
                    "area": area,
                    "iscrowd": 0,
                    # segmentation omitted; bbox-only annotation
                    "segmentation": [],
                }
            )
            ann_id += 1

        image_id += 1

    coco_dict = {
        "info": {
            "description": "Episode exported from ConceptOps",
            "version": "0.1",
            "episode_id": episode.episode_id,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path.write_text(json.dumps(coco_dict, indent=2), encoding="utf-8")
    return str(out_path)