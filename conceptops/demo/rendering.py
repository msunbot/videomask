"""
Rendering helpers for Phase 4 demo.

We keep these as pure helpers so:
- the Streamlit app stays readable
- we avoid polluting core pipeline code
- we can later reuse these utilities for Phase 5 launch assets (screenshots, short clips, etc.)

Assumptions about episode dir structure (Phase 3 outputs):
- frames_raw/           (image files, e.g. 000001.jpg or similar)
- masks/                (mask images, matching frame ids)
- pred_events.json       (list of spans with label, start, end, score)
- episode.json           (metadata)

Important: file naming conventions may differ.
We use a "best effort" approach:
- sort by filename
- treat index N as the Nth sorted frame
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go


def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_pred_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    return data if isinstance(data, list) else []


def _sorted_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file()])


def _event_start_end(event: Dict[str, Any]) -> Tuple[int, int]:
    """Support both (start_frame/end_frame) and legacy (start/end)."""
    s = event.get("start_frame", event.get("start", 0))
    e = event.get("end_frame", event.get("end", s))
    try:
        s = int(s)
        e = int(e)
    except Exception:
        s, e = 0, 0
    if e < s:
        s, e = e, s
    return s, e


def _get_frame_and_mask_paths(episode_dir: Path, frame_idx: int) -> tuple[Optional[Path], Optional[Path]]:
    frames_dir = episode_dir / "frames_raw"
    masks_dir = episode_dir / "masks"

    frames = _sorted_files(frames_dir)
    masks = _sorted_files(masks_dir)

    if not frames:
        return None, None

    frame_idx = max(0, min(frame_idx, len(frames) - 1))
    frame_path = frames[frame_idx]

    mask_path = None
    if masks:
        stem = frame_path.stem
        same_stem = [m for m in masks if m.stem == stem]
        if same_stem:
            mask_path = same_stem[0]
        elif frame_idx < len(masks):
            mask_path = masks[frame_idx]

    return frame_path, mask_path


def overlay_frame_with_mask(
    episode_dir: Path,
    frame_idx: int,
    overlay_alpha: float = 0.55,
) -> Optional[Image.Image]:
    frame_path, mask_path = _get_frame_and_mask_paths(episode_dir, frame_idx)
    if frame_path is None or not frame_path.exists():
        return None

    frame = Image.open(frame_path).convert("RGB")

    if mask_path is None or not mask_path.exists():
        return frame

    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    fg = (mask_np > 0).astype(np.uint8)

    overlay = Image.new("RGB", frame.size, (255, 0, 0))
    overlay_np = np.array(overlay).astype(np.float32)
    frame_np = np.array(frame).astype(np.float32)
    fg3 = np.repeat(fg[:, :, None], 3, axis=2).astype(np.float32)

    out = frame_np * (1.0 - fg3 * overlay_alpha) + overlay_np * (fg3 * overlay_alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def build_event_timeline_plotly(events: List[Dict[str, Any]], total_frames: int) -> go.Figure:
    fig = go.Figure()

    if not events:
        fig.update_layout(
            height=250,
            title="No predicted events",
            xaxis_title="Frame",
            yaxis_title="Event",
        )
        return fig

    for i, e in enumerate(events):
        start, end = _event_start_end(e)
        label = str(e.get("label", "unknown"))
        score = e.get("score", None)

        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[i, i],
                mode="lines",
                line=dict(width=10),
                hovertemplate=(
                    f"label: {label}<br>"
                    f"start_frame: {start}<br>"
                    f"end_frame: {end}<br>"
                    f"score: {score}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=min(450, 80 + 35 * len(events)),
        xaxis_title="Frame index",
        yaxis_title="Event index",
        xaxis=dict(range=[0, max(total_frames - 1, 1)]),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def render_event_slice_gallery(
    episode_dir: Path,
    event: Dict[str, Any],
    max_frames: int = 12,
) -> None:
    import streamlit as st  # local import

    start, end = _event_start_end(event)
    span = max(1, end - start + 1)

    if span <= max_frames:
        idxs = list(range(start, end + 1))
    else:
        idxs = [start + int(i * (span - 1) / (max_frames - 1)) for i in range(max_frames)]

    imgs = []
    for idx in idxs:
        img = overlay_frame_with_mask(episode_dir, idx)
        if img is not None:
            imgs.append(img)

    if not imgs:
        st.warning("Could not render slice frames (missing frames_raw/ or masks/).")
        return

    cols = st.columns(3)
    for i, img in enumerate(imgs):
        cols[i % 3].image(img, use_container_width=True, caption=f"frame {idxs[i]}")


def zip_directory_to_bytes(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in dir_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(dir_path)))
    return buf.getvalue()