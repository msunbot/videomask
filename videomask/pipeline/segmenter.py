"""
High-level pipeline orchestration.
VideoSegmenter:
    video.mp4 -> frames -> masks -> temporal smoothing -> export.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List

import numpy as np
from PIL import Image

from videomask.core.video_reader import extract_frames_ffmpeg
from videomask.backends.dummy_backend import DummyBackend
from videomask.backends.base import BaseSegmentationBackend
from videomask.pipeline.temporal import smooth_masks_sequence
from videomask.exporters.folder_exporter import save_masks, write_metadata
from videomask.backends.sam3_backend import SAM3Backend

BackendName = Literal["dummy"]  # will add "sam3" later

def _load_image(path: str) -> np.ndarray:
    """Load an image from disk into an RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)

class VideoSegmenter:
    """
    High-level segmentation pipeline.

    Typical usage:
        seg = VideoSegmenter(backend="dummy", fps=2, resize=512)
        seg.run("input.mp4", out_dir="dataset/run1")
    """
    def __init__(
        self,
        backend: BackendName = "dummy",
        fps: int = 2,
        resize: Optional[int] = 512,
        max_frames: Optional[int] = None,
        video_reader: Literal["ffmpeg"] = "ffmpeg",
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend_name = backend
        self.fps = fps
        self.resize = resize
        self.max_frames = max_frames
        self.video_reader = video_reader
        self.backend_kwargs = backend_kwargs or {}

        self.backend: BaseSegmentationBackend = self._create_backend()

    def _create_backend(self) -> BaseSegmentationBackend:
        if self.backend_name == "dummy":
            return DummyBackend(**self.backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")

    def run(self, video_path: str, out_dir: str) -> None:
        """
        Run the full segmentation pipeline on a video.

        Steps:
          1. Extract frames with ffmpeg.
          2. Run segmentation backend per frame.
          3. Apply simple temporal smoothing.
          4. Write masks and metadata to disk.
        """
        out_dir_path = Path(out_dir)
        frames_dir = out_dir_path / "frames_raw"
        masks_dir = out_dir_path / "masks_raw"
        frames_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # 1) extract frames
        frame_paths = extract_frames_ffmpeg(
            video_path=video_path,
            out_dir=str(frames_dir),
            fps=self.fps,
            resize=self.resize,
        )

        if self.max_frames is not None:
            frame_paths = frame_paths[: self.max_frames]

        masks = []
        for frame_path in frame_paths:
            frame = _load_image(frame_path)
            mask = self.backend.segment_frame(frame)
            masks.append(mask)

        # temporal smoothing
        masks = smooth_masks_sequence(masks)

        # save masks + metadata
        masks_dir = out_dir_path / "masks"
        mask_paths = save_masks(masks, frame_paths, str(masks_dir))

        write_metadata(
            out_dir=str(out_dir_path),
            frame_paths=frame_paths,
            mask_paths=mask_paths,
            config={
                "backend": self.backend_name,
                "fps": self.fps,
                "resize": self.resize,
                "max_frames": self.max_frames,
            },
        )

    def _create_backend(self) -> BaseSegmentationBackend:
            if self.backend_name == "dummy":
                return DummyBackend(**self.backend_kwargs)
            elif self.backend_name == "sam3":
                return SAM3Backend(**self.backend_kwargs)
            else:
                raise ValueError(f"Unknown backend: {self.backend_name}")
