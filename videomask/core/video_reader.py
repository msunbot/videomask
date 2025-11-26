"""
Video frame extraction using ffmpeg.
Shell out to ffmpeg for robustness across platforms.
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Optional


class FrameExtractionError(Exception):
    """Raised when frame extraction fails for any reason."""


def build_filter(fps: int, resize: Optional[int]) -> str:
    """
    Build the ffmpeg -vf filter chain string.
    Example: "fps=2,scale=512:-1"
    """
    parts = [f"fps={fps}"]
    if resize is not None:
        parts.append(f"scale={resize}:-1")
    return ",".join(parts)


def extract_frames_ffmpeg(
    video_path: str,
    out_dir: str,
    fps: int = 2,
    resize: Optional[int] = 512,
) -> List[str]:
    """
    Extract frames from a video using ffmpeg.

    Args:
        video_path: Path to the input video file.
        out_dir: Directory where extracted frames will be stored.
        fps: Target frames per second.
        resize: Resize shorter side to this many pixels. If None, keep original size.

    Returns:
        Sorted list of frame image paths.

    Raises:
        FrameExtractionError: If ffmpeg fails or no frames are produced.
    """
    video_path = str(video_path)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    pattern = str(out_dir_path / "frame_%06d.jpg")
    vf_filter = build_filter(fps=fps, resize=resize)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        vf_filter,
        "-q:v",
        "2",
        pattern,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise FrameExtractionError(f"ffmpeg failed: {e}") from e

    frames = sorted(str(p) for p in out_dir_path.glob("frame_*.jpg"))
    if not frames:
        raise FrameExtractionError(
            "No frames extracted – check the video file and ffmpeg installation."
        )
    return frames