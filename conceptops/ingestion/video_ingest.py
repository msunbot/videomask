# conceptops/ingestion/video_ingest.py

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Tuple

from videomask.core.video_reader import extract_frames_ffmpeg
from conceptops.types import VideoMetadata  # <-- canonical video schema

logger = logging.getLogger(__name__)

@dataclass
class VideoIngestResult:
    """
    Container for the result of ingesting a video.

    This is what downstream steps (segmentation, events, etc.) will rely on.
    """
    metadata: VideoMetadata         # High-level metadata about the ingest
    frame_paths: List[str]          # List of frame image paths (ordered)
    frames_dir: str                 # Directory that contains the extracted frames

def _run_ffprobe(video_path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[float]]:
    """
    Try to read basic metadata from the video using ffprobe.

    Returns:
        (original_fps, width, height, duration_sec)

    Any field may be None if ffprobe is not available or parsing fails.
    """
    try:
        # We ask ffprobe for: frame rate, width, height, and duration.
        #
        # -v error: only show errors
        # -select_streams v:0: first video stream
        # -show_entries: which fields to output
        # -of default=...: print values one per line, no labels
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,width,height,duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,  # we handle failures ourselves
        )
    except FileNotFoundError:
        # ffprobe is not installed; we log and return None metadata.
        logger.warning("ffprobe not found on PATH; skipping video metadata probe.")
        return None, None, None, None

    if result.returncode != 0:
        logger.warning(
            "ffprobe failed for %s with code %s: %s",
            video_path,
            result.returncode,
            result.stderr.strip(),
        )
        return None, None, None, None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 4:
        # Unexpected format; just bail out gracefully.
        logger.warning("ffprobe output for %s had unexpected format: %s", video_path, lines)
        return None, None, None, None

    avg_frame_rate_str, width_str, height_str, duration_str = lines[:4]

    # Parse FPS: ffprobe often returns as "num/den" (e.g. "30000/1001").
    def _parse_fps(s: str) -> Optional[float]:
        if "/" in s:
            num, den = s.split("/", 1)
            try:
                return float(num) / float(den)
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None

    original_fps = _parse_fps(avg_frame_rate_str)

    try:
        width = int(width_str)
    except ValueError:
        width = None

    try:
        height = int(height_str)
    except ValueError:
        height = None

    try:
        duration_sec = float(duration_str)
    except ValueError:
        duration_sec = None

    return original_fps, width, height, duration_sec


# ---------- Public API ----------

def ingest_video(
    video_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    fps: int,
    resize: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> VideoIngestResult:
    """
    Ingest a raw video into a set of extracted frames + metadata.

    This is the first step of the integrated pipeline. It is deliberately
    side-effect *transparent*:
      - It writes frames to a known directory (frames_raw/).
      - It returns a dataclass that fully describes what was done.

    Args:
        video_path:
            Path to the input video file.
        out_dir:
            Root directory for all pipeline artifacts for this run.
            We will create (if needed) `out_dir/frames_raw/` and put frames there.
        fps:
            Frame sampling rate used during extraction (frames per second).
            This is *not* necessarily the original video FPS.
        resize:
            If provided, we resize frames so that the shorter side equals this value,
            preserving aspect ratio. If None, we keep original resolution.
            (We simply pass this value through to `extract_frames_ffmpeg`.)
        max_frames:
            Optional hard cap on the number of frames *returned* by this function.
            Useful for quick experiments. Note that the underlying extractor may
            still write more than `max_frames` frames to disk; we just truncate
            the returned list for downstream steps.

    Returns:
        VideoIngestResult:
            - metadata: VideoMetadata with FPS, size, duration, etc.
            - frame_paths: ordered list of frame image paths (as strings).
            - frames_dir: directory containing the frames.

    Raises:
        FileNotFoundError: if the input video does not exist.
        RuntimeError: if no frames could be extracted.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    # 1) Validate input & prepare output directory
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    frames_dir = out_dir / "frames_raw"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 2) Probe the video with ffprobe (if available) to get baseline metadata.
    original_fps, orig_width, orig_height, duration_sec = _run_ffprobe(video_path)

    # 3) Extract frames using the existing VideoMask helper.
    #
    # We intentionally reuse `extract_frames_ffmpeg` so ingestion behavior stays
    # consistent across VideoMask and ConceptOps.
    frame_paths: List[str] = extract_frames_ffmpeg(
        video_path=str(video_path),
        out_dir=str(frames_dir),
        fps=fps,
        resize=resize,
    )

    if not frame_paths:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    # Optionally truncate the list of frames we *use* downstream.
    if max_frames is not None and max_frames > 0:
        frame_paths = frame_paths[: max_frames]

    num_frames = len(frame_paths)

    # 4) Estimate effective frame size after resize.
    #
    # We don't open images here to keep ingestion light; instead, we:
    #   - Prefer ffprobe's reported size if resize is None.
    #   - If resize is set, we only know the *shorter* side (resize), so we
    #     store width/height as None for now and let later steps infer them
    #     when they actually need pixel data.
    if resize is None:
        width = orig_width
        height = orig_height
    else:
        width = None
        height = None

    # 5) Approximate duration if ffprobe failed.
    #
    # If ffprobe gave us a duration, trust it. Otherwise, approximate with
    # (#frames / extraction_fps).
    if duration_sec is None and fps > 0:
        duration_sec = num_frames / float(fps)

    metadata = VideoMetadata(
        video_path=str(video_path),
        original_fps=original_fps,
        extraction_fps=float(fps),
        num_frames=num_frames,
        width=width,
        height=height,
        duration_sec=duration_sec,
        resize_short_side=resize,
    )

    return VideoIngestResult(
        metadata=metadata,
        frame_paths=frame_paths,
        frames_dir=str(frames_dir),
    )