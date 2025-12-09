# conceptops/types.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


# ---------- Core shared types ----------

@dataclass
class VideoMetadata:
    """
    Canonical metadata about a video in the ConceptOps stack.

    This is the *single* definition we want everyone to use:
    - Ingestion
    - Episode building
    - Export layers (RLDS / LeRobot later)

    Keeping this here avoids duplicating slightly-different versions
    of "video metadata" across modules.
    """
    video_path: str                 # Original input path
    original_fps: Optional[float]   # FPS from ffprobe, if available
    extraction_fps: Optional[float] # FPS used for frame extraction
    num_frames: int                 # Number of frames we will use
    width: Optional[int]            # Frame width (post-resize or original)
    height: Optional[int]           # Frame height (post-resize or original)
    duration_sec: Optional[float]   # Approx. duration in seconds
    resize_short_side: Optional[int]# Shorter side resize value (pixels), if used


@dataclass
class FrameRecord:
    """
    A single frame in an episode, optionally linked to a segmentation mask.

    We keep this fairly generic so it works for:
      - VideoMask (single-object masks today)
      - Multi-object masks later (by extending `metadata`)
    """
    index: int                      # 0-based frame index within the episode
    image_path: str                 # Path to RGB frame image (relative or absolute)
    mask_path: Optional[str] = None # Path to mask image, if available
    timestamp_sec: Optional[float] = None  # Time from start of video in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    """
    A single event / action segment over a span of frames.

    This will be produced by Ego2Robot in v0.5+.
    For now, we'll keep the schema ready but events=[].
    """
    event_id: int
    label: str                      # e.g. "pick", "place", "open_door"
    start_frame: int                # inclusive frame index
    end_frame: int                  # inclusive or exclusive (document your convention)
    score: Optional[float] = None   # confidence score from model / heuristic
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """
    Canonical episode object for the integrated pipeline.

    This is the in-memory representation that everything else works with.
    From here, we can:
      - Export to JSON (our internal format).
      - Later: convert to RLDS, LeRobot, COCO, etc.
    """
    episode_id: int
    video: VideoMetadata
    frames: List[FrameRecord]
    events: List[EventRecord] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ----- Serialization helpers -----

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dict.

        We explicitly use `asdict` for dataclasses so nested dataclasses
        turn into nested dicts cleanly.
        """
        return {
            "episode_id": self.episode_id,
            "video": asdict(self.video),
            "frames": [asdict(f) for f in self.frames],
            "events": [asdict(e) for e in self.events],
            "extra": self.extra,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert episode to a JSON string.

        Use this when writing to disk as `episode.json`.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
# ----- Deserialization helpers -----

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """
        Reconstruct an Episode from a dict (the shape produced by to_dict).

        This is the inverse of `to_dict`:
          - Nested dicts are turned back into dataclasses.
          - Missing optional fields fall back to sensible defaults.

        We keep it explicit so schema changes are easy to reason about.
        """
        video_md = VideoMetadata(**data["video"])

        frames = [
            FrameRecord(**frame_dict)
            for frame_dict in data.get("frames", [])
        ]

        events = [
            EventRecord(**event_dict)
            for event_dict in data.get("events", [])
        ]

        return cls(
            episode_id=data["episode_id"],
            video=video_md,
            frames=frames,
            events=events,
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Episode":
        """
        Reconstruct an Episode from a JSON string (inverse of to_json).
        """
        data = json.loads(json_str)
        return cls.from_dict(data)