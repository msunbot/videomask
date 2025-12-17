# conceptops/types.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

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
class InstanceMask:
    """
    One instance mask for an object in a frame.

    This is the building block for multi-object support: 
        - Each frame can have 0..N InstanceMask objects
        - For now, we treat the legacy per-frame 'mask_path' as 
          "instance 0", but this schema can holdmultiple masks
          per frame when we plug in multi-object backends.
    
    Fields:
        instance_id: Stable ID for the object across frames (Phase 3)
        mask_path: Path to the binary mask image for this instance
        area_px: Number of foreground pixels (optional quality measure)
        area_ratio: area_px / total_pixels (0..1)
        bbox: Optional bounding box (x_min, y_min, x_max, y_max)
        metadata: Free-form dict for backend-specific info
    """
    instance_id: int
    mask_path: str
    area_px: Optional[int] = None
    area_ratio: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FrameRecord:
    """
    A single frame in an episode, optionally linked to a segmentation mask.

    Backward-compatible fields:
      - `mask_path`: legacy single-mask path, kept for simplicity.

    Multi-object aware:
      - `instances`: list of InstanceMask objects for this frame.
        For current single-object VideoMask, this will usually be
        either 0 or 1 instance, but Phase 3 backends will populate
        multiple instances.
    """
    index: int                      # 0-based frame index within the episode
    image_path: str                 # Path to RGB frame image (relative or absolute)
    mask_path: Optional[str] = None # Path to mask image, if available
    timestamp_sec: Optional[float] = None  # Time from start of video in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    instances: List[InstanceMask] = field(default_factory=list)

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

        This explicitly rebuilds nested dataclasses (VideoMetadata,
        FrameRecord, InstanceMask, EventRecord) instead of relying on
        dataclass expansion, so we don't end up with dicts in places that
        should be dataclasses.
        """
        video_md = VideoMetadata(**data["video"])

        frames: List[FrameRecord] = []
        for frame_dict in data.get("frames", []):
            # Handle instances: list of dicts -> list of InstanceMask
            inst_dicts = frame_dict.get("instances", [])
            instances: List[InstanceMask] = []
            for inst in inst_dicts:
                # if it's already an InstanceMask, keep it
                if isinstance(inst, InstanceMask):
                    instances.append(inst)
                else:
                    instances.append(InstanceMask(**inst))

            frame = FrameRecord(
                index=frame_dict["index"],
                image_path=frame_dict["image_path"],
                mask_path=frame_dict.get("mask_path"),
                timestamp_sec=frame_dict.get("timestamp_sec"),
                metadata=frame_dict.get("metadata", {}),
                instances=instances,
            )
            frames.append(frame)

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
        data = json.loads(json_str)
        return cls.from_dict(data)