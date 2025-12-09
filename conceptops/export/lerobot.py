# conceptops/export/lerobot.py

from __future__ import annotations

from typing import Any, Dict, List

from conceptops.types import Episode


def episode_to_lerobot(episode: Episode) -> Dict[str, Any]:
    """
    Convert an Episode into a minimal LeRobot-style dict.

    This is *not* a full RLDS implementation yet. The goal is:
      - Provide a stable conversion surface.
      - Make it trivial to serialize into the actual LeRobot / RLDS
        formats later by adapting this function.

    Structure (simplified):

      {
        "episode_id": int,
        "observations": {
          "image_paths": [...],
          "timestamps_sec": [...],
          "mask_paths": [...],
        },
        "actions": [],  # placeholder for now
        "events": [...],
        "metadata": {...},
      }
    """
    image_paths: List[str] = []
    timestamps: List[float] = []
    mask_paths: List[str] = []

    for frame in episode.frames:
        image_paths.append(frame.image_path)
        timestamps.append(frame.timestamp_sec if frame.timestamp_sec is not None else 0.0)
        mask_paths.append(frame.mask_path or "")

    events_serialized: List[Dict[str, Any]] = []
    for ev in episode.events:
        events_serialized.append(
            {
                "event_id": ev.event_id,
                "label": ev.label,
                "start_frame": ev.start_frame,
                "end_frame": ev.end_frame,
                "score": ev.score,
                "metadata": ev.metadata,
            }
        )

    return {
        "episode_id": episode.episode_id,
        "observations": {
            "image_paths": image_paths,
            "timestamps_sec": timestamps,
            "mask_paths": mask_paths,
        },
        # Placeholder actions: you will later fill with Ego2Robot outputs.
        "actions": [],
        "events": events_serialized,
        "metadata": {
            "video": episode.video.__dict__,
            "extra": episode.extra,
        },
    }