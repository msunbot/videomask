# conceptops/export/lerobot.py

from __future__ import annotations

from typing import Any, Dict, List

from conceptops.types import Episode


def episode_to_lerobot(episode: Episode) -> Dict[str, Any]:
    """
    Convert an Episode into a LeRobot-style dictionary.

    This is still library-agnostic, but structured to be close to how
    LeRobot / RLDS datasets are organized:

      {
        "episode_id": int,
        "observations": {
          "image_paths": [...],
          "timestamps_sec": [...],
          "mask_paths": [...],
          "num_instances": [...],
        },
        "actions": [...],       # placeholder for now
        "events": [...],        # per-event records
        "metadata": {...},      # video + extra
      }

    Later, you can adapt this dict into a concrete LeRobot Dataset object.
    """

    image_paths: List[str] = []
    timestamps: List[float] = []
    mask_paths: List[str] = []
    num_instances: List[int] = []

    for frame in episode.frames:
        image_paths.append(frame.image_path)
        timestamps.append(frame.timestamp_sec if frame.timestamp_sec is not None else 0.0)
        mask_paths.append(frame.mask_path or "")
        num_instances.append(len(frame.instances))

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

    # Placeholder actions: will be filled by Ego2Robot model later.
    actions: List[Any] = [None] * len(episode.frames)

    return {
        "episode_id": episode.episode_id,
        "observations": {
            "image_paths": image_paths,
            "timestamps_sec": timestamps,
            "mask_paths": mask_paths,
            "num_instances": num_instances,
        },
        "actions": actions,
        "events": events_serialized,
        "metadata": {
            "video": episode.video.__dict__,
            "extra": episode.extra,
        },
    }