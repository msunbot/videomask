# conceptops/export/rlds.py

from __future__ import annotations

from typing import Any, Dict, List

from conceptops.types import Episode


def episode_to_rlds(episode: Episode) -> Dict[str, Any]:
    """
    Convert an Episode into a RLDS-like dictionary.

    This is a simplified structure intended to be close to the RLDS
    convention (a list of steps with observation / action / reward / done):

      {
        "episode_id": int,
        "steps": [
          {
            "timestep": int,
            "observation": {...},
            "action": ...,
            "reward": float,
            "discount": float,
            "is_terminal": bool,
          },
          ...
        ],
        "metadata": {...},
      }

    Rewards / actions are placeholders for now and can be filled in later
    once you have real task definitions.
    """
    steps: List[Dict[str, Any]] = []

    num_frames = len(episode.frames)
    for t, frame in enumerate(episode.frames):
        obs = {
            "image_path": frame.image_path,
            "timestamp_sec": frame.timestamp_sec if frame.timestamp_sec is not None else 0.0,
            "mask_paths": [inst.mask_path for inst in frame.instances],
            "num_instances": len(frame.instances),
        }

        step = {
            "timestep": t,
            "observation": obs,
            "action": None,      # placeholder
            "reward": 0.0,       # placeholder
            "discount": 1.0,
            "is_terminal": (t == num_frames - 1),
        }
        steps.append(step)

    return {
        "episode_id": episode.episode_id,
        "steps": steps,
        "metadata": {
            "video": episode.video.__dict__,
            "extra": episode.extra,
        },
    }