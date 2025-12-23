# conceptops/export/rlds.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

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

# ----------------------------
# Phase 4 adapter API
# ----------------------------

def export_episode_to_rlds(episode_dir: Union[str, Path], out_dir: Union[str, Path]) -> str:
    """
    Export a RLDS-like JSON from episode_dir/episode.json WITHOUT constructing dataclasses.

    Output:
      out_dir/rlds.json
    """
    episode_dir = Path(episode_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_json_path = episode_dir / "episode.json"
    if not ep_json_path.exists():
        raise FileNotFoundError(f"episode.json not found in {episode_dir}")

    episode_json: Dict[str, Any] = json.loads(ep_json_path.read_text())
    frames = episode_json.get("frames", []) or []
    video_meta = episode_json.get("video", {}) or {}
    extra = episode_json.get("extra", {}) or {}

    steps: List[Dict[str, Any]] = []
    num_frames = len(frames)

    for t, fr in enumerate(frames):
        instances = fr.get("instances", []) or []
        mask_paths = []
        if isinstance(instances, list):
            for inst in instances:
                mp = inst.get("mask_path")
                if mp:
                    mask_paths.append(mp)

        obs = {
            "image_path": fr.get("image_path", ""),
            "timestamp_sec": fr.get("timestamp_sec", 0.0) or 0.0,
            "mask_paths": mask_paths,
            "num_instances": len(mask_paths),
        }

        steps.append(
            {
                "timestep": t,
                "observation": obs,
                "action": None,
                "reward": 0.0,
                "discount": 1.0,
                "is_terminal": (t == num_frames - 1),
            }
        )

    out = {
        "episode_id": episode_json.get("episode_id", 0),
        "steps": steps,
        "metadata": {
            "video": video_meta,
            "extra": extra,
        },
    }

    out_path = out_dir / "rlds.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return str(out_path)