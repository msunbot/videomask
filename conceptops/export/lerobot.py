# conceptops/export/lerobot.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

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

# ----------------------------
# Phase 4 adapter API
# ----------------------------

def export_episode_to_lerobot(episode_dir: Union[str, Path], out_dir: Union[str, Path]) -> str:
    """
    Export a LeRobot-like JSON from episode_dir/episode.json WITHOUT constructing dataclasses.

    Output:
      out_dir/lerobot.json
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

    image_paths: List[str] = []
    timestamps: List[float] = []
    mask_paths: List[str] = []
    num_instances: List[int] = []

    for fr in frames:
        image_paths.append(fr.get("image_path", ""))
        timestamps.append(fr.get("timestamp_sec", 0.0) or 0.0)

        instances = fr.get("instances", []) or []
        inst_mask_paths = []
        if isinstance(instances, list):
            for inst in instances:
                mp = inst.get("mask_path")
                if mp:
                    inst_mask_paths.append(mp)

        # Keep a single representative mask_path list flattened for now
        mask_paths.append(fr.get("mask_path", "") or (inst_mask_paths[0] if inst_mask_paths else ""))
        num_instances.append(len(inst_mask_paths))

    # Events: prefer pred_events.json if present
    pred_path = episode_dir / "pred_events.json"
    events: List[Dict[str, Any]] = []
    if pred_path.exists():
        try:
            ev = json.loads(pred_path.read_text())
            if isinstance(ev, dict) and "events" in ev:
                ev = ev["events"]
            if isinstance(ev, list):
                events = ev
        except Exception:
            events = []

    out = {
        "episode_id": episode_json.get("episode_id", 0),
        "observations": {
            "image_paths": image_paths,
            "timestamps_sec": timestamps,
            "mask_paths": mask_paths,
            "num_instances": num_instances,
        },
        "actions": [None] * len(frames),  # placeholder
        "events": events,
        "metadata": {
            "video": video_meta,
            "extra": extra,
        },
    }

    out_path = out_dir / "lerobot.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return str(out_path)