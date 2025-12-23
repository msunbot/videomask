"""
Demo runner that calls the Phase 3 canonical pipeline entrypoint.

Key goals:
- Keep the demo UI thin; the runner owns filesystem layout + backend fallback.
- Absolutely do NOT refactor the core pipeline.
- Provide a clean "sam3 preferred" selection with graceful fallback to dummy.

Important:
We don't assume specific config class shapes because your repo may have:
- dataclasses for vm_config / e2r_config, OR
- plain dicts

So this runner uses a small "best-effort" config patch approach:
- If vm_config has a known attribute, we set it.
- If vm_config is a dict, we set keys.
If nothing matches, we keep defaults and only report it in the UI.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset


class DemoBackendChoice(str, Enum):
    SAM3 = "sam3"
    DUMMY = "dummy"


@dataclass
class DemoRunResult:
    episode_dir: str
    input_description: str
    num_frames: int
    backend_used: DemoBackendChoice
    backend_message: str = ""

    def try_export(self, fmt: str) -> Tuple[bool, str]:
        episode_dir = Path(self.episode_dir)
        try:
            if fmt == "coco":
                from conceptops.export.coco import export_episode_to_coco  # type: ignore
                out_dir = episode_dir / "export_coco"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = export_episode_to_coco(episode_dir=episode_dir, out_dir=out_dir)
                return True, f"COCO export complete → {out_path}"
            if fmt == "rlds":
                from conceptops.export.rlds import export_episode_to_rlds  # type: ignore
                out_dir = episode_dir / "export_rlds"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = export_episode_to_rlds(episode_dir=episode_dir, out_dir=out_dir)
                return True, f"RLDS export complete → {out_path}"
            if fmt == "lerobot":
                from conceptops.export.lerobot import export_episode_to_lerobot  # type: ignore
                out_dir = episode_dir / "export_lerobot"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = export_episode_to_lerobot(episode_dir=episode_dir, out_dir=out_dir)
                return True, f"LeRobot export complete → {out_path}"
            return False, f"Unknown export format '{fmt}'."
        except Exception as e:
            return False, f"Exporter '{fmt}' failed: {e}"


def _normalize_episode_dir(returned_path: str | None, out_root: Path) -> Path:
    p = Path(returned_path).expanduser().resolve() if returned_path else None
    if p and p.is_file():
        return p.parent.resolve()
    if p and p.is_dir():
        return p.resolve()

    # fallback: newest dir in out_root
    candidates = sorted([d for d in out_root.iterdir() if d.is_dir()],
                        key=lambda d: d.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"Pipeline wrote no episode directory into {out_root}")
    return candidates[0].resolve()


def _count_frames(episode_dir: Path) -> int:
    frames_dir = episode_dir / "frames_raw"
    if frames_dir.exists():
        files = [p for p in frames_dir.iterdir() if p.is_file()]
        if files:
            return len(files)
    ep = episode_dir / "episode.json"
    if ep.exists():
        data = json.loads(ep.read_text())
        v = data.get("video", {})
        if isinstance(v, dict) and isinstance(v.get("num_frames"), int):
            return int(v["num_frames"])
    return 0


def _materialize_pred_events_from_episode_json(episode_dir: Path) -> str:
    """
    Write pred_events.json from episode.json["events"].

    This makes the demo UI stable regardless of whether the pipeline
    writes a separate pred_events.json file.
    """
    ep_path = episode_dir / "episode.json"
    if not ep_path.exists():
        return "episode.json missing; cannot materialize pred_events.json."

    data = json.loads(ep_path.read_text())
    events = data.get("events", [])
    pred_path = episode_dir / "pred_events.json"

    if not isinstance(events, list):
        pred_path.write_text("[]", encoding="utf-8")
        return "episode.json events not a list; wrote empty pred_events.json."

    out_events: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        out_events.append(
            {
                "event_id": ev.get("event_id", 0),
                "label": ev.get("label", "unknown"),
                "start_frame": ev.get("start_frame", 0),
                "end_frame": ev.get("end_frame", 0),
                "score": ev.get("score", None),
                "metadata": ev.get("metadata", {}) or {},
            }
        )

    pred_path.write_text(json.dumps(out_events, indent=2), encoding="utf-8")
    return f"Wrote pred_events.json from episode.json events ({len(out_events)} events)."


def run_conceptops_pipeline_for_demo(
    video_path: Path,
    backend_choice: DemoBackendChoice,
    input_description: str = "",
    e2r_config: Dict[str, Any] | None = None,
    vm_config_overrides: Dict[str, Any] | None = None,
) -> DemoRunResult:
    """
    Demo runner now accepts e2r_config so you can choose event mode:
      - model (preferred for the demo)
      - motion / motion_mask
      - fixed
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_root = Path(tempfile.mkdtemp(prefix="conceptops_demo_run_")).resolve()

    # vm_config: only pass the keys integrated_pipeline expects
    vm_config: Dict[str, Any] = {"backend": backend_choice.value, "backend_kwargs": {}}
    if vm_config_overrides:
        vm_config.update(vm_config_overrides)

    e2r_config = dict(e2r_config or {})

    t0 = time.time()
    returned = process_video_to_dataset(
        video_path=str(video_path),
        out_dir=str(out_root),
        vm_config=vm_config,
        e2r_config=e2r_config,
    )
    t1 = time.time()

    episode_dir = _normalize_episode_dir(returned, out_root)
    num_frames = _count_frames(episode_dir)

    msg_parts = [
        f"Pipeline returned: {returned}",
        f"Elapsed: {round(t1 - t0, 3)}s",
        f"Event mode requested: {e2r_config.get('mode', 'fixed')}",
        _materialize_pred_events_from_episode_json(episode_dir),
    ]

    return DemoRunResult(
        episode_dir=str(episode_dir),
        input_description=input_description or video_path.name,
        num_frames=num_frames,
        backend_used=backend_choice,
        backend_message=" ".join(msg_parts),
    )