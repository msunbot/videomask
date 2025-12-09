# conceptops/pipelines/integrated_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from conceptops.ingestion.video_ingest import ingest_video
from conceptops.types import FrameRecord, Episode
from conceptops.core.events import SimpleEventDetector, SimpleEventConfig
from videomask.pipeline.segmenter import VideoSegmenter


def _load_videomask_metadata(out_dir: Path) -> Dict[str, Any]:
    """
    Load VideoMask's metadata.json from `out_dir`.

    VideoMask v0.1 stores:
      - frames: list of frame paths (strings)
      - masks: list of mask paths (strings)
      - fps: frame sampling rate
      - config: backend + parameters (depends on implementation)

    We keep this helper small and focused so we can easily swap in
    newer metadata formats later.
    """
    metadata_path = out_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"VideoMask metadata.json not found at: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_frame_records(
    ingest_frame_paths: list[str],
    vm_metadata: Dict[str, Any],
    extraction_fps: Optional[float],
) -> list[FrameRecord]:
    """
    Build a list of FrameRecord objects by combining:

      - The ordered frame paths from ingestion.
      - The lists of frames/masks from VideoMask's metadata.json.

    We prefer the paths from metadata.json if present (they might
    already be relative to out_dir), but fall back to ingestion paths.
    """
    frames_from_vm = vm_metadata.get("frames") or ingest_frame_paths
    masks_from_vm = vm_metadata.get("masks") or []

    frame_records: list[FrameRecord] = []

    for idx, frame_path in enumerate(frames_from_vm):
        mask_path = masks_from_vm[idx] if idx < len(masks_from_vm) else None
        timestamp_sec = None
        if extraction_fps and extraction_fps > 0:
            timestamp_sec = idx / float(extraction_fps)

        frame_records.append(
            FrameRecord(
                index=idx,
                image_path=str(frame_path),
                mask_path=str(mask_path) if mask_path is not None else None,
                timestamp_sec=timestamp_sec,
                metadata={},
            )
        )

    return frame_records


def process_video_to_dataset(
    video_path: str,
    out_dir: str,
    *,
    vm_config: Optional[Dict[str, Any]] = None,
    e2r_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Integrated pipeline v0.5:

      1. Ingest video → frames + VideoMetadata.
      2. Run VideoMask segmentation.
      3. Read VideoMask's metadata.json.
      4. Run a simple event detector (v0.5).
      5. Build Episode and write `episode.json`.

    Returns:
        Path to `episode.json` as a string.
    """
    vm_config = vm_config or {}
    e2r_config = e2r_config or {}

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) Ingestion
    ingest_result = ingest_video(
        video_path=video_path,
        out_dir=out_dir_path,
        fps=int(vm_config.get("fps", 2)),
        resize=vm_config.get("resize", 512),
        max_frames=vm_config.get("max_frames"),
    )

    # 2) Segmentation via VideoMask
    seg = VideoSegmenter(
        backend=vm_config.get("backend", "dummy"),
        fps=int(ingest_result.metadata.extraction_fps or vm_config.get("fps", 2)),
        resize=ingest_result.metadata.resize_short_side,
        max_frames=ingest_result.metadata.num_frames,
        backend_kwargs=vm_config.get("backend_kwargs", {}),
    )
    seg.run(video_path, out_dir=str(out_dir_path))

    # 3) Load VideoMask metadata.json
    vm_metadata = _load_videomask_metadata(out_dir_path)

    # 4) Build canonical FrameRecord list
    frame_records = _build_frame_records(
        ingest_frame_paths=ingest_result.frame_paths,
        vm_metadata=vm_metadata,
        extraction_fps=ingest_result.metadata.extraction_fps,
    )

    # 5) Run simple event detector (v0.5)
    frames_per_event = int(e2r_config.get("frames_per_event", 16))
    base_label = e2r_config.get("base_label", "segment")
    detector = SimpleEventDetector(
        config=SimpleEventConfig(
            frames_per_event=frames_per_event,
            base_label=base_label,
        )
    )
    events = detector.detect(frame_records)

    episode = Episode(
        episode_id=0,
        video=ingest_result.metadata,
        frames=frame_records,
        events=events,
        extra={
            "videomask_metadata": vm_metadata,
            "event_config": {
                "frames_per_event": frames_per_event,
                "base_label": base_label,
            },
        },
    )

    episode_path = out_dir_path / "episode.json"
    episode_path.write_text(episode.to_json(indent=2), encoding="utf-8")

    return str(episode_path)