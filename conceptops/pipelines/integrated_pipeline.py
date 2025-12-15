# conceptops/pipelines/integrated_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from conceptops.ingestion.video_ingest import ingest_video
from conceptops.types import FrameRecord, Episode, InstanceMask
from conceptops.core.events import (
    SimpleEventDetector,
    SimpleEventConfig,
    MotionEventDetector,
    MotionEventConfig,
    ModelEventDetector,
    ModelEventConfig,
)
from conceptops.perception.mask_metrics import compute_mask_stats
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
    ingest_frame_paths: List[str],
    vm_metadata: Dict[str, Any],
    extraction_fps: Optional[float],
) -> List[FrameRecord]:
    """
    Build a list of FrameRecord objects by combining:

      - The ordered frame paths from ingestion.
      - The lists of frames/masks from VideoMask's metadata.json.

    Supports two mask formats in vm_metadata["masks"]:
      1) List[str]: one mask path per frame (legacy single-object).
      2) List[List[str]]: list of mask paths per frame (multi-object).
    """
    frames_from_vm = vm_metadata.get("frames") or ingest_frame_paths
    masks_raw = vm_metadata.get("masks") or []

    frame_records: List[FrameRecord] = []

    # Normalize masks into a list of list-of-paths for uniform handling.
    # Case 1: single list of strings
    if masks_raw and isinstance(masks_raw[0], str):
        masks_per_frame: List[List[str]] = [[p] for p in masks_raw]
    else:
        # Assume already list-of-lists or empty.
        masks_per_frame = masks_raw

    for idx, frame_path in enumerate(frames_from_vm):
        # For safety, if masks_per_frame is shorter than frames, use empty list.
        frame_mask_paths: List[str] = []
        if idx < len(masks_per_frame) and masks_per_frame[idx] is not None:
            # Some backends may produce None entries; filter them out.
            frame_mask_paths = [
                str(p) for p in masks_per_frame[idx] if p is not None
            ]

        timestamp_sec = None
        if extraction_fps and extraction_fps > 0:
            timestamp_sec = idx / float(extraction_fps)

        instances: List[InstanceMask] = []
        frame_metadata: Dict[str, Any] = {}

        # Compute stats for each instance mask in this frame.
        areas: List[float] = []
        for inst_id, m_path in enumerate(frame_mask_paths):
            stats = compute_mask_stats(m_path)
            instances.append(
                InstanceMask(
                    instance_id=inst_id,
                    mask_path=m_path,
                    area_px=stats.area_px,
                    area_ratio=stats.area_ratio,
                    bbox=None,
                    metadata={},
                )
            )
            areas.append(stats.area_ratio)

        # Legacy single-mask field: point to the first instance, if any.
        mask_path_legacy: Optional[str] = frame_mask_paths[0] if frame_mask_paths else None

        # Aggregate mask quality for this frame.
        if areas:
            mean_area = sum(areas) / len(areas)
            frame_metadata["mask_quality"] = {
                "area_ratio": mean_area,
                "max_area_ratio": max(areas),
                "min_area_ratio": min(areas),
                "mean_area_ratio": mean_area,
                "num_instances": len(areas),
            }

        frame_records.append(
            FrameRecord(
                index=idx,
                image_path=str(frame_path),
                mask_path=mask_path_legacy,
                timestamp_sec=timestamp_sec,
                metadata=frame_metadata,
                instances=instances,
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

    # 5) Run event detector (v0.5)
    mode = e2r_config.get("mode", "fixed")  # "fixed", "motion", "motion_mask", or "model"

    if mode in ("motion", "motion_mask"):
        motion_cfg = MotionEventConfig(
            threshold_multiplier=float(e2r_config.get("threshold_multiplier", 2.0)),
            min_event_length=int(e2r_config.get("min_event_length", 3)),
            base_label=e2r_config.get("base_label", "move"),
            min_area_ratio=float(e2r_config.get("min_area_ratio", 0.0)),
        )
        detector = MotionEventDetector(config=motion_cfg)
        event_config = {
            "mode": mode,
            "threshold_multiplier": motion_cfg.threshold_multiplier,
            "min_event_length": motion_cfg.min_event_length,
            "base_label": motion_cfg.base_label,
            "min_area_ratio": motion_cfg.min_area_ratio,
        }

    # >>> Phase 3: model mode uses trained artifact-backed ModelEventDetector
    elif mode == "model":
        # New Phase 3 model config: point to trained artifact directory.
        #
        # Expected e2r_config keys (passed from scripts/batch_build_episodes.py or user code):
        # - model_dir: str path to directory containing model.pt, labels.json, feature_spec.json
        # - window_size, stride, topk: proposal settings
        # - min_score: score threshold for returning events
        # - nms_iou: temporal IoU threshold for dedup (NMS)
        model_dir = e2r_config.get("model_dir", None)
        if not model_dir:
            raise ValueError("e2r_config['model_dir'] is required when mode='model'")

        detector = ModelEventDetector(
            model_dir=str(model_dir),
            window_size=int(e2r_config.get("window_size", 8)),
            stride=int(e2r_config.get("stride", 4)),
            topk=int(e2r_config.get("topk", 5)),
            min_score=float(e2r_config.get("min_score", 0.55)),
            nms_iou=float(e2r_config.get("nms_iou", 0.5)),
        )

        # Keep event_config purely descriptive for metadata/debugging.
        event_config = {
            "mode": "model",
            "model_dir": str(model_dir),
            "window_size": int(e2r_config.get("window_size", 8)),
            "stride": int(e2r_config.get("stride", 4)),
            "topk": int(e2r_config.get("topk", 5)),
            "min_score": float(e2r_config.get("min_score", 0.55)),
            "nms_iou": float(e2r_config.get("nms_iou", 0.5)),
        }

    else:
        frames_per_event = int(e2r_config.get("frames_per_event", 16))
        base_label = e2r_config.get("base_label", "segment")
        detector = SimpleEventDetector(
            config=SimpleEventConfig(
                frames_per_event=frames_per_event,
                base_label=base_label,
            )
        )
        event_config = {
            "mode": "fixed",
            "frames_per_event": frames_per_event,
            "base_label": base_label,
        }

    events = detector.detect(
        frame_records=frame_records,
        episode_dir=str(out_dir_path),
    )
    
    episode = Episode(
        episode_id=0,
        video=ingest_result.metadata,
        frames=frame_records,
        events=events,
        extra={
            "videomask_metadata": vm_metadata,
            "event_config": event_config,
        },
    )
    episode_path = out_dir_path / "episode.json"
    episode_path.write_text(episode.to_json(indent=2), encoding="utf-8")

    return str(episode_path)