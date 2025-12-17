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
)
from conceptops.perception.mask_metrics import compute_mask_stats
from videomask.pipeline.segmenter import VideoSegmenter


def _load_videomask_metadata(out_dir: Path) -> Dict[str, Any]:
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
    frames_from_vm = vm_metadata.get("frames") or ingest_frame_paths
    masks_raw = vm_metadata.get("masks") or []

    frame_records: List[FrameRecord] = []

    if masks_raw and isinstance(masks_raw[0], str):
        masks_per_frame: List[List[str]] = [[p] for p in masks_raw]
    else:
        masks_per_frame = masks_raw

    for idx, frame_path in enumerate(frames_from_vm):
        frame_mask_paths: List[str] = []
        if idx < len(masks_per_frame) and masks_per_frame[idx] is not None:
            frame_mask_paths = [str(p) for p in masks_per_frame[idx] if p is not None]

        timestamp_sec = None
        if extraction_fps and extraction_fps > 0:
            timestamp_sec = idx / float(extraction_fps)

        instances: List[InstanceMask] = []
        frame_metadata: Dict[str, Any] = {}

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

        mask_path_legacy: Optional[str] = frame_mask_paths[0] if frame_mask_paths else None

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


def _resolve_model_dir(e2r_config: Dict[str, Any]) -> str:
    """
    Backwards compatible resolution for model artifacts.

    Preferred (new):
      e2r_config["model_dir"] = "/path/to/model_artifacts"

    Legacy (tests / older code):
      e2r_config["model_name"] = "stub_v1"  -> resolves to "data/models/stub_v1"

    Safety fallback (so demos/tests don’t hard-fail if someone forgets):
      data/models/event_model_demo3_wt if present, else data/models/event_model_demo3, else error.
    """
    # 1) New explicit path
    if e2r_config.get("model_dir"):
        return str(e2r_config["model_dir"])

    # 2) Legacy name -> data/models/<name>
    model_name = e2r_config.get("model_name")
    if model_name:
        candidate = Path("data/models") / str(model_name)
        return str(candidate)

    # 3) Safe defaults (prefer the best demo model)
    for p in [
        Path("data/models/event_model_demo3_wt"),
        Path("data/models/event_model_demo3"),
    ]:
        if (p / "model.pt").exists() and (p / "labels.json").exists():
            return str(p)

    raise ValueError(
        "mode='model' requires either e2r_config['model_dir'] or legacy e2r_config['model_name'], "
        "and no default demo model artifacts were found under data/models/."
    )


def process_video_to_dataset(
    video_path: str,
    out_dir: str,
    *,
    vm_config: Optional[Dict[str, Any]] = None,
    e2r_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Integrated pipeline v0.5:
      1) ingest video
      2) run VideoMask
      3) build FrameRecord list
      4) run event detector
      5) build Episode and write episode.json
    """
    vm_config = vm_config or {}
    e2r_config = e2r_config or {}

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ingest_result = ingest_video(
        video_path=video_path,
        out_dir=out_dir_path,
        fps=int(vm_config.get("fps", 2)),
        resize=vm_config.get("resize", 512),
        max_frames=vm_config.get("max_frames"),
    )

    seg = VideoSegmenter(
        backend=vm_config.get("backend", "dummy"),
        fps=int(ingest_result.metadata.extraction_fps or vm_config.get("fps", 2)),
        resize=ingest_result.metadata.resize_short_side,
        max_frames=ingest_result.metadata.num_frames,
        backend_kwargs=vm_config.get("backend_kwargs", {}),
    )
    seg.run(video_path, out_dir=str(out_dir_path))

    vm_metadata = _load_videomask_metadata(out_dir_path)

    frame_records = _build_frame_records(
        ingest_frame_paths=ingest_result.frame_paths,
        vm_metadata=vm_metadata,
        extraction_fps=ingest_result.metadata.extraction_fps,
    )

    mode = e2r_config.get("mode", "fixed")  # fixed, motion, motion_mask, model

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

    elif mode == "model":
        # Backwards compatible: accept model_dir or model_name.
        model_dir = _resolve_model_dir(e2r_config)

        inference_profile = (e2r_config or {}).get("inference_profile", "default")

        detector = ModelEventDetector(
            model_dir=str(model_dir),
            window_size=int(e2r_config.get("window_size", 8)),
            stride=int(e2r_config.get("stride", 4)),
            topk=int(e2r_config.get("topk", 5)),
            min_score=float(e2r_config.get("min_score", 0.55)),
            nms_iou=float(e2r_config.get("nms_iou", 0.5)),
            inference_profile=str(inference_profile),
        )

        event_config = {
            "mode": "model",
            "model_dir": str(model_dir),
            "model_name": e2r_config.get("model_name"),  # may be None
            "window_size": int(e2r_config.get("window_size", 8)),
            "stride": int(e2r_config.get("stride", 4)),
            "topk": int(e2r_config.get("topk", 5)),
            "min_score": float(e2r_config.get("min_score", 0.55)),
            "nms_iou": float(e2r_config.get("nms_iou", 0.5)),
            "inference_profile": str(inference_profile),
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

    if mode == "model":
        events = detector.detect(
            frame_records=frame_records,
            episode_dir=str(out_dir_path),
        )
    else:
        events = detector.detect(frame_records)

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