import json
from pathlib import Path
from datetime import datetime

from videomask.pipeline.segmenter import VideoSegmenter


def run_conceptops_mask_pipeline(
    video_path: Path,
    out_dir: Path,
    backend: str = "dummy",
    fps: float = 1.0,
    resize: int = 256,
    max_frames: int = 30,
) -> None:
    """
    Phase 1 pipeline:
      raw video → VideoMask → frames_raw/, masks/, metadata.json
      plus a lightweight ConceptOps manifest for later phases.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vm_out_dir = out_dir  # for now, share the same dir

    seg = VideoSegmenter(
        backend=backend,
        fps=fps,
        resize=resize,
        max_frames=max_frames,
    )

    seg.run(str(video_path), out_dir=str(vm_out_dir))

    # Write a small ConceptOps manifest that future stages will consume.
    manifest = {
        "version": "0.1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "video_path": str(video_path),
        "video_mask_output_dir": str(vm_out_dir),
        "frames_dir": str(vm_out_dir / "frames_raw"),
        "masks_dir": str(vm_out_dir / "masks"),
        "metadata_path": str(vm_out_dir / "metadata.json"),
        "backend": backend,
        "fps": fps,
        "resize": resize,
        "max_frames": max_frames,
        "stages": {
            "masks": "completed",
            "events": "pending",
            "concepts": "pending",
            "lerobot_episode": "pending",
        },
    }

    manifest_path = out_dir / "conceptops_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ConceptOps] Wrote manifest → {manifest_path}")
