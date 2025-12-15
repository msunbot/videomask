from __future__ import annotations

import argparse
from pathlib import Path

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset


def _default_clip_id(video_path: Path) -> str:
    """
    Create a stable clip id from filename.

    Keep it simple for Phase 3:
    - clip_id = file stem (e.g. "clip_001" from "clip_001.mp4")
    Later (Phase 5–6) we can use hashing + full provenance.
    """
    return video_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch build ConceptOps episodes from raw video clips.")
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Directory containing raw clips (e.g. data/raw_clips).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/episodes",
        help="Output directory for built episodes (default: data/episodes).",
    )
    parser.add_argument(
        "--vm-backend",
        type=str,
        default="dummy",
        help="VideoMask backend to use (default: dummy for infra work).",
    )
    parser.add_argument(
        "--e2r-mode",
        type=str,
        default="motion",
        choices=["simple", "motion", "model"],
        help="Event detection mode for building episode.json (default: motion).",
    )
    # Phase 3: allow model dir for e2r-mode=model
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to trained model dir (required if --e2r-mode model).",
    )

    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    out_root = Path(args.out_dir)

    if not videos_dir.exists():
        raise FileNotFoundError(f"--videos-dir not found: {videos_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Minimal configs.
    # We keep them plain dicts because your integrated pipeline already accepts vm_config/e2r_config.
    vm_config = {
        "backend": args.vm_backend,
    }
    e2r_config = {
        "mode": args.e2r_mode,
    }
    # pass model_dir + inference knobs 
    if args.e2r_mode == "model":
        if not args.model_dir:
            raise ValueError("--model-dir is required when --e2r-mode model")
        e2r_config["model_dir"] = args.model_dir

        # Optional knobs (keep defaults consistent with ModelEventDetector)
        e2r_config["window_size"] = 8
        e2r_config["stride"] = 4
        e2r_config["topk"] = 5
        e2r_config["min_score"] = 0.0
        e2r_config["nms_iou"] = 0.5

    # Iterate common video extensions. Add more if you need.
    exts = {".mp4", ".mov", ".mkv", ".avi"}
    video_paths = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in exts])

    if not video_paths:
        print(f"No videos found in {videos_dir} with extensions {sorted(exts)}")
        return

    for vp in video_paths:
        clip_id = _default_clip_id(vp)
        episode_dir = out_root / clip_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Building episode for clip_id={clip_id} ===")
        print(f"video:   {vp}")
        print(f"out_dir: {episode_dir}")

        # This is the important part: we use the EXISTING single entrypoint.
        # No parallel pipelines.
        process_video_to_dataset(
            video_path=str(vp),
            out_dir=str(episode_dir),
            vm_config=vm_config,
            e2r_config=e2r_config,
        )

        # Expected outputs (from your existing pipeline):
        # - episode.json
        # - frames_raw/
        # - masks/
        # - metadata.json
        #
        # We do NOT create labels here; labeling is a separate, explicit step.


if __name__ == "__main__":
    main()