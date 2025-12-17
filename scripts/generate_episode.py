# scripts/generate_episode.py

from pathlib import Path
import argparse

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset


def generate_episode(video_path: Path, out_root: Path) -> None:
    """
    Run process_video_to_dataset on a single video and write episode.json
    into data/episodes/<clip_name>/.
    """
    clip_name = video_path.stem  # "clip_001" from "clip_001.mp4"
    out_dir = out_root / clip_name
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_path_str = process_video_to_dataset(
        video_path=str(video_path),
        out_dir=str(out_dir),
        vm_config={"fps": 2, "resize": 512, "backend": "dummy", "max_frames": 200},
        e2r_config={"mode": "fixed", "frames_per_event": 10},
    )

    print(f"[OK] Generated episode for {video_path} -> {episode_path_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (e.g. data/raw_clips/clip_001.mp4)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/episodes",
        help="Root directory where per-clip episode folders live.",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    out_root = Path(args.out_root).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    generate_episode(video_path, out_root)


if __name__ == "__main__":
    main()