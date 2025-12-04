import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class EpisodeConfig:
    out_dir: Path
    episode_id: int = 0

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "EpisodeConfig":
        return cls(
            out_dir=Path(args.out),
            episode_id=int(args.episode_id),
        )


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path, root: Path) -> str:
    """Return path relative to root, as posix string."""
    return os.path.relpath(path, root).replace("\\", "/")


def _load_manifest(out_dir: Path) -> Dict:
    manifest_path = out_dir / "conceptops_manifest.json"
    return _load_json(manifest_path)


def _save_manifest(out_dir: Path, manifest: Dict) -> None:
    manifest_path = out_dir / "conceptops_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_episode_payload(
    out_dir: Path,
    episode_id: int = 0,
) -> Dict[str, Any]:
    """
    Phase 4: Build a single-episode JSON payload from:
      - conceptops_manifest.json
      - metadata.json (from VideoMask)
      - events.json
      - concepts.json
    """
    manifest = _load_manifest(out_dir)

    metadata_path = Path(manifest["metadata_path"])
    metadata = _load_json(metadata_path)

    events_path = out_dir / "events.json"
    concepts_path = out_dir / "concepts.json"

    # These may not exist if stages were skipped
    events_payload = _load_json(events_path) if events_path.exists() else {"events": []}
    concepts_payload = (
        _load_json(concepts_path)
        if concepts_path.exists()
        else {"events_concepts": [], "labels_vocab": []}
    )

    events: List[Dict] = events_payload.get("events", [])
    events_concepts: List[Dict] = concepts_payload.get("events_concepts", [])
    labels_vocab: List[str] = concepts_payload.get("labels_vocab", [])

    root = out_dir

    video_path = Path(manifest["video_path"])
    frames_dir = Path(manifest["frames_dir"])
    masks_dir = Path(manifest["masks_dir"])

    # We assume VideoMask's metadata.json has ordered lists of frames/masks.
    frame_paths = [Path(p) for p in metadata.get("frames", [])]
    mask_paths = [Path(p) for p in metadata.get("masks", [])]

    # Fallback to scanning disk if metadata does not include frames/masks.
    if not frame_paths and frames_dir.exists():
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not mask_paths and masks_dir.exists():
        mask_paths = sorted(masks_dir.glob("mask_*.jpg"))

    num_frames = len(frame_paths)

    fps = float(manifest.get("fps", metadata.get("fps", 1.0)))

    thumbnails_dir = manifest.get("thumbnails_dir", None)

    episode = {
        "episode_id": episode_id,
        "version": "0.1.0",
        "fps": {
            "video": fps,
        },
        "paths": {
            "video": _rel(video_path, root=root),
            "frames_dir": _rel(frames_dir, root=root),
            "masks_dir": _rel(masks_dir, root=root),
            "events": _rel(events_path, root=root) if events_path.exists() else None,
            "concepts": _rel(concepts_path, root=root) if concepts_path.exists() else None,
        },
        "length": {
            "num_frames": num_frames,
            "num_events": len(events),
        },
        "observations": {
            "images": {
                "main": [_rel(p, root=root) for p in frame_paths],
            },
            "segmentation_masks": [_rel(p, root=root) for p in mask_paths],
        },
        "events": events,
        "events_concepts": events_concepts,
        "metadata": {
            "backend": manifest.get("backend", "unknown"),
            "labels_vocab": labels_vocab,
            "thumbnails_dir": (
                _rel(Path(thumbnails_dir), root=root) if thumbnails_dir else None
            ),
        },
    }

    return episode


def run_episode_stage(cfg: EpisodeConfig) -> Path:
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ConceptOps] Phase 4: building LeRobot-style episode JSON.")
    episode = build_episode_payload(out_dir=out_dir, episode_id=cfg.episode_id)

    episode_path = out_dir / "episode.json"
    with episode_path.open("w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2)

    # Update manifest
    manifest = _load_manifest(out_dir)
    manifest.setdefault("stages", {})
    manifest["stages"]["lerobot_episode"] = "completed"
    manifest["episode_path"] = str(episode_path)
    _save_manifest(out_dir, manifest)

    print(f"[ConceptOps] Wrote episode → {episode_path}")
    return episode_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptops-episode",
        description="ConceptOps LeRobot-style episode builder (Phase 4).",
    )
    parser.add_argument(
        "out",
        type=str,
        help="Output directory from previous stages (where conceptops_manifest.json lives).",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=0,
        help="Episode id to store in the JSON (default: 0).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = EpisodeConfig.from_args(args)
    run_episode_stage(cfg)


if __name__ == "__main__":
    main()
