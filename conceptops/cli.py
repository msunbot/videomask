import sys
import argparse
from pathlib import Path
import shutil

from conceptops.pipelines.mask_pipeline import run_conceptops_mask_pipeline
from conceptops.core.config import MaskPipelineConfig
from conceptops.core.events import EventConfig, run_event_stage
from conceptops.core.concepts import ConceptConfig, run_concept_stage, DEFAULT_LABELS
from conceptops.core.episode import EpisodeConfig, run_episode_stage

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptops",
        description="ConceptOps MVP: raw video → masks (via VideoMask). Phase 0 stub.",
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (e.g. .mp4).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/conceptops_run",
        help="Output directory for intermediate artifacts.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=["dummy", "sam3"],
        help="Segmentation backend (uses VideoMask backends).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for extraction.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Shorter side pixel size (0 to keep original).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Optional frame cap for quick runs.",
    )
    parser.add_argument(
        "--event-iou-threshold",
        type=float,
        default=0.9,
        help="IoU threshold for starting a new event (default: 0.9).",
    )
    parser.add_argument(
        "--event-min-length",
        type=int,
        default=2,
        help="Minimum number of frames per event (default: 2).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional override of CLIP label vocab. If omitted, uses DEFAULT_LABELS.",
    )
    parser.add_argument(
        "--concept-top-k",
        type=int,
        default=3,
        help="Number of top CLIP concepts to keep per event (default: 3).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    # Allow: `conceptops run input.mp4 --out outdir/ ...`
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        argv = argv[1:]

    args = parser.parse_args(argv)

    cfg = MaskPipelineConfig.from_args(args)

    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Input video not found: {cfg.video_path}")

    print("[ConceptOps] Phase 1: masks via VideoMask (config-driven).")
    print(f"  - video   : {cfg.video_path}")
    print(f"  - out_dir : {cfg.out_dir}")
    print(f"  - backend : {cfg.backend}")
    print(f"  - fps     : {cfg.fps}")
    print(f"  - resize  : {cfg.resize}")
    print(f"  - max_frames: {cfg.max_frames}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: masks
    run_conceptops_mask_pipeline(
        video_path=cfg.video_path,
        out_dir=cfg.out_dir,
        backend=cfg.backend,
        fps=cfg.fps,
        resize=cfg.resize,
        max_frames=cfg.max_frames,
    )

    print("[ConceptOps] Phase 1 complete: frames + masks generated via VideoMask.")

    # Stage 2: events (using defaults for now)
    print("[ConceptOps] Phase 2: temporal events (IoU-based).")
    event_cfg = EventConfig(
        out_dir=cfg.out_dir,
        iou_threshold=args.event_iou_threshold,
        min_event_length=args.event_min_length,
    )

    print("[ConceptOps] Masks + events ready. Next: concepts → LeRobot episodes.")

    # Stage 3: concepts (CLIP)
    print("[ConceptOps] Phase 3: CLIP concept tagging.")
    labels = args.labels if args.labels else DEFAULT_LABELS
    concept_cfg = ConceptConfig(
        out_dir=cfg.out_dir,
        labels=labels,
        top_k=args.concept_top_k,
    )

    print("[ConceptOps] Masks + events + concepts ready. Next: LeRobot episodes.")

    # Stage 4: LeRobot-style episode JSON
    print("[ConceptOps] Phase 4: building episode JSON.")
    episode_cfg = EpisodeConfig(out_dir=cfg.out_dir, episode_id=0)
    run_episode_stage(episode_cfg)

    print("[ConceptOps] Full pipeline complete: video → masks → events → concepts → episode.")

    # Optional: copy demo notebook template into run folder
    template_nb = Path(__file__).resolve().parent / "demos" / "ConceptOps_Demo_Template.ipynb"
    target_nb = cfg.out_dir / "demo.ipynb"
    if template_nb.exists():
        shutil.copy(template_nb, target_nb)
        print(f"[ConceptOps] Copied demo notebook → {target_nb}")
    else:
        print("[ConceptOps] NOTE: No demo notebook template found. "
              "Create conceptops/demos/ConceptOps_Demo_Template.ipynb to enable this.")

    print("[ConceptOps] Full pipeline complete: video → masks → events → concepts → episode.")

if __name__ == "__main__":
    main()
