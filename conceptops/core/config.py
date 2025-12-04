from dataclasses import dataclass
from pathlib import Path


@dataclass
class MaskPipelineConfig:
    video_path: Path
    out_dir: Path
    backend: str = "dummy"
    fps: float = 1.0
    resize: int = 256
    max_frames: int = 30

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "MaskPipelineConfig":
        return cls(
            video_path=Path(args.video),
            out_dir=Path(args.out),
            backend=args.backend,
            fps=float(args.fps),
            resize=int(args.resize),
            max_frames=int(args.max_frames),
        )
