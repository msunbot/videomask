from __future__ import annotations

"""
Command-line interface for videomask.

Example:
    videomask segment input.mp4 --out outputs/run1 --backend dummy
"""

import click

from videomask.pipeline.segmenter import VideoSegmenter


@click.group()
def cli() -> None:
    """VideoMask: programmatic video segmentation into datasets."""
    # Group for subcommands. Currently we only expose `segment`.
    pass


@cli.command()
@click.argument("video_path", type=str)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=str,
    help="Output directory for frames, masks, and metadata.",
)
@click.option(
    "--backend",
    default="dummy",
    type=str,
    show_default=True,
    help='Segmentation backend to use, e.g. "dummy" or "sam3".',
)
@click.option(
    "--fps",
    default=2,
    type=int,
    show_default=True,
    help="Frame extraction rate (frames per second).",
)
@click.option(
    "--resize",
    default=512,
    type=int,
    show_default=True,
    help="Resize shorter side to this many pixels. Use 0 to keep original.",
)
@click.option(
    "--max-frames",
    default=None,
    type=int,
    help="Hard limit on number of frames processed (for quick tests).",
)
def segment(
    video_path: str,
    out_dir: str,
    backend: str,
    fps: int,
    resize: int,
    max_frames: int | None,
) -> None:
    """Run the segmentation pipeline on a video."""
    resize_arg: int | None = resize if resize > 0 else None

    seg = VideoSegmenter(
        backend=backend,  # "dummy" or "sam3"
        fps=fps,
        resize=resize_arg,
        max_frames=max_frames,
    )
    seg.run(video_path, out_dir)
    click.echo(f"Segmentation complete. Output at: {out_dir}")


def main() -> None:
    """Entrypoint for the `videomask` console script."""
    cli()