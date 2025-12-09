# tests/test_video_ingest.py

from pathlib import Path

from conceptops.ingestion.video_ingest import ingest_video


def get_example_video(name: str) -> Path:
    """
    Helper for tests: resolve a video in conceptops/examples/.
    We intentionally resolve relative to this file's location,
    not the current working directory.
    """
    # tests/ -> repo-root/, then into examples/
    return (Path(__file__).parent.parent / "examples" / name).resolve()


def test_ingest_video_basic(tmp_path):
    video_path = get_example_video("desk_demo.mp4")
    out_dir = tmp_path / "ingest_run"

    result = ingest_video(
        video_path=video_path,
        out_dir=out_dir,
        fps=2,
        resize=512,
        max_frames=5,
    )

    # Basic sanity checks
    assert result.metadata.video_path == str(video_path)
    assert len(result.frame_paths) <= 5
    assert (out_dir / "frames_raw").exists()
    for f in result.frame_paths:
        assert Path(f).is_file()