# tests/test_episode_roundtrip.py

from conceptops.types import VideoMetadata, FrameRecord, EventRecord, Episode


def test_episode_roundtrip_simple():
    # Build a tiny in-memory episode
    video_md = VideoMetadata(
        video_path="examples/dummy.mp4",
        original_fps=30.0,
        extraction_fps=10.0,
        num_frames=3,
        width=640,
        height=480,
        duration_sec=0.3,
        resize_short_side=None,
    )

    frames = [
        FrameRecord(index=0, image_path="frame_000.png", mask_path=None, timestamp_sec=0.0),
        FrameRecord(index=1, image_path="frame_001.png", mask_path="mask_001.png", timestamp_sec=0.1),
        FrameRecord(index=2, image_path="frame_002.png", mask_path="mask_002.png", timestamp_sec=0.2),
    ]

    events = [
        EventRecord(
            event_id=0,
            label="segment_0",
            start_frame=0,
            end_frame=1,
            score=0.9,
            metadata={"source": "test"},
        )
    ]

    episode = Episode(
        episode_id=42,
        video=video_md,
        frames=frames,
        events=events,
        extra={"note": "roundtrip test"},
    )

    # Round-trip through JSON
    json_str = episode.to_json(indent=2)
    restored = Episode.from_json(json_str)

    assert restored.episode_id == episode.episode_id
    assert restored.video.video_path == episode.video.video_path
    assert len(restored.frames) == len(episode.frames)
    assert len(restored.events) == len(episode.events)
    assert restored.extra["note"] == "roundtrip test"


def test_episode_roundtrip_pipeline_output(tmp_path):
    """
    Sanity check: load an actual episode.json written by the pipeline
    and ensure we can parse it back into an Episode.
    """
    from pathlib import Path
    from conceptops.pipelines.integrated_pipeline import process_video_to_dataset

    # locate example video
    examples_dir = Path(__file__).parent.parent / "examples"
    video_path = (examples_dir / "desk_demo.mp4").resolve()

    out_dir = tmp_path / "run"
    episode_path_str = process_video_to_dataset(
        str(video_path),
        str(out_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 10},
        e2r_config={"frames_per_event": 5},
    )

    episode_path = Path(episode_path_str)
    json_str = episode_path.read_text(encoding="utf-8")

    restored = Episode.from_json(json_str)

    # Basic invariants
    assert restored.video.video_path
    assert len(restored.frames) > 0
    assert len(restored.events) > 0