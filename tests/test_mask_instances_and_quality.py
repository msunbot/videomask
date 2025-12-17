from pathlib import Path
import json

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset


def test_episode_has_mask_instances_and_quality(tmp_path):
    examples_dir = Path(__file__).parent.parent / "examples"
    video = (examples_dir / "desk_demo.mp4").resolve()

    out_dir = tmp_path / "run_mask_metrics"
    episode_path_str = process_video_to_dataset(
        str(video),
        str(out_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 5},
        e2r_config={"mode": "fixed", "frames_per_event": 2},
    )

    ep_path = Path(episode_path_str)
    data = json.loads(ep_path.read_text(encoding="utf-8"))

    assert len(data["frames"]) > 0

    # At least one frame should have mask_quality + instances
    frames_with_masks = [
        f for f in data["frames"]
        if f.get("mask_path") is not None
    ]
    assert frames_with_masks, "Expected at least one frame with a mask"

    f0 = frames_with_masks[0]
    assert "mask_quality" in f0["metadata"]
    assert "area_ratio" in f0["metadata"]["mask_quality"]
    assert "instances" in f0
    assert len(f0["instances"]) >= 1