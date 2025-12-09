from pathlib import Path
from conceptops.pipelines.integrated_pipeline import process_video_to_dataset
import json

def get_example_video(name: str) -> Path:
    return (Path(__file__).parent.parent / "examples" / name).resolve()

def test_process_video_to_dataset(tmp_path):
    video = get_example_video("desk_demo.mp4")
    out_dir = tmp_path / "run"

    episode_path = process_video_to_dataset(
        video_path=str(video),
        out_dir=str(out_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 5},
    )

    ep_path = Path(episode_path)
    assert ep_path.exists()

    # Basic sanity: episode.json references frames + masks
    data = ep_path.read_text(encoding="utf-8")
    assert '"frames"' in data
    assert '"episode_id"' in data

def test_integrated_pipeline_has_events(tmp_path):
    video = get_example_video("desk_demo.mp4")
    out_dir = tmp_path / "run_with_events"

    episode_path_str = process_video_to_dataset(
        str(video),
        str(out_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 12},
        e2r_config={"frames_per_event": 4},
    )

    ep_path = Path(episode_path_str)
    data = json.loads(ep_path.read_text(encoding="utf-8"))

    assert "events" in data
    assert len(data["events"]) >= 2  # 12 frames / 4 frames_per_event = 3 events