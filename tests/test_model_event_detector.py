# tests/test_model_event_detector.py

from pathlib import Path
import json

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset


def get_example_video(name: str) -> Path:
    return (Path(__file__).parent.parent / "examples" / name).resolve()


def test_model_event_detector_stub(tmp_path):
    video = get_example_video("desk_demo.mp4")
    out_dir = tmp_path / "run_model"

    episode_path_str = process_video_to_dataset(
        str(video),
        str(out_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 10},
        e2r_config={"mode": "model", "model_name": "stub_v1"},
    )

    ep_path = Path(episode_path_str)
    data = json.loads(ep_path.read_text(encoding="utf-8"))

    assert "events" in data
    assert len(data["events"]) >= 1

    ev0 = data["events"][0]
    assert ev0["label"].startswith("model_event")
    assert "source" in ev0["metadata"]
    assert ev0["metadata"]["source"] == "model_stub"
    assert ev0["metadata"]["model_name"] == "stub_v1"