# tests/test_exporters.py

from pathlib import Path
import json

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset
from conceptops.types import Episode
from conceptops.export.lerobot import episode_to_lerobot
from conceptops.export.rlds import episode_to_rlds
from conceptops.export.coco import export_coco_from_episode


def get_example_video(name: str) -> Path:
    return (Path(__file__).parent.parent / "examples" / name).resolve()


def test_exporters_basic(tmp_path):
    video = get_example_video("desk_demo.mp4")
    run_dir = tmp_path / "run_export"
    episode_path_str = process_video_to_dataset(
        str(video),
        str(run_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 5},
        e2r_config={"mode": "fixed", "frames_per_event": 2},
    )

    ep_path = Path(episode_path_str)
    ep_json = ep_path.read_text(encoding="utf-8")
    episode = Episode.from_json(ep_json)

    # LeRobot export
    lerobot = episode_to_lerobot(episode)
    assert "observations" in lerobot
    assert len(lerobot["observations"]["image_paths"]) == len(episode.frames)
    assert "events" in lerobot

    # RLDS export
    rlds = episode_to_rlds(episode)
    assert "steps" in rlds
    assert len(rlds["steps"]) == len(episode.frames)
    assert "observation" in rlds["steps"][0]

    # COCO export
    coco_path = run_dir / "episode_coco.json"
    coco_file = export_coco_from_episode(episode, coco_path)
    assert Path(coco_file).exists()

    coco = json.loads(Path(coco_file).read_text(encoding="utf-8"))
    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco