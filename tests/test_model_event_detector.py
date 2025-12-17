# tests/test_model_event_detector.py

from pathlib import Path

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset


def _get_example_video(filename: str) -> Path:
    """
    Local helper to avoid import path issues with tests.utils.

    We assume example videos live under:
      videomask/examples/<filename>

    Adjust this if your repo stores sample videos elsewhere.
    """
    # This test file is at <repo_root>/tests/test_model_event_detector.py
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "examples" / filename
    if not p.exists():
        raise FileNotFoundError(f"Example video not found: {p}")
    return p


def test_model_event_detector_artifacts(tmp_path):
    """
    Integration-ish test: mode='model' should run end-to-end using real artifacts.

    We avoid relying on a non-existent "stub_v1" artifact directory.
    Instead we point to a known checked-in model dir.
    """
    video = _get_example_video("desk_demo.mp4")
    out_dir = tmp_path / "run_model"

    model_dir = Path("data/models/event_model_v0")
    assert model_dir.exists(), f"Expected model artifacts to exist at {model_dir}"
    assert (model_dir / "model.pt").exists(), f"Missing model.pt at {model_dir / 'model.pt'}"
    assert (model_dir / "labels.json").exists(), f"Missing labels.json at {model_dir / 'labels.json'}"

    episode_path_str = process_video_to_dataset(
        str(video),
        str(out_dir),
        vm_config={"fps": 2, "resize": 256, "backend": "dummy", "max_frames": 10},
        e2r_config={
            "mode": "model",
            "model_dir": str(model_dir),     # canonical
            "inference_profile": "default",  # don't use demo profiles in tests
            "min_score": 0.0,                # make the test robust even if model is weak
        },
    )

    episode_path = Path(episode_path_str)
    assert episode_path.exists(), "episode.json should be written"
    txt = episode_path.read_text(encoding="utf-8")
    assert '"events"' in txt, "episode.json should include events field"