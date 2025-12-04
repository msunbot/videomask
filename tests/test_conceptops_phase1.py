from pathlib import Path
import json

from conceptops.pipelines.mask_pipeline import run_conceptops_mask_pipeline


def test_mask_pipeline_smoke(tmp_path: Path):
    # Assumes a small test video exists under examples/
    video = Path("examples") / "demo_video.mp4"
    if not video.exists():
        # Skip if you don't have a test asset wired up yet
        return

    out_dir = tmp_path / "conceptops_phase1"
    run_conceptops_mask_pipeline(
        video_path=video,
        out_dir=out_dir,
        backend="dummy",
        fps=1.0,
        resize=256,
        max_frames=5,
    )

    assert (out_dir / "frames_raw").exists()
    assert (out_dir / "masks").exists()
    manifest_path = out_dir / "conceptops_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["stages"]["masks"] == "completed"
