
"""
Day 1 smoke test:
- extracts frames with ffmpeg
- runs DummyBackend
- writes frames + masks + metadata.json
"""
from videomask.pipeline.segmenter import VideoSegmenter

def main() -> None:
    seg = VideoSegmenter(
        backend="dummy",
        fps=1,
        resize=256,
        max_frames=10,
    )
    # TODO: replace this with a real short .mp4 path on your machine
    video_path = "examples/fish.mp4"

    seg.run(video_path, out_dir="outputs/day1_test")
    print("Done. Check outputs/day1_test/ for frames, masks, and metadata.json")

if __name__ == "__main__":
    main()
