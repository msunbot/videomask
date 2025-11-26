from videomask.pipeline.segmenter import VideoSegmenter

if __name__ == "__main__":
    seg = VideoSegmenter(
        backend="dummy",
        fps=2,
        resize=256,
        max_frames=30,
    )
    seg.run("examples/fish.mp4", out_dir="outputs/day2_test")
    print("Done. Check outputs/day2_test/frames_raw, masks, metadata.json")