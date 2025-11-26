"""
examples/sam3_example.py

Example script demonstrating how to run the VideoSegmenter with the SAM-3 backend.

NOTE:
    This script must be run in an environment where:
    - `torch` has CUDA enabled
    - `sam3` is installed
    - You have a valid Hugging Face token configured for SAM-3
"""

from videomask.pipeline.segmenter import VideoSegmenter


def main() -> None:
    # Update this path to a real video file in your GPU environment
    video_path = "path/to/short_video.mp4"

    seg = VideoSegmenter(
        backend="sam3",
        fps=1,
        resize=512,
        max_frames=20,
        backend_kwargs={
            "device": "cuda",        # assume a GPU box
            "text_prompt": "person", # adjust for your use case
            "score_threshold": 0.0,
        },
    )

    out_dir = "outputs/sam3_example"
    seg.run(video_path, out_dir=out_dir)
    print(f"Done. Check '{out_dir}' for frames, masks, and metadata.json")


if __name__ == "__main__":
    main()