# VideoMask SDK

VideoMask is a Python-first SDK that turns raw videos into segmentation-ready datasets.

Core features (v0.1):
- Frame extraction via ffmpeg
- Pluggable segmentation backends (dummy, SAM-3)
- Lightweight temporal smoothing
- Simple folder-format export (frames, masks, metadata)
- CLI and Python API

## Installation

1. Clone the repository:

   git clone https://github.com/yourname/videomask.git  
   cd videomask

2. Create and activate a virtual environment (example using venv):

   python -m venv .venv  
   source .venv/bin/activate

3. Install the package in editable mode:

   pip install -e .

4. Install ffmpeg (example for macOS with Homebrew):

   brew install ffmpeg

## Quickstart (dummy backend, CPU)

Minimal Python usage:

- Import:

  from videomask.pipeline.segmenter import VideoSegmenter

- Use:

  seg = VideoSegmenter(  
      backend="dummy",  
      fps=2,  
      resize=512,  
      max_frames=30,  
  )  

  seg.run("path/to/video.mp4", out_dir="outputs/basic_example")

Result:
- frames saved under `outputs/basic_example/frames_raw/`  
- masks saved under `outputs/basic_example/masks/`  
- metadata stored in `outputs/basic_example/metadata.json`

## CLI Usage

From the project root (after installation):

- Dummy backend:

  videomask segment path/to/video.mp4 --out outputs/run1 --backend dummy

Key options:
- `--backend` (default: `dummy`)
- `--fps` (frames per second)
- `--resize` (shorter side in pixels, 0 to keep original)
- `--max-frames` (optional limit for quick tests)

Example:

  videomask segment path/to/video.mp4 --out outputs/run2 --backend dummy --fps 1 --resize 256

## SAM-3 Backend (GPU only)

The `sam3` backend requires:
- CUDA-enabled PyTorch
- The `sam3` library installed (from the official repo)
- A Hugging Face token with access to SAM-3

Typical setup (high-level):
1. Use a GPU environment (Colab or remote VM).
2. Install CUDA PyTorch (e.g. torch 2.4.1 with cu121 wheels).
3. Clone and install SAM-3 from the facebookresearch GitHub repo.
4. Log in to Hugging Face using `huggingface-cli login`.

Then, run:

- Python example (in a GPU environment):

  from videomask.pipeline.segmenter import VideoSegmenter  

  seg = VideoSegmenter(  
      backend="sam3",  
      fps=1,  
      resize=512,  
      max_frames=20,  
      backend_kwargs={  
          "device": "cuda",  
          "text_prompt": "person"  
      },  
  )  

  seg.run("path/to/video.mp4", out_dir="outputs/sam3_example")

- CLI example (GPU environment):

  videomask segment path/to/video.mp4 --out outputs/sam3_run --backend sam3 --fps 1 --resize 512

## Output Format

Folder export (v0.1):

- `out_dir/frames_raw/`  
  - Extracted RGB frames as images

- `out_dir/masks/`  
  - Binary masks as PNG (0 or 255 intensity)

- `out_dir/metadata.json`  
  - Lists of frame paths, mask paths, and run configuration

## Roadmap

See `ROADMAP.md` for:
- v0.2 plans (COCO export, mask strategies, additional backends)
- ConceptOps direction (concept-centric segmentation and datasets)