# VideoMask SDK & ConceptOps

VideoMask is a Python-first SDK that turns raw videos into segmentation-ready datasets.  
ConceptOps is a higher-level pipeline built on top of VideoMask that turns **video → masks → temporal events → semantic concepts → LeRobot-style episode JSON**.

---

## Part 1 – VideoMask SDK (v0.1)

Core features:
- Frame extraction via ffmpeg  
- Pluggable segmentation backends (`dummy`, `sam3`)  
- Lightweight temporal smoothing  
- Folder-format export (frames, masks, metadata)  
- CLI + Python API  

### Installation

1. Clone the repo:  
   git clone https://github.com/msunbot/videomask.git  
   cd videomask  

2. Create venv:  
   python -m venv .venv  
   source .venv/bin/activate  

3. Install package:  
   pip install -e .  

4. Install ffmpeg (macOS):  
   brew install ffmpeg  

### Quickstart (dummy backend, CPU)

Python example:

    from videomask.pipeline.segmenter import VideoSegmenter
    seg = VideoSegmenter(backend="dummy", fps=2, resize=512, max_frames=30)
    seg.run("path/to/video.mp4", out_dir="outputs/basic_example")

CLI example:

    videomask segment path/to/video.mp4 --out outputs/run1 --backend dummy

### SAM-3 Backend (GPU)

Requires: CUDA PyTorch, SAM-3 library, Hugging Face token.

High-level steps (e.g. in Colab):
- Switch to GPU runtime  
- Install torch + sam3  
- Login to Hugging Face  
- Run:

    videomask segment path/to/video.mp4 --out outputs/sam3_run --backend sam3 --fps 1 --resize 512

### VideoMask Output Format

`out_dir` contains:
- frames_raw/   (RGB frames)  
- masks/        (binary masks)  
- metadata.json  

---

## Part 2 – ConceptOps MVP (video → events → concepts → episode)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/msunbot/videomask/blob/main/conceptops/demos/ConceptOps_Demo_Template.ipynb
)

ConceptOps extends VideoMask as:

    raw video
      → frames + masks (SAM-3 or dummy)
      → temporal events (IoU + centroid + area heuristics)
      → CLIP concepts (semantic labels + uncertainty)
      → episode.json (LeRobot-style)

### What ConceptOps produces

Given an input video, ConceptOps creates:
- frames_raw/  
- masks/  
- metadata.json  
- conceptops_manifest.json  
- events.json  
- concepts.json  
- thumbnails/  
- episode.json  
- demo.ipynb  (copy of visualization template, if present)

### ConceptOps CLI (single command)

    conceptops run <video_path> \
      --out <out_dir> \
      --backend {dummy,sam3} \
      --fps 2 \
      --resize 320 \
      --max-frames 40 \
      --event-iou-threshold 0.7 \
      --event-min-length 3 \
      --labels label1 label2 ... \
      --concept-top-k 3

---

## Quickstart – Local CPU (dummy backend) with desk_demo.mp4

Assuming you have a small demo clip at `examples/desk_demo.mp4`:

    conceptops run examples/desk_demo.mp4 \
      --out outputs/desk_demo_dummy \
      --backend dummy \
      --fps 2 \
      --resize 320 \
      --max-frames 40 \
      --event-iou-threshold 0.7 \
      --event-min-length 3 \
      --labels laptop keyboard monitor mouse "coffee mug" phone notebook

This will create:

    outputs/desk_demo_dummy/
      frames_raw/
      masks/
      metadata.json
      conceptops_manifest.json
      events.json
      concepts.json
      thumbnails/
      episode.json
      demo.ipynb    (if ConceptOps_Demo_Template.ipynb exists in conceptops/demos)

You can then open `outputs/desk_demo_dummy/demo.ipynb` or the template notebook and set:

    RUN_DIR = Path("outputs/desk_demo_dummy")

to visualize events and labels.

---

## Quickstart – GPU (SAM-3 in Colab)

In a Colab notebook:

    !git clone https://github.com/msunbot/videomask.git
    %cd videomask
    !pip install -e .
    !pip install git+https://github.com/openai/CLIP.git matplotlib huggingface_hub

Login to Hugging Face:

    from huggingface_hub import login
    login()  # paste token

Fix SAM-3 BPE vocab (expected path):

    !mkdir -p /usr/local/lib/python3.12/dist-packages/assets
    !wget -q \
      https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz \
      -O /usr/local/lib/python3.12/dist-packages/assets/bpe_simple_vocab_16e6.txt.gz

Upload a test video to `data/` (e.g. `hero_sam3.mp4`), then run:

    conceptops run data/hero_sam3.mp4 \
      --out outputs/hero_sam3 \
      --backend sam3 \
      --fps 2 \
      --resize 320 \
      --max-frames 40 \
      --event-iou-threshold 0.7 \
      --event-min-length 3 \
      --labels "human hand" "person" "robot arm" "box" "drawer" "cup" "bottle" "tool" "table" "keyboard" "monitor"

Then open the demo notebook template:

- `conceptops/demos/ConceptOps_Demo_Template.ipynb`

Set:

    RUN_DIR = Path("outputs/hero_sam3")

and run all cells to view:
- stage status  
- event summary  
- thumbnails with CLIP labels  
- flipbook of frames per event  

---

## ConceptOps Output Files

**events.json**  
Each event has frame range + timestamps, for example:

    {
      "event_id": 0,
      "start_frame": 2,
      "end_frame": 4,
      "num_frames": 3,
      "start_time_sec": 1.0,
      "end_time_sec": 2.5,
      "key_frame_index": 3,
      "key_frame_path": "outputs/.../masks/mask_000004.jpg"
    }

**concepts.json**  
CLIP labels and scores per event, plus uncertainty flag:

    {
      "event_id": 0,
      "frame_path": "outputs/.../frames_raw/frame_000004.jpg",
      "thumbnail_path": "outputs/.../thumbnails/event_0000.jpg",
      "labels": ["laptop", "keyboard", "coffee mug"],
      "scores": [0.42, 0.31, 0.12],
      "top_score": 0.42,
      "uncertain": false
    }

**episode.json**  
Combined LeRobot-style payload containing:
- fps and path metadata  
- observations (image + segmentation mask paths)  
- events (from events.json)  
- events_concepts (from concepts.json)  
- metadata (backend info, labels_vocab, thumbnails_dir)

---

## Example Video

A small desk demo clip (`examples/desk_demo.mp4`) is recommended for local testing.

To run:

    conceptops run examples/desk_demo.mp4 \
      --out outputs/desk_demo_dummy \
      --backend dummy

---
## Architecture Diagram
+------------------+
|   Input Video    |
+------------------+
          |
          v
+------------------+
|  Frame Extractor |
|  (ffmpeg)        |
+------------------+
          |
          v
+---------------------------+
|  Segmentation Backend     |
|  (SAM-3 or Dummy)         |
+---------------------------+
          |
          v
+-----------------------------+
|  frames_raw/  masks/       |
|  metadata.json             |
+-----------------------------+
          |
          v
+-------------------------------+
|   Event Segmenter             |
|   (IoU + centroid + area)     |
+-------------------------------+
          |
          v
+----------------------+
|     events.json      |
+----------------------+
          |
          v
+------------------------------+
|        CLIP Tagger           |
|   (top-k labels + uncertain) |
+------------------------------+
          |
          v
+----------------------+
|   concepts.json      |
|   thumbnails/        |
+----------------------+
          |
          v
+-----------------------------+
|     Episode Builder         |
|     (LeRobot-style JSON)    |
+-----------------------------+
          |
          v
+----------------------+
|     episode.json     |
+----------------------+

---

## Roadmap

Planned:
- Additional segmentation backends  
- Richer event heuristics  
- Domain-specific CLIP vocabularies  
- Additional episode formats (downstream)  
- Notebook and visualization improvements  