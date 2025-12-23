"""
Egocentric-10K sample loader (Hugging Face datasets).

Reality check:
- builddotai/Egocentric-10K is marked as *gated* on HF. That means:
  - you can list files, but downloading typically requires accepting terms + auth token.
- The demo needs to handle "no access" gracefully with clear instructions.

Design:
- We provide a curated set of SampleSpecs.
- We attempt to load the dataset via `datasets` and extract a video sample.
- We cache the resulting MP4 locally so repeated runs are fast.

If gated access fails, we raise a helpful error message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# We import datasets lazily inside functions so the repo can still import this module
# even if datasets isn't installed yet.
# (Streamlit will show an error only when you click "prepare sample".)


@dataclass(frozen=True)
class Egocentric10KSampleSpec:
    dataset_id: str
    split: str
    index: int
    display_name: str


def get_default_sample_specs() -> List[Egocentric10KSampleSpec]:
    """
    Curated sample list.

    Note: indices are best-effort; the dataset may change.
    If an index is out of range, the loader will ask you to pick a different one.
    """
    return [
        Egocentric10KSampleSpec(
            dataset_id="builddotai/Egocentric-10K",
            split="train",
            index=0,
            display_name="BuildAI Egocentric-10K (train #0)",
        ),
        Egocentric10KSampleSpec(
            dataset_id="builddotai/Egocentric-10K",
            split="train",
            index=25,
            display_name="BuildAI Egocentric-10K (train #25)",
        ),
        # OPTIONAL fallback for local testing (not a substitute for the required dataset):
        Egocentric10KSampleSpec(
            dataset_id="Voxel51/Egocentric_10K_subset",
            split="train",
            index=0,
            display_name="Voxel51 subset (fallback for dev/testing)",
        ),
    ]


def _ensure_cache_dir() -> Path:
    cache_dir = Path(os.environ.get("CONCEPTOPS_DEMO_CACHE", ".conceptops_demo_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def prepare_egocentric10k_sample_video(spec: Egocentric10KSampleSpec) -> Path:
    """
    Download (or load-from-cache) a single video sample and write it to an MP4 file.

    Returns:
      Path to a local MP4 file.

    Expected dataset structures:
    - Many HF video datasets expose a column named 'video' whose element is a Video feature.
    - Some expose 'path' or 'file' columns.
    We try common patterns in a deterministic order.
    """
    cache_dir = _ensure_cache_dir()
    out_path = cache_dir / f"{spec.dataset_id.replace('/','__')}__{spec.split}__{spec.index}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: `datasets`. Install it with:\n"
            "  pip install datasets huggingface_hub\n"
            f"Original error: {e}"
        )

    # If the dataset is gated, HF usually requires a token.
    # `datasets` will pick up HF_TOKEN / HUGGINGFACEHUB_API_TOKEN env vars.
    # We explicitly mention HF_TOKEN in the UI text.
    ds = None
    try:
        ds = load_dataset(spec.dataset_id, split=spec.split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset {spec.dataset_id} [{spec.split}]. "
            "If this dataset is gated, you likely need to:\n"
            "  1) accept the dataset terms on Hugging Face\n"
            "  2) export HF_TOKEN=<your_token>\n"
            "  3) try again\n"
            f"Original error: {e}"
        )

    if spec.index < 0 or spec.index >= len(ds):
        raise IndexError(
            f"Sample index out of range: {spec.index} for split={spec.split} (len={len(ds)}). "
            "Pick a smaller index."
        )

    row = ds[spec.index]

    # Try common column names for videos.
    # HF Video feature often decodes into dict-like object with 'path' to a local cached file.
    candidate_keys = ["video", "videos", "mp4", "file", "path"]
    video_obj = None
    chosen_key = None
    for k in candidate_keys:
        if k in row:
            video_obj = row[k]
            chosen_key = k
            break

    if video_obj is None:
        # If we can't find a video column, show keys to help you adjust quickly.
        raise RuntimeError(
            f"Could not find a video column in dataset row. Available keys: {list(row.keys())}. "
            "Update `candidate_keys` or add dataset-specific handling."
        )

    # `datasets` Video feature decoding patterns:
    # - sometimes `video_obj` is a dict with {'path': '/.../something.mp4'}
    # - sometimes it's already a string path
    # - sometimes it's bytes-like (rare)
    src_path: Optional[Path] = None

    if isinstance(video_obj, str):
        src_path = Path(video_obj)
    elif isinstance(video_obj, dict) and "path" in video_obj:
        src_path = Path(video_obj["path"])
    else:
        # Best effort: try to access .get("path") style or attribute
        try:
            maybe = getattr(video_obj, "path", None)
            if maybe:
                src_path = Path(maybe)
        except Exception:
            src_path = None

    if src_path is None or not src_path.exists():
        raise RuntimeError(
            f"Found candidate video column '{chosen_key}', but could not resolve a local file path. "
            f"Value type: {type(video_obj)}. Value: {video_obj}"
        )

    # Copy the cached HF file into our demo cache with a stable filename.
    out_path.write_bytes(src_path.read_bytes())
    return out_path