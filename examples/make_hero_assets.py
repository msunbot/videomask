"""
examples/make_hero_assets.py

Generate launch assets from a VideoMask SAM-3 run:

- hero_raw.jpg          : a selected raw frame
- hero_overlay.jpg      : the same frame with the mask overlaid in red
- hero_side_by_side.jpg : raw + overlay side-by-side
- hero_hero.gif         : simple GIF cycling through [raw, overlay, side_by_side]

Usage:
    python examples/make_hero_assets.py

Notes:
    - Adjust OUTPUT_DIR if your pipeline output path is different.
    - Adjust FRAME_INDEX if you want a specific moment, or leave as None to auto-pick middle frame.
"""

import os
from typing import Optional

import numpy as np
from PIL import Image
import imageio.v2 as imageio


# === CONFIG ===
OUTPUT_DIR = "outputs/sam3_demo"  # folder created by your SAM-3 run
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames_raw")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks_raw")

# If None, pick the middle frame
FRAME_INDEX: Optional[int] = None

ASSET_DIR = os.path.join(OUTPUT_DIR, "launch_assets")
os.makedirs(ASSET_DIR, exist_ok=True)


def _sorted_files(folder: str):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return files


def pick_frame_and_mask() -> tuple[str, str]:
    """
    Pick a frame + mask pair by index.

    If FRAME_INDEX is None, pick the middle frame.
    """
    frames = _sorted_files(FRAMES_DIR)
    masks = _sorted_files(MASKS_DIR)

    if not frames:
        raise RuntimeError(f"No frames found in {FRAMES_DIR}")
    if not masks:
        raise RuntimeError(f"No masks found in {MASKS_DIR}")

    if len(frames) != len(masks):
        print(
            f"[WARN] Frame count ({len(frames)}) != mask count ({len(masks)}). "
            f"Will match by index; ensure they correspond."
        )

    if FRAME_INDEX is None:
        idx = len(frames) // 2
    else:
        idx = FRAME_INDEX
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"FRAME_INDEX {idx} out of range for {len(frames)} frames")

    frame_path = os.path.join(FRAMES_DIR, frames[idx])
    mask_path = os.path.join(MASKS_DIR, masks[idx])

    print(f"[INFO] Using frame: {frame_path}")
    print(f"[INFO] Using mask : {mask_path}")

    return frame_path, mask_path


def make_overlay(frame_path: str, mask_path: str) -> tuple[str, str, str]:
    """
    Generate:
        hero_raw.jpg
        hero_overlay.jpg
        hero_side_by_side.jpg

    Returns the three output file paths.
    """
    # Load images
    frame = Image.open(frame_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")  # single-channel

    frame_np = np.array(frame)
    mask_np = np.array(mask_img)

    # Normalize mask to {0,1}
    mask_bin = (mask_np > 127).astype(np.uint8)

    # Create overlay: red where mask == 1
    overlay_np = frame_np.copy()
    overlay_np[mask_bin == 1] = [255, 0, 0]  # red

    # Blend original + red overlay for nicer look
    blended_np = (0.6 * frame_np + 0.4 * overlay_np).astype(np.uint8)

    # Save hero_raw
    raw_path = os.path.join(ASSET_DIR, "hero_raw.jpg")
    frame.save(raw_path, quality=95)

    # Save hero_overlay
    overlay_path = os.path.join(ASSET_DIR, "hero_overlay.jpg")
    Image.fromarray(blended_np).save(overlay_path, quality=95)

    # Side-by-side canvas
    w, h = frame.size
    side_by_side = Image.new("RGB", (w * 2, h))
    side_by_side.paste(frame, (0, 0))
    side_by_side.paste(Image.fromarray(blended_np), (w, 0))

    side_path = os.path.join(ASSET_DIR, "hero_side_by_side.jpg")
    side_by_side.save(side_path, quality=95)

    print(f"[OK] Saved raw frame       -> {raw_path}")
    print(f"[OK] Saved overlay frame   -> {overlay_path}")
    print(f"[OK] Saved side-by-side    -> {side_path}")

    return raw_path, overlay_path, side_path


def make_gif(raw_path: str, overlay_path: str, side_path: str) -> str:
    """
    Make a simple GIF cycling through raw -> overlay -> side-by-side.

    You can add folder tree and title card later in a video editor if desired.
    """
    frames = []
    durations = []

    for path, dur in [(raw_path, 800), (overlay_path, 800), (side_path, 1000)]:
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img))
        durations.append(dur)

    gif_path = os.path.join(ASSET_DIR, "hero.gif")
    imageio.mimsave(gif_path, frames, duration=[d / 1000.0 for d in durations])

    print(f"[OK] Saved GIF             -> {gif_path}")
    return gif_path


def main():
    frame_path, mask_path = pick_frame_and_mask()
    raw_path, overlay_path, side_path = make_overlay(frame_path, mask_path)
    make_gif(raw_path, overlay_path, side_path)
    print("[DONE] Launch assets created in:", ASSET_DIR)


if __name__ == "__main__":
    main()