import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
import clip

THUMBNAIL_SIZE = (256, 256)
MIN_TOP_SCORE = 0.25 # below this, label as 'uncertain'

DEFAULT_LABELS = [
    "human hand",
    "robot arm",
    "box",
    "drawer",
    "bottle",
    "tool",
    "table",
    "conveyor belt",
    "keyboard",
    "monitor",
]


@dataclass
class ConceptConfig:
    out_dir: Path
    labels: List[str]
    top_k: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "ConceptConfig":
        labels = args.labels if args.labels else DEFAULT_LABELS
        return cls(
            out_dir=Path(args.out),
            labels=labels,
            top_k=int(args.top_k),
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )


def _load_manifest(out_dir: Path) -> Dict:
    manifest_path = out_dir / "conceptops_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"conceptops_manifest.json not found under {out_dir}. "
            "Run the mask + event stages first."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_events(out_dir: Path) -> List[Dict]:
    events_path = out_dir / "events.json"
    if not events_path.exists():
        raise FileNotFoundError(
            f"events.json not found under {out_dir}. "
            "Run the event stage first."
        )
    with events_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("events", [])


def _load_clip_model(device: str):
    print(f"[ConceptOps] Loading CLIP on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def _encode_labels(model, labels: List[str], device: str) -> torch.Tensor:
    with torch.no_grad():
        text_tokens = clip.tokenize(labels).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # [num_labels, dim]


def _build_frame_path_from_mask(mask_path: Path, frames_dir: Path) -> Path:
    """
    Heuristic: VideoMask uses 'frame_000009.jpg' and 'mask_000009.jpg'.
    We map the mask filename into frame filename by replacing the prefix.
    """
    name = mask_path.name
    if name.startswith("mask_"):
        frame_name = "frame_" + name[len("mask_") :]
    else:
        # fallback: reuse the same name, assume frames share names
        frame_name = name
    return frames_dir / frame_name


def _score_image(
    model,
    preprocess,
    image_path: Path,
    text_features: torch.Tensor,
    device: str,
) -> Tuple[List[int], List[float]]:
    img = Image.open(image_path).convert("RGB")
    image_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        logits = 100.0 * image_features @ text_features.T  # [1, num_labels]
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    # Return indices sorted by probability, descending
    sorted_idx = probs.argsort()[::-1]
    sorted_probs = probs[sorted_idx]
    return list(sorted_idx), list(sorted_probs)

def _save_thumbnail(frame_path: Path, thumb_dir: Path, event_id: int) -> Path:
    """
    Save a small thumbnail for the given frame and return its path.
    """
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"event_{event_id:04d}.jpg"

    img = Image.open(frame_path).convert("RGB")
    img.thumbnail(THUMBNAIL_SIZE)
    img.save(thumb_path, format="JPEG", quality=90)

    return thumb_path

def run_concept_stage(cfg: ConceptConfig) -> Path:
    """
    Phase 3: CLIP concept tagging.
    For each event, take the key frame, run CLIP against candidate labels,
    and store the top_k concepts.
    """
    out_dir = cfg.out_dir
    manifest = _load_manifest(out_dir)
    events = _load_events(out_dir)

    frames_dir = Path(manifest["frames_dir"])
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    if not events:
        print("[ConceptOps] No events found. Nothing to tag.")
        return out_dir / "concepts.json"

    thumbnails_dir = out_dir / "thumbnails"

    model, preprocess = _load_clip_model(cfg.device)
    text_features = _encode_labels(model, cfg.labels, cfg.device)

    concepts: List[Dict] = []

    print(f"[ConceptOps] Tagging {len(events)} events with {len(cfg.labels)} labels...")
    for ev in events:
        key_mask_path = Path(ev["key_frame_path"])
        frame_path = _build_frame_path_from_mask(key_mask_path, frames_dir)

        if not frame_path.exists():
            print(f"[ConceptOps] WARNING: frame not found for event {ev['event_id']}: {frame_path}")
            continue

        indices, probs = _score_image(model, preprocess, frame_path, text_features, cfg.device)

        top_k = min(cfg.top_k, len(indices))
        top_labels = [cfg.labels[i] for i in indices[:top_k]]
        top_scores = [float(p) for p in probs[:top_k]]

        top_score = top_scores[0] if top_scores else 0.0
        is_uncertain = top_score < MIN_TOP_SCORE
        thumb_path = _save_thumbnail(frame_path, thumbnails_dir, ev["event_id"])

        concepts.append(
            {
                "event_id": ev["event_id"],
                "frame_path": str(frame_path),
                "thumbnail_path": str(thumb_path),
                "labels": top_labels,
                "scores": top_scores,
                "top_score": top_score,
                "uncertain": is_uncertain,
            }
        )

    concepts_path = out_dir / "concepts.json"
    with concepts_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_vocab": cfg.labels,
                "events_concepts": concepts,
            },
            f,
            indent=2,
        )

    # Update manifest
    manifest.setdefault("stages", {})
    manifest["stages"]["concepts"] = "completed"
    manifest["concepts_path"] = str(concepts_path)
    manifest["labels_vocab"] = cfg.labels
    manifest["thumbnails_dir"] = str(thumbnails_dir)

    manifest_path = out_dir / "conceptops_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ConceptOps] Wrote concepts → {concepts_path}")
    print(f"[ConceptOps] Tagged {len(concepts)} events.")
    return concepts_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptops-concepts",
        description="ConceptOps CLIP-based concept tagging (Phase 3).",
    )
    parser.add_argument(
        "out",
        type=str,
        help="Output directory from previous stages (where conceptops_manifest.json lives).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional override of label list. If omitted, uses DEFAULT_LABELS.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top concepts to store per event.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (e.g. 'cpu', 'cuda'). Default: cuda if available, else cpu.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = ConceptConfig.from_args(args)
    run_concept_stage(cfg)


if __name__ == "__main__":
    main()
