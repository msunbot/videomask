from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from conceptops.types import Episode, EventRecord
from conceptops.training.model import TinyEventMLP
from conceptops.training.features import extract_window_features
from conceptops.training.proposals import sliding_window_proposals


def _load_episode(episode_dir: Path) -> Episode:
    episode_path = episode_dir / "episode.json"
    if not episode_path.exists():
        raise FileNotFoundError(f"episode.json not found: {episode_path}")

    # Episode.from_json expects JSON string in your codebase.
    txt = episode_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"episode.json is empty: {episode_path}")

    return Episode.from_json(txt)


def _load_model_artifacts(model_dir: Path) -> Tuple[TinyEventMLP, List[str]]:
    model_path = model_dir / "model.pt"
    labels_path = model_dir / "labels.json"
    feat_path = model_dir / "feature_spec.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_spec.json not found: {feat_path}")

    labels_payload = json.loads(labels_path.read_text())
    labels = labels_payload["labels"]
    num_classes = len(labels)

    feat_payload = json.loads(feat_path.read_text())
    in_dim = int(feat_payload["feature_dim"])

    model = TinyEventMLP(in_dim=in_dim, num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline event model inference on one episode.")
    parser.add_argument("--episode-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=8, help="Window size in frames (inclusive span).")
    parser.add_argument("--stride", type=int, default=4, help="Stride in frames.")
    parser.add_argument("--topk", type=int, default=5, help="Return top-K scored spans.")
    parser.add_argument("--out", type=str, default="pred_events.json", help="Output JSON file name in episode dir.")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    model_dir = Path(args.model_dir)

    episode = _load_episode(episode_dir)
    model, labels = _load_model_artifacts(model_dir)

    num_frames = len(episode.frames)
    proposals = sliding_window_proposals(num_frames, window_size=args.window_size, stride=args.stride)
    if not proposals:
        raise RuntimeError("No proposals generated (unexpected).")

    # Build feature matrix for all proposals
    X = []
    spans = []
    for sp in proposals:
        feats = extract_window_features(
            episode=episode,
            episode_dir=episode_dir,
            start_frame=sp.start_frame,
            end_frame=sp.end_frame,
        )
        X.append(feats)
        spans.append((sp.start_frame, sp.end_frame))

    X_t = torch.from_numpy(np.stack(X).astype(np.float32))
    with torch.no_grad():
        logits = model(X_t)
        probs = F.softmax(logits, dim=1).numpy()  # shape [N, C]
        best_cls = np.argmax(probs, axis=1)
        best_p = probs[np.arange(len(best_cls)), best_cls]

    # Rank proposals by confidence (descending)
    ranked = sorted(
        [(i, float(best_p[i]), int(best_cls[i]), spans[i][0], spans[i][1]) for i in range(len(spans))],
        key=lambda x: x[1],
        reverse=True,
    )

    topk = ranked[: max(1, args.topk)]

    # Convert to EventRecord list
    pred_events: List[EventRecord] = []
    for event_id, (i, conf, cls_id, s, e) in enumerate(topk):
        label = labels[cls_id]
        pred_events.append(
            EventRecord(
                event_id=event_id,
                label=label,
                start_frame=s,
                end_frame=e,
                score=conf,
                metadata={
                    "proposal_method": "sliding_window",
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "proposal_index": i,
                },
            )
        )

    out_path = episode_dir / args.out
    # We keep this output separate from episode.json for now (no pipeline integration yet).
    payload = [
        {
            "event_id": ev.event_id,
            "label": ev.label,
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
            "score": ev.score,
            "metadata": ev.metadata,
        }
        for ev in pred_events
    ]
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(pred_events)} predicted events to: {out_path}")


if __name__ == "__main__":
    main()