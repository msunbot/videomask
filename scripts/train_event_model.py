# scripts/train_event_model.py
"""
Train Phase 3 event classification model (ConceptOps).

Repo-aligned:
- load labeled events via iter_labeled_episodes (taxonomy-aware)
- featurize using extract_window_features (same as inference)

Adds:
- --label_collapse {none,demo3}
- --use_class_weights (recommended for your current imbalance)

Usage:
  python scripts/train_event_model.py \
    --episodes_root data/episodes \
    --taxonomy_path conceptops/config/event_taxonomy.json \
    --out_dir data/models/event_model_demo3_wt \
    --label_collapse demo3 \
    --use_class_weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from conceptops.training.model import TinyEventMLP
from conceptops.training.features import extract_window_features
from conceptops.training.dataset import iter_labeled_episodes, build_label_set


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_root", type=str, default="data/episodes")
    ap.add_argument("--taxonomy_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/models/event_model_demo3_wt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--label_collapse", type=str, default="none")
    ap.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss (helps prevent collapse).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episodes_root = Path(args.episodes_root)
    taxonomy_path = Path(args.taxonomy_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_eps = list(
        iter_labeled_episodes(
            episodes_root=episodes_root,
            taxonomy_path=taxonomy_path,
            require_labels=True,
            label_collapse=args.label_collapse,
        )
    )
    if not labeled_eps:
        raise RuntimeError("No labeled episodes found.")

    labels = build_label_set(labeled_eps)
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(labels)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    label_counts: Dict[str, int] = {lab: 0 for lab in labels}

    for le in labeled_eps:
        episode = le.episode
        ep_dir = le.episode_dir
        for ev in le.events:
            feats = extract_window_features(
                episode=episode,
                episode_dir=ep_dir,
                start_frame=int(ev.start_frame),
                end_frame=int(ev.end_frame),
            )
            X_list.append(feats)
            y_list.append(label_to_id[ev.label])
            label_counts[ev.label] += 1

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    in_dim = int(X.shape[1])
    model = TinyEventMLP(in_dim=in_dim, num_classes=num_classes)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # --- NEW: class weights ---
    class_weights_t = None
    if args.use_class_weights:
        # Inverse frequency weighting (simple, effective at small scale).
        # weight[c] = total / (num_classes * count[c])
        total = float(len(y_list))
        weights = []
        for lab in labels:
            c = float(label_counts[lab])
            w = total / (max(1.0, c) * float(num_classes))
            weights.append(w)
        class_weights_t = torch.tensor(weights, dtype=torch.float32)
        print("[train] class weights:", {labels[i]: float(weights[i]) for i in range(len(labels))})

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    model.train()
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if epoch in (0, 1, 2) or (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = float((pred == y_t).float().mean().item())
                # Also print predicted label distribution to detect collapse
                pred_counts = {}
                for p in pred.numpy().tolist():
                    lab = labels[int(p)]
                    pred_counts[lab] = pred_counts.get(lab, 0) + 1
            print(f"[train] epoch={epoch+1:03d} loss={loss.item():.4f} acc={acc:.3f} pred_counts={pred_counts}")

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), str(model_path))

    labels_obj: Dict[str, Any] = {
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
    }
    save_json(out_dir / "labels.json", labels_obj)

    feature_spec = {
        "feature_dim": int(in_dim),
        "label_collapse": args.label_collapse,
        "label_counts": label_counts,
        "episodes_root": str(episodes_root),
        "taxonomy_path": str(taxonomy_path),
        "num_examples": int(len(y_list)),
        "num_episodes": int(len(labeled_eps)),
        "use_class_weights": bool(args.use_class_weights),
    }
    save_json(out_dir / "feature_spec.json", feature_spec)

    print("\nSaved:")
    print(f"- {model_path}")
    print(f"- {out_dir / 'labels.json'}")
    print(f"- {out_dir / 'feature_spec.json'}")
    print("\nLabel counts:", label_counts)
    print("Labels:", labels)


if __name__ == "__main__":
    main()