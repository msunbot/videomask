from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from conceptops.training.dataset import iter_labeled_episodes, build_label_set, normalize_events_to_nonoverlapping_spans
from conceptops.training.features import extract_window_features
from conceptops.training.model import TinyEventMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 baseline event model from labeled episodes.")
    parser.add_argument("--episodes-root", type=str, default="data/episodes")
    parser.add_argument("--taxonomy", type=str, default="config/event_taxonomy.json")
    parser.add_argument("--out-dir", type=str, default="data/models/event_model_v0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-mode", type=str, default="span", choices=["span", "frame"])
    args = parser.parse_args()

    episodes_root = Path(args.episodes_root)
    taxonomy_path = Path(args.taxonomy)
    # People will naturally pass "config/event_taxonomy.json".
    # In this repo, the canonical file lives at "conceptops/config/event_taxonomy.json".
    # So if the user-provided path doesn't exist, try the common fallback.
    if not taxonomy_path.exists():
        candidate = Path("conceptops") / taxonomy_path  # e.g. conceptops/config/event_taxonomy.json
        if candidate.exists():
            taxonomy_path = candidate
        else:
            raise FileNotFoundError(
                f"Taxonomy file not found: {args.taxonomy} (also tried {candidate})"
            )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load labeled episodes
    labeled_eps = list(iter_labeled_episodes(episodes_root, taxonomy_path, require_labels=True))
    if not labeled_eps:
        raise RuntimeError("No labeled episodes found. Create event_labels.json for at least 1 clip first.")

    # 2) Label set present in data (training only on what you actually labeled)
    labels = build_label_set(labeled_eps)
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    print(f"Found {len(labeled_eps)} labeled episodes.")
    print(f"Training labels ({len(labels)}): {labels}")

    # 3) Build training dataset: one sample per labeled span
    X_list = []
    y_list = []

    # Phase 3: frame-mode sample builder
    def _frame_labels_from_spans(spans, num_frames: int, background="__none__"):
        labels = [background] * num_frames
        for (s, e, lab) in spans:
            for i in range(max(0, s), min(num_frames - 1, e) + 1):
                labels[i] = lab
        return labels

    # Train on normalized non-overlapping spans
    # Phase 3: span vs frame training
    for le in labeled_eps:
        spans = normalize_events_to_nonoverlapping_spans(le.episode, le.events)
        num_frames = len(le.episode.frames)

        if args.train_mode == "span":
            for (s, e, label) in spans:
                feats = extract_window_features(le.episode, le.episode_dir, s, e)
                X_list.append(feats)
                y_list.append(label_to_id[label])

        else:  # frame mode
            frame_labels = _frame_labels_from_spans(spans, num_frames=num_frames, background="__none__")

            # Train only on labeled frames (skip background) to start.
            for t, lab in enumerate(frame_labels):
                if lab == "__none__":
                    continue

                # Small symmetric window around frame t
                half = 4
                s = max(0, t - half)
                e = min(num_frames - 1, t + half)

                feats = extract_window_features(le.episode, le.episode_dir, s, e)
                X_list.append(feats)
                y_list.append(label_to_id[lab])

    print(f"Built {len(y)} training samples (one per labeled span).")

    # 4) Train tiny model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyEventMLP(in_dim=X.shape[1], num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    for epoch in range(args.epochs):
        model.train()
        logits = model(X_t)
        loss = F.cross_entropy(logits, y_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y_t).float().mean().item()
            print(f"epoch {epoch+1:03d} loss={loss.item():.4f} acc={acc:.3f}")

    # 5) Save artifacts
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "labels.json").write_text(json.dumps(
        {"labels": labels, "label_to_id": label_to_id, "id_to_label": id_to_label},
        indent=2
    ))

    # Include feature spec for future compatibility
    (out_dir / "feature_spec.json").write_text(json.dumps(
        {"feature_dim": int(X.shape[1]), "features": ["area_mean","area_std","area_min","area_max","area_mad","diff_mean","diff_max"]},
        indent=2
    ))

    print(f"\nSaved model to: {out_dir / 'model.pt'}")
    print(f"Saved labels to: {out_dir / 'labels.json'}")
    print(f"Saved feature spec to: {out_dir / 'feature_spec.json'}")


if __name__ == "__main__":
    main()