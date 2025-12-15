from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from conceptops.types import Episode
from conceptops.labeling.io import load_labeled_event_records

def _events_to_frame_labels(events, num_frames: int, background: str = "__none__") -> list[str]:
    """
    Convert possibly-overlapping EventRecords into a per-frame label sequence.

    Rule (dominant label per frame):
    - Initialize all frames to background.
    - Iterate events in the given order; later events overwrite earlier ones on overlapping frames.
      (Simple, deterministic, and easy to reason about.)
    """
    labels = [background] * num_frames
    for ev in events:
        s = max(0, int(ev.start_frame))
        e = min(num_frames - 1, int(ev.end_frame))
        for i in range(s, e + 1):
            labels[i] = ev.label
    return labels


def _frame_labels_to_spans(frame_labels: list[str], background: str = "__none__") -> list[tuple[int, int, str]]:
    """
    Compress per-frame labels into non-overlapping spans.

    Example: [none, pick, pick, move, move] -> [(1,2,"pick"), (3,4,"move")]
    """
    spans = []
    if not frame_labels:
        return spans

    cur_label = frame_labels[0]
    cur_start = 0

    for i in range(1, len(frame_labels)):
        if frame_labels[i] != cur_label:
            if cur_label != background:
                spans.append((cur_start, i - 1, cur_label))
            cur_label = frame_labels[i]
            cur_start = i

    # tail
    if cur_label != background:
        spans.append((cur_start, len(frame_labels) - 1, cur_label))

    return spans


def normalize_events_to_nonoverlapping_spans(episode, events) -> list[tuple[int, int, str]]:
    """
    Public helper: take Episode + EventRecord list -> non-overlapping labeled spans.

    This enforces:
    - one label per frame (dominant label per frame)
    - spans become disjoint (no overlap)
    """
    num_frames = len(episode.frames)
    frame_labels = _events_to_frame_labels(events, num_frames=num_frames, background="__none__")
    spans = _frame_labels_to_spans(frame_labels, background="__none__")
    return spans


@dataclass(frozen=True)
class LabeledEpisode:
    """
    Bundles an Episode with its labeled EventRecords (can be empty if unlabeled).
    """
    episode_dir: Path
    episode: Episode
    events: List  # List[EventRecord] but keep loose typing if your codebase uses attrs/dataclasses


def iter_episode_dirs(episodes_root: Path) -> List[Path]:
    """
    Returns episode directories under data/episodes/*.
    """
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root not found: {episodes_root}")

    dirs = [p for p in episodes_root.iterdir() if p.is_dir()]
    return sorted(dirs)


def load_labeled_episode(
    episode_dir: Path,
    taxonomy_path: Path,
) -> LabeledEpisode:
    """
    Load episode.json and its optional event_labels.json, returning canonical EventRecord list.
    """
    episode_path = episode_dir / "episode.json"
    if not episode_path.exists():
        raise FileNotFoundError(f"episode.json not found in {episode_dir}")

    # >>> MODIFIED (Phase 3: robust episode.json loading + skip broken files)
    episode_path = episode_dir / "episode.json"
    if not episode_path.exists():
        raise FileNotFoundError(f"episode.json not found in {episode_dir}")

    # IMPORTANT:
    # In this codebase, Episode.from_json(...) expects a JSON STRING, not a filepath.
    # So we must read the file contents ourselves (or use Episode.from_json_file if it exists).
    try:
        json_text = episode_path.read_text(encoding="utf-8").strip()
        if not json_text:
            # Empty file -> skip upstream artifact without crashing training
            raise ValueError("episode.json is empty")

        episode = Episode.from_json(json_text)

    except Exception as e:
        # If one episode is malformed, we don't want to kill the whole training run.
        # Phase 3 data is often messy; skipping is the pragmatic choice.
        raise ValueError(f"Failed to load episode.json at {episode_path}: {e}") from e
    # <<< END MODIFIED

    # Number of frames = number of FrameRecord objects (canonical).
    num_frames = len(episode.frames)

    # Uses your label IO bridge; returns [] if unlabeled.
    events = load_labeled_event_records(
        episode_dir=episode_dir,
        taxonomy_path=taxonomy_path,
        num_frames=num_frames,
    )

    return LabeledEpisode(episode_dir=episode_dir, episode=episode, events=events)


def iter_labeled_episodes(
    episodes_root: Path,
    taxonomy_path: Path,
    require_labels: bool = True,
) -> Iterator[LabeledEpisode]:
    """
    Iterate through episode folders and yield LabeledEpisode.

    require_labels=True means we skip unlabeled episodes (default for training).
    """
    for ep_dir in iter_episode_dirs(episodes_root):
        try:
            le = load_labeled_episode(ep_dir, taxonomy_path)
        except Exception as e:
            # Print a readable warning and skip.
            print(f"[WARN] Skipping episode at {ep_dir} due to load error: {e}")
            continue
        if require_labels and len(le.events) == 0:
            continue
        yield le


def build_label_set(labeled_episodes: List[LabeledEpisode]) -> List[str]:
    """
    Collect the set of labels present in the dataset (only from labeled spans).
    """
    labels = set()
    for le in labeled_episodes:
        for ev in le.events:
            labels.add(ev.label)
    return sorted(labels)