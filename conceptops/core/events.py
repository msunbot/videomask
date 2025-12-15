from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F 

from conceptops.types import FrameRecord, EventRecord
from conceptops.training.model import TinyEventMLP
from conceptops.training.features import extract_window_features
from conceptops.training.proposals import sliding_window_proposals

from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


DEFAULT_IOU_THRESHOLD = 0.90
MIN_EVENT_LENGTH_FRAMES = 2
CENTROID_MOVE_THRESHOLD = 0.08  # ~8% of frame width/height
AREA_CHANGE_THRESHOLD = 0.10    # 10% relative area change

@dataclass
class ModelEventConfig:
    """
    Configuration for a model-based event detector.

    This is the "serious ML" interface. For now it's a scaffold that
    can be backed by a stub or by a real trained model later.
    """
    model_name: str = "stub_v1"
    score_threshold: float = 0.0
    base_label: str = "model_event"
    # Extra fields (e.g., path to weights, device) can be added later.

@dataclass
class ModelEventDetector:
    """
    Model-backed event detector.

    Phase 3 baseline:
    - Proposals: sliding windows over frames
    - Features: handcrafted (mask area stats + frame diff)
    - Model: Tiny MLP classifier trained on labeled spans
    """

    def __init__(
            self,
            model_dir: str,
            window_size: int = 8,
            stride: int = 4,
            topk: int = 5,
            min_score: float = 0.55,
            nms_iou: float = 0.5,
        ):
        self.model_dir = Path(model_dir)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.topk = int(topk)
        self.min_score = float(min_score)
        self.nms_iou = float(nms_iou)

        # Load artifacts once at init (fast inference).
        self._model, self._labels, self._in_dim = self._load_model_artifacts(self.model_dir)

    def _load_model_artifacts(self, model_dir: Path):
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

        return model, labels, in_dim

    # Phase 3: temporal IoU + NMS
    def _temporal_iou(self, a_s: int, a_e: int, b_s: int, b_e: int) -> float:
        inter_s = max(a_s, b_s)
        inter_e = min(a_e, b_e)
        if inter_e < inter_s:
            return 0.0
        inter = inter_e - inter_s + 1
        a_len = a_e - a_s + 1
        b_len = b_e - b_s + 1
        union = a_len + b_len - inter
        return float(inter) / float(union)


    def _nms(self, candidates: list[tuple[int, float, int, int, int]]) -> list[tuple[int, float, int, int, int]]:
        """
        candidates: [(proposal_index, score, cls_id, start, end), ...] sorted desc by score.

        Keep highest-score spans; suppress spans with IoU >= nms_iou against any kept span.
        """
        kept: list[tuple[int, float, int, int, int]] = []
        for cand in candidates:
            _, score, _, s, e = cand
            if score < self.min_score:
                continue

            too_close = False
            for kept_c in kept:
                _, kept_score, _, ks, ke = kept_c
                if self._temporal_iou(s, e, ks, ke) >= self.nms_iou:
                    too_close = True
                    break

            if not too_close:
                kept.append(cand)

            if len(kept) >= self.topk:
                break

        return kept

    def detect(self, frame_records, episode_dir: str) -> list:
        """
        Return List[EventRecord] for the episode.

        - Accept frame_records (List[FrameRecord]) because Episode doesn't exist yet when process_video_to_dataset runs event detection
        - Still use episode_dir to read frames_raw/ for frame-diff features 
        """
        from conceptops.types import EventRecord  # local import to avoid circulars

        ep_dir = Path(episode_dir)
        num_frames = len(frame_records)

        # >>> Phase 3: build area_series from frame_records
        def _area_ratio(fr) -> float:
            mq = (fr.metadata or {}).get("mask_quality", {})
            if "area_ratio" in mq:
                return float(mq["area_ratio"])
            if "mean_area_ratio" in mq:
                return float(mq["mean_area_ratio"])
            return 0.0

        area_series = [_area_ratio(fr) for fr in frame_records]

        # >>> Phase 3: minimal adapter so features.py can stay unchanged
        class _Ep:
            def __init__(self, frames):
                self.frames = frames

        episode_like = _Ep(frame_records)

        # 1) propose spans (coverage baseline)
        # use motion-guided proposals + coverage
        from conceptops.training.proposals import motion_guided_proposals_from_area

        proposals = motion_guided_proposals_from_area(
            area_series=area_series,
            window_size=self.window_size,
            stride=self.stride,
            topk=10,
        )
        if not proposals:
            return []

        # 2) featurize all proposals
        X = []
        spans = []
        for sp in proposals:
            feats = extract_window_features(
                episode=episode_like,
                episode_dir=ep_dir,
                start_frame=sp.start_frame,
                end_frame=sp.end_frame,
            )
            X.append(feats)
            spans.append((sp.start_frame, sp.end_frame))

        X_np = np.stack(X).astype(np.float32)
        X_t = torch.from_numpy(X_np)

        # 3) score with model
        with torch.no_grad():
            logits = self._model(X_t)
            probs = F.softmax(logits, dim=1).numpy()
            cls_ids = np.argmax(probs, axis=1)
            confs = probs[np.arange(len(cls_ids)), cls_ids]

        # 4) rank spans by confidence, take topk
        ranked = sorted(
            [(i, float(confs[i]), int(cls_ids[i]), spans[i][0], spans[i][1]) for i in range(len(spans))],
            key=lambda x: x[1],
            reverse=True,
        )

        # apply score threshold + NMS dedup 
        top = self._nms(ranked)

        # 5) emit EventRecords
        events = []
        for event_id, (proposal_index, score, cls_id, s, e) in enumerate(top):
            label = self._labels[cls_id]
            events.append(
                EventRecord(
                    event_id=event_id,
                    label=label,
                    start_frame=s,
                    end_frame=e,
                    score=score,
                    metadata={
                        "source": "model",
                        "proposal_method": "sliding_window",
                        "window_size": self.window_size,
                        "stride": self.stride,
                        "proposal_index": proposal_index,
                        "model_dir": str(self.model_dir),
                    },
                )
            )
        # Phase 3: debug counts
        # print(f"[ModelEventDetector] proposals={len(proposals)} ranked={len(ranked)} kept={len(top)} min_score={self.min_score}")
        
        return events

@dataclass
class EventConfig:
    out_dir: Path
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    min_event_length: int = MIN_EVENT_LENGTH_FRAMES

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "EventConfig":
        return cls(
            out_dir=Path(args.out),
            iou_threshold=float(args.iou_threshold),
            min_event_length=int(args.min_event_length),
        )
@dataclass
class SimpleEventConfig:
    """
    Configuration for the simple v0.5 event detector.

    This is deliberately minimal and deterministic:
      - We segment frames into fixed windows of N frames.
      - Each window becomes an EventRecord with a generic label.

    Later, you'll replace this with Ego2Robot's model-based logic,
    but keep the same public interface so the pipeline doesn't break.
    """
    frames_per_event: int = 16   # e.g. ~0.5s at 30fps, ~2s at 8fps
    base_label: str = "segment"  # generic label prefix

@dataclass
class SimpleEventDetector:
    """
    v0.5 "toy" event detector.

    For now it just groups consecutive frames into fixed-length segments.
    This gives you:
      - A concrete EventRecord schema in episode.json.
      - A clean interface where a more advanced Ego2Robot detector
        can be dropped in later.

    Usage:
        detector = SimpleEventDetector(config=SimpleEventConfig(frames_per_event=8))
        events = detector.detect(frame_records)
    """

    def __init__(self, config: Optional[SimpleEventConfig] = None) -> None:
        self.config = config or SimpleEventConfig()

    def detect(self, frames: List[FrameRecord]) -> List[EventRecord]:
        """
        Segment the list of FrameRecord objects into fixed-size windows.

        Args:
            frames: Ordered list of FrameRecord objects (as built in the pipeline).

        Returns:
            List[EventRecord] with monotonically increasing event_id.
        """
        events: List[EventRecord] = []
        n = len(frames)
        if n == 0:
            return events

        k = self.config.frames_per_event
        event_id = 0

        for start in range(0, n, k):
            end = min(start + k - 1, n - 1)  # inclusive end index
            label = f"{self.config.base_label}_{event_id}"

            events.append(
                EventRecord(
                    event_id=event_id,
                    label=label,
                    start_frame=start,
                    end_frame=end,
                    score=None,         # no meaningful score yet
                    metadata={
                        "frames_per_event": k,
                    },
                )
            )
            event_id += 1

        return events

@dataclass
class MotionEventConfig:
    """
    Configuration for the motion-based event detector.

    We compute a simple per-frame motion magnitude (mean absolute pixel
    difference vs previous frame), then segment contiguous high-motion
    regions into events.
    """
    threshold_multiplier: float = 2.0   # how far above median to treat as "active"
    min_event_length: int = 3          # min number of frames per event
    base_label: str = "move"           # label prefix for motion events
    min_area_ratio: float = 0.0         # minimum mask area to consider frame "object-present"

class MotionEventDetector:
    """
    Motion-based event detector v0.5.

    This is still heuristic, but more meaningful than fixed windows:

      - It loads frames and computes per-frame motion magnitude.
      - Frames with unusually high motion are marked as "active".
      - Contiguous active regions become EventRecord segments.

    Later you can replace/augment the motion score with Ego2Robot
    features, but keep the same `detect(frames) -> List[EventRecord]`
    interface.
    """

    def __init__(self, config: Optional[MotionEventConfig] = None) -> None:
        self.config = config or MotionEventConfig()

    def _compute_motion_series(self, frames: List[FrameRecord]) -> List[float]:
        """
        Compute a motion magnitude per frame (starting from index 1).

        motion[i] is the mean absolute grayscale difference between
        frame i and frame i-1. For frame 0, we define motion[0] = 0.0.
        """
        n = len(frames)
        if n == 0:
            return []

        motion = [0.0] * n  # frame 0 has zero by definition

        prev_img = self._load_gray(frames[0].image_path)

        for i in range(1, n):
            curr_img = self._load_gray(frames[i].image_path)
            # mean absolute difference as a scalar motion metric
            diff = np.abs(curr_img.astype(np.float32) - prev_img.astype(np.float32))
            motion[i] = float(diff.mean())
            prev_img = curr_img

        return motion

    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        """
        Load an image from disk and convert to a grayscale numpy array.
        """
        img = Image.open(path).convert("L")  # "L" = 8-bit grayscale
        return np.array(img)

    def detect(self, frames: List[FrameRecord]) -> List[EventRecord]:
        """
        Detect motion-based events from a list of FrameRecord.

        Returns:
            List[EventRecord] with event_id, label, start/end frames, score, metadata.
        """
        events: List[EventRecord] = []
        n = len(frames)
        if n == 0:
            return events

        motion = self._compute_motion_series(frames)

        # Compute a per-frame "max instance area_ratio".
        max_area_per_frame: List[float] = []
        for f in frames:
            if f.instances:
                max_area = max(
                    (inst.area_ratio or 0.0) for inst in f.instances
                )
            else:
                max_area = 0.0
            max_area_per_frame.append(max_area)

        # Compute a data-driven threshold based on the median motion.
        median_motion = float(np.median(motion)) if motion else 0.0
        if median_motion == 0.0:
            threshold = 0.0
        else:
            threshold = median_motion * self.config.threshold_multiplier

        # combine motion & area
        active: List[bool] = []
        for i in range(n): 
            motion_ok = motion[i] >= threshold
            area_ok = max_area_per_frame[i] >= self.config.min_area_ratio
            active.append(motion_ok and area_ok)
            
        # Group contiguous active spans into events.
        event_id = 0
        i = 0
        while i < n:
            if not active[i]:
                i += 1
                continue

            # Found the start of an active run.
            start = i
            while i + 1 < n and active[i + 1]:
                i += 1
            end = i  # inclusive

            length = end - start + 1
            if length >= self.config.min_event_length:
                events.append(
                    EventRecord(
                        event_id=event_id,
                        label=f"{self.config.base_label}_{event_id}",
                        start_frame=start,
                        end_frame=end,
                        score=None,  # could store avg motion here later
                        metadata={
                            "threshold": threshold,
                            "median_motion": median_motion,
                            "min_event_length": self.config.min_event_length,
                        },
                    )
                )
                event_id += 1

            i += 1

        # Fallback: if we saw frames but no events, create one full-span event.
        if not events and n > 0:
            events.append(
                EventRecord(
                    event_id=0,
                    label=f"{self.config.base_label}_0",
                    start_frame=0,
                    end_frame=n - 1,
                    score=None,
                    metadata={
                        "fallback": True,
                        "threshold": threshold,
                        "median_motion": median_motion,
                        "min_event_length": self.config.min_event_length,
                    },
                )
            )

        return events

def _load_manifest(out_dir: Path) -> Dict:
    manifest_path = out_dir / "conceptops_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"conceptops_manifest.json not found under {out_dir}. "
            "Run the mask stage first."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(out_dir: Path, manifest: Dict) -> None:
    manifest_path = out_dir / "conceptops_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _load_metadata(metadata_path: Path) -> Dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mask(mask_path: Path) -> np.ndarray:
    """
    Load a binary mask as a boolean numpy array.
    Assumes VideoMask exports 0/255 grayscale PNGs.
    """
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    return arr > 0  # bool mask

def _mask_stats(mask: np.ndarray):
    """
    Compute normalized centroid (cx, cy) in [0,1] and area fraction in [0,1].
    If mask is empty, return center + zero area.
    """
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return 0.5, 0.5, 0.0
    cx = xs.mean() / float(w)
    cy = ys.mean() / float(h)
    area = len(xs) / float(h * w)
    return float(cx), float(cy), float(area)

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Intersection-over-Union between two boolean masks.
    If union is zero (no foreground in both), treat as IoU = 1.0 (no change).
    """
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)

def _build_events(
    mask_paths: List[Path],
    fps: float,
    iou_threshold: float,
    min_event_length: int,
) -> List[Dict]:
    """
    Robust temporal segmentation:

      - Always returns at least 1 event if there are masks.
      - Starts event 0 at frame 0.
      - For each frame t, compare IoU(mask[t-1], mask[t]).
      - If IoU < threshold OR centroid/area change is large => close current event at t-1 and start new one at t.
      - After loop, close the final event at last frame.
      - Filter out very short events; if all are filtered, fall back to [0..N-1].
    """
    if not mask_paths:
        return []

    # Precompute stats
    masks = [_load_mask(p) for p in mask_paths]
    n = len(masks)
    stats = [_mask_stats(m) for m in masks]  # list of (cx, cy, area)

    raw_events: List[Dict] = []
    current_start = 0
    event_id = 0

    def make_event(start_idx: int, end_idx: int, eid: int) -> Dict:
        length = end_idx - start_idx + 1
        mid_idx = (start_idx + end_idx) // 2
        return {
            "event_id": eid,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "num_frames": length,
            "start_time_sec": start_idx / fps,
            "end_time_sec": (end_idx + 1) / fps,
            "key_frame_index": mid_idx,
            "key_frame_path": str(mask_paths[mid_idx]),
        }

    ious: List[float] = []

    for idx in range(1, n):
        m_prev, m_curr = masks[idx - 1], masks[idx]
        iou = _mask_iou(m_prev, m_curr)
        ious.append(iou)

        cx0, cy0, a0 = stats[idx - 1]
        cx1, cy1, a1 = stats[idx]
        dc = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
        da = abs(a1 - a0) / max(a0, a1, 1e-6)

        significant_motion = (
            iou < iou_threshold
            or dc > CENTROID_MOVE_THRESHOLD
            or da > AREA_CHANGE_THRESHOLD
        )

        if significant_motion:
            raw_events.append(make_event(current_start, idx - 1, event_id))
            event_id += 1
            current_start = idx

    raw_events.append(make_event(current_start, n - 1, event_id))

    print("[ConceptOps][DEBUG] IoUs:", [round(float(x), 3) for x in ious[:10]])

    # Filter by min_event_length
    filtered = [ev for ev in raw_events if ev["num_frames"] >= min_event_length]

    # Fallback: if everything got filtered out, keep a single full-length event
    if not filtered:
        filtered = [make_event(0, n - 1, 0)]

    # Normalize event IDs to 0..N-1 and remove duplicate (start,end) segments
    unique: Dict[Tuple[int, int], Dict] = {}
    for ev in filtered:
        key = (ev["start_frame"], ev["end_frame"])
        if key not in unique:
            unique[key] = ev

    normalized: List[Dict] = []
    for new_id, ev in enumerate(sorted(unique.values(), key=lambda e: e["start_frame"])):
        ev["event_id"] = new_id
        normalized.append(ev)

    return normalized

def run_event_stage(cfg: EventConfig) -> Path:
    """
    Phase 2: temporal event extraction.
    Reads conceptops_manifest.json and metadata.json, writes events.json,
    and updates the manifest.
    """
    out_dir = cfg.out_dir
    manifest = _load_manifest(out_dir)

    metadata_path = Path(manifest["metadata_path"])
    metadata = _load_metadata(metadata_path)

    # Assume VideoMask metadata lists mask paths in order.
    mask_paths = [Path(p) for p in metadata.get("masks", [])]
    if not mask_paths:
        raise ValueError(
            f"No mask paths found in metadata.json at {metadata_path}. "
            "Expected key 'masks'."
        )

    fps = float(manifest.get("fps", manifest.get("config", {}).get("fps", 1.0)))

    print("[ConceptOps] Event stage: computing temporal events")
    print(f"  - masks: {len(mask_paths)} frames")
    print(f"  - fps: {fps}")
    print(f"  - IoU threshold: {cfg.iou_threshold}")
    print(f"  - min_event_length: {cfg.min_event_length}")

    events = _build_events(
        mask_paths=mask_paths,
        fps=fps,
        iou_threshold=cfg.iou_threshold,
        min_event_length=cfg.min_event_length,
    )

    events_path = out_dir / "events.json"
    with events_path.open("w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=2)

    # Update manifest
    manifest.setdefault("stages", {})
    manifest["stages"]["events"] = "completed"
    manifest["events_path"] = str(events_path)
    _save_manifest(out_dir, manifest)

    print(f"[ConceptOps] Wrote events → {events_path}")
    print(f"[ConceptOps] Found {len(events)} events.")
    return events_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptops-events",
        description="ConceptOps temporal event extraction (Phase 2).",
    )
    parser.add_argument(
        "out",
        type=str,
        help="Output directory from the mask stage (where conceptops_manifest.json lives).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for starting a new event (default: 0.90).",
    )
    parser.add_argument(
        "--min-event-length",
        type=int,
        default=MIN_EVENT_LENGTH_FRAMES,
        help="Minimum number of frames per event (default: 2).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = EventConfig.from_args(args)
    run_event_stage(cfg)


if __name__ == "__main__":
    main()
