# conceptops/core/events.py
"""
Event detection modules for ConceptOps.

We intentionally keep these detectors simple and swappable:
- SimpleEventDetector: fixed-window baseline
- MotionEventDetector: motion-based baseline (mask-aware option)
- ModelEventDetector: trained artifact-backed inference (Phase 3)

Phase 3 quality requirement:
- ModelEventDetector must be "demo-clean" when requested (reduce FP spam).

We now treat:
- inference_profile="demo_clean" as an alias for "demo_clean_v2"
so there is ONE canonical demo mode going into Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------
# Simple fixed-window baseline
# -----------------------------

@dataclass
class SimpleEventConfig:
    frames_per_event: int = 16
    base_label: str = "segment"


class SimpleEventDetector:
    def __init__(self, config: SimpleEventConfig):
        self.config = config

    def detect(self, frame_records) -> list:
        """
        Very simple baseline: chunk frames into equal segments.
        """
        from conceptops.types import EventRecord  # local import to avoid circulars

        n = len(frame_records)
        events: List[EventRecord] = []
        if n == 0:
            return events

        event_id = 0
        step = max(1, int(self.config.frames_per_event))
        for start in range(0, n, step):
            end = min(n, start + step)
            events.append(
                EventRecord(
                    event_id=event_id,
                    label=self.config.base_label,
                    start_frame=start,
                    end_frame=end,
                    score=1.0,
                    metadata={
                        "source": "heuristic",
                        "proposal_method": "fixed_window",
                        "frames_per_event": self.config.frames_per_event,
                    },
                )
            )
            event_id += 1
        return events


# -----------------------------
# Motion baseline
# -----------------------------

@dataclass
class MotionEventConfig:
    threshold_multiplier: float = 2.0
    min_event_length: int = 3
    base_label: str = "move"
    min_area_ratio: float = 0.0  # used for motion_mask mode


class MotionEventDetector:
    def __init__(self, config: MotionEventConfig):
        self.config = config

    def detect(self, frame_records) -> list:
        """
        Motion baseline based on per-frame mask area ratio series.
        """
        from conceptops.types import EventRecord  # local import to avoid circulars

        if not frame_records:
            return []

        def _area_ratio(fr) -> float:
            mq = (fr.metadata or {}).get("mask_quality", {})
            if "area_ratio" in mq:
                return float(mq["area_ratio"])
            if "mean_area_ratio" in mq:
                return float(mq["mean_area_ratio"])
            return 0.0

        series = np.array([_area_ratio(fr) for fr in frame_records], dtype=np.float32)

        if self.config.min_area_ratio > 0:
            series = np.where(series >= float(self.config.min_area_ratio), series, 0.0)

        diffs = np.abs(np.diff(series, prepend=series[0]))

        med = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - med))) + 1e-6
        thresh = med + float(self.config.threshold_multiplier) * mad

        events: List[EventRecord] = []
        in_evt = False
        start = 0
        event_id = 0

        for i, d in enumerate(diffs):
            if (not in_evt) and d >= thresh:
                in_evt = True
                start = i
            elif in_evt and d < thresh:
                end = i
                if (end - start) >= int(self.config.min_event_length):
                    events.append(
                        EventRecord(
                            event_id=event_id,
                            label=self.config.base_label,
                            start_frame=int(start),
                            end_frame=int(end),
                            score=float(min(1.0, (np.max(diffs[start:end]) / (thresh + 1e-9)))),
                            metadata={
                                "source": "heuristic",
                                "proposal_method": "motion",
                                "threshold_multiplier": self.config.threshold_multiplier,
                                "min_event_length": self.config.min_event_length,
                                "min_area_ratio": self.config.min_area_ratio,
                            },
                        )
                    )
                    event_id += 1
                in_evt = False

        if in_evt:
            end = len(diffs)
            if (end - start) >= int(self.config.min_event_length):
                events.append(
                    EventRecord(
                        event_id=event_id,
                        label=self.config.base_label,
                        start_frame=int(start),
                        end_frame=int(end),
                        score=float(min(1.0, (np.max(diffs[start:end]) / (thresh + 1e-9)))),
                        metadata={
                            "source": "heuristic",
                            "proposal_method": "motion",
                            "threshold_multiplier": self.config.threshold_multiplier,
                            "min_event_length": self.config.min_event_length,
                            "min_area_ratio": self.config.min_area_ratio,
                        },
                    )
                )

        return events


# -----------------------------
# Model-backed detector (Phase 3)
# -----------------------------

@dataclass
class InferenceProfile:
    """
    Centralized inference knobs for output hygiene.
    """
    proposal_topk: int = 10
    topk: int = 10
    min_score: float = 0.55
    nms_iou: float = 0.35
    max_events: int = 12
    enforce_nonoverlap: bool = False


def get_inference_profile(name: str) -> InferenceProfile:
    """
    Profiles:

    - default: permissive (research/debug). May output many spans.
    - demo_clean / demo_clean_v2: canonical demo mode.
      Clean outputs, non-overlapping spans, calibrated to avoid empty outputs.
    """
    n = (name or "default").lower().strip()

    # ---- FINAL chosen demo params ----
    if n in ("demo", "demo_clean", "clean", "demo_clean_v2", "demo2", "clean2"):
        return InferenceProfile(
            proposal_topk=20,
            topk=20,
            min_score=0.30,
            nms_iou=0.30,
            max_events=10,
            enforce_nonoverlap=True,
        )

    # default profile: permissive + low min_score so it doesn't go empty
    return InferenceProfile(
        proposal_topk=15,
        topk=25,
        min_score=0.20,
        nms_iou=0.50,
        max_events=50,
        enforce_nonoverlap=False,
    )


@dataclass
class ModelEventConfig:
    model_dir: str
    window_size: int = 8
    stride: int = 4
    topk: int = 5
    min_score: float = 0.55
    nms_iou: float = 0.5
    inference_profile: str = "default"


class ModelEventDetector:
    def __init__(
        self,
        model_dir: str,
        window_size: int = 8,
        stride: int = 4,
        topk: int = 5,
        min_score: float = 0.55,
        nms_iou: float = 0.5,
        inference_profile: str = "default",
    ):
        self.model_dir = str(model_dir)
        self.window_size = int(window_size)
        self.stride = int(stride)

        # base knobs
        self.topk = int(topk)
        self.min_score = float(min_score)
        self.nms_iou = float(nms_iou)

        self.inference_profile = inference_profile
        self.profile = get_inference_profile(inference_profile)

        # demo-ish profiles fully override knobs
        if (inference_profile or "").lower().strip() in ("demo", "demo_clean", "clean", "demo_clean_v2", "demo2", "clean2"):
            self.topk = int(self.profile.topk)
            self.min_score = float(self.profile.min_score)
            self.nms_iou = float(self.profile.nms_iou)

        self.max_events = int(self.profile.max_events)

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_dir = Path(self.model_dir)
        model_path = model_dir / "model.pt"
        labels_path = model_dir / "labels.json"

        if not model_path.exists():
            raise FileNotFoundError(f"model.pt not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found: {labels_path}")

        self._labels = json_load(labels_path)
        if isinstance(self._labels, dict) and "labels" in self._labels:
            self._labels = self._labels["labels"]
        if not isinstance(self._labels, list) or not self._labels:
            raise ValueError(f"labels.json must be a non-empty list. Got: {type(self._labels)}")

        from conceptops.training.model import TinyEventMLP

        self._model = TinyEventMLP(in_dim=7, num_classes=len(self._labels))
        state = torch.load(str(model_path), map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    def _temporal_iou(self, a_s: int, a_e: int, b_s: int, b_e: int) -> float:
        inter = max(0, min(a_e, b_e) - max(a_s, b_s))
        union = max(1e-9, (a_e - a_s) + (b_e - b_s) - inter)
        return float(inter / union)

    def _overlaps_strict(self, a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
        # touching endpoints is allowed
        return (a_s < b_e) and (b_s < a_e)

    def _enforce_nonoverlap(
        self, ranked: List[Tuple[int, float, int, int, int]]
    ) -> List[Tuple[int, float, int, int, int]]:
        kept: List[Tuple[int, float, int, int, int]] = []
        for cand in ranked:
            _, _, _, s, e = cand
            ok = True
            for k in kept:
                _, _, _, ks, ke = k
                if self._overlaps_strict(int(s), int(e), int(ks), int(ke)):
                    ok = False
                    break
            if ok:
                kept.append(cand)
            if len(kept) >= int(self.max_events):
                break
        return kept

    def _nms(self, ranked: List[Tuple[int, float, int, int, int]]) -> List[Tuple[int, float, int, int, int]]:
        ranked = [r for r in ranked if float(r[1]) >= float(self.min_score)]
        if not ranked:
            return []

        kept: List[Tuple[int, float, int, int, int]] = []
        for cand in ranked:
            _, _, _, s, e = cand
            suppress = False
            for k in kept:
                _, _, _, ks, ke = k
                iou = self._temporal_iou(int(s), int(e), int(ks), int(ke))
                if iou >= float(self.nms_iou):
                    suppress = True
                    break
            if not suppress:
                kept.append(cand)
            if len(kept) >= int(self.max_events):
                break

        if bool(self.profile.enforce_nonoverlap):
            kept = self._enforce_nonoverlap(kept)

        return kept

    def detect(self, frame_records, episode_dir: str) -> list:
        from conceptops.types import EventRecord
        from conceptops.training.features import extract_window_features
        from conceptops.training.proposals import motion_guided_proposals_from_area

        ep_dir = Path(episode_dir)
        if not frame_records:
            return []

        def _area_ratio(fr) -> float:
            mq = (fr.metadata or {}).get("mask_quality", {})
            if "area_ratio" in mq:
                return float(mq["area_ratio"])
            if "mean_area_ratio" in mq:
                return float(mq["mean_area_ratio"])
            return 0.0

        area_series = [_area_ratio(fr) for fr in frame_records]

        class _Ep:
            def __init__(self, frames):
                self.frames = frames

        episode_like = _Ep(frame_records)

        proposals = motion_guided_proposals_from_area(
            area_series=area_series,
            window_size=self.window_size,
            stride=self.stride,
            topk=int(self.profile.proposal_topk),
        )
        if not proposals:
            return []

        X: List[np.ndarray] = []
        spans: List[Tuple[int, int]] = []
        for sp in proposals:
            feats = extract_window_features(
                episode=episode_like,
                episode_dir=ep_dir,
                start_frame=sp.start_frame,
                end_frame=sp.end_frame,
            )
            X.append(feats)
            spans.append((sp.start_frame, sp.end_frame))

        X_t = torch.from_numpy(np.stack(X).astype(np.float32))

        with torch.no_grad():
            logits = self._model(X_t)
            probs = F.softmax(logits, dim=1).numpy()
            cls_ids = np.argmax(probs, axis=1)
            confs = probs[np.arange(len(cls_ids)), cls_ids]

        ranked = sorted(
            [(i, float(confs[i]), int(cls_ids[i]), spans[i][0], spans[i][1]) for i in range(len(spans))],
            key=lambda x: x[1],
            reverse=True,
        )

        ranked = ranked[: int(self.topk)]
        top = self._nms(ranked)

        events: List[EventRecord] = []
        for event_id, (proposal_index, score, cls_id, s, e) in enumerate(top):
            events.append(
                EventRecord(
                    event_id=event_id,
                    label=self._labels[cls_id],
                    start_frame=int(s),
                    end_frame=int(e),
                    score=float(score),
                    metadata={
                        "source": "model",
                        "proposal_method": "motion_guided_from_area",
                        "window_size": self.window_size,
                        "stride": self.stride,
                        "proposal_index": int(proposal_index),
                        "model_dir": str(self.model_dir),
                        "inference_profile": str(self.inference_profile),
                        "profile": {
                            "proposal_topk": int(self.profile.proposal_topk),
                            "topk": int(self.topk),
                            "min_score": float(self.min_score),
                            "nms_iou": float(self.nms_iou),
                            "max_events": int(self.max_events),
                            "enforce_nonoverlap": bool(self.profile.enforce_nonoverlap),
                        },
                    },
                )
            )

        return events


# -----------------------------
# Small JSON helper
# -----------------------------

import json


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)