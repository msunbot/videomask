from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def temporal_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """
    Temporal IoU for inclusive frame spans.

    IoU = intersection / union
    where intersection = overlap length
          union = total covered length

    Inclusive span length = end - start + 1
    """
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_end < inter_start:
        return 0.0

    inter = inter_end - inter_start + 1
    a_len = a_end - a_start + 1
    b_len = b_end - b_start + 1
    union = a_len + b_len - inter
    return float(inter) / float(union)


@dataclass(frozen=True)
class MatchResult:
    """
    One-to-one matching result between predictions and ground truth.
    """
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def match_spans_one_to_one(
    preds: List[Tuple[int, int, str, float]],
    gts: List[Tuple[int, int, str]],
    iou_thresh: float = 0.3,
    require_label_match: bool = True,
) -> MatchResult:
    """
    Greedy one-to-one matching.

    preds: [(start, end, label, score), ...]
    gts:   [(start, end, label), ...]

    Matching rules:
    - A pred can match at most 1 gt, and vice versa.
    - Match requires IoU >= iou_thresh.
    - If require_label_match=True, labels must match too.

    This is a simple, explainable baseline evaluator.
    """
    used_gt = set()

    # Sort predictions by score (high confidence matched first)
    preds_sorted = sorted(preds, key=lambda x: x[3], reverse=True)

    tp = 0
    fp = 0

    for (ps, pe, pl, pscore) in preds_sorted:
        best = None  # (gt_idx, iou)
        for gi, (gs, ge, gl) in enumerate(gts):
            if gi in used_gt:
                continue

            if require_label_match and (pl != gl):
                continue

            iou = temporal_iou(ps, pe, gs, ge)
            if iou >= iou_thresh:
                if best is None or iou > best[1]:
                    best = (gi, iou)

        if best is not None:
            tp += 1
            used_gt.add(best[0])
        else:
            fp += 1

    fn = len(gts) - len(used_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return MatchResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)