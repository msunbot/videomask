# scripts/eval_dashboard_summary.py
"""
Batch evaluation summary (dashboard-style).

Compares:
- Label-match: requires same label + temporal IoU >= threshold
- Span-only  : ignores label, only timing IoU >= threshold

Supports optional label collapsing (demo3).

Ground truth:
- event_labels.json (your schema: labeled_events with start_frame_idx/end_frame_idx)

Predictions:
- IMPORTANT: Prefer pred_events.json (written by scripts/batch_predict_events.py)
- Fallback to episode.json if pred_events.json not present

Usage:
  python scripts/eval_dashboard_summary.py --episodes_root data/episodes --iou 0.30 --label_collapse demo3
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_int(d: Dict[str, Any], keys: List[str]) -> Optional[int]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                pass
    return None


def normalize_spans_from_list(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of event dicts into:
      {"start": int, "end": int, "label": str}
    Supports key variants:
      start/end
      start_frame/end_frame
      start_frame_idx/end_frame_idx
    """
    out: List[Dict[str, Any]] = []
    for s in raw:
        start = _get_int(s, ["start_frame", "start_frame_idx", "start"])
        end = _get_int(s, ["end_frame", "end_frame_idx", "end"])
        label = s.get("label") or s.get("name") or "UNKNOWN"
        if start is None or end is None:
            continue
        out.append({"start": start, "end": end, "label": str(label)})
    return out


def normalize_gt(label_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = label_data.get("labeled_events") or label_data.get("spans") or label_data.get("events") or []
    return normalize_spans_from_list(raw)


def load_pred_spans_for_episode(ep_dir: Path) -> List[Dict[str, Any]]:
    """
    Prefer pred_events.json (new, model outputs).
    Fallback to episode.json['events'] (older behavior).
    """
    pred_path = ep_dir / "pred_events.json"
    if pred_path.exists():
        pred_raw = load_json(pred_path)
        # pred_events.json might be list directly or a wrapper dict
        if isinstance(pred_raw, list):
            return normalize_spans_from_list(pred_raw)
        if isinstance(pred_raw, dict):
            raw_list = pred_raw.get("events") or pred_raw.get("pred_events") or []
            if isinstance(raw_list, list):
                return normalize_spans_from_list(raw_list)
        return []

    # Fallback: episode.json
    episode_path = ep_dir / "episode.json"
    if episode_path.exists():
        ep = load_json(episode_path)
        raw_list = ep.get("events") or ep.get("pred_events") or []
        if isinstance(raw_list, list):
            return normalize_spans_from_list(raw_list)
    return []


def temporal_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(1e-9, (a_end - a_start) + (b_end - b_start) - inter)
    return float(inter / union)


def greedy_match(
    gt_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
    *,
    require_label: bool,
    iou_thresh: float,
) -> Tuple[int, int, int]:
    used = set()
    tp = 0

    for g in gt_spans:
        best_j = None
        best_iou = 0.0

        for j, p in enumerate(pred_spans):
            if j in used:
                continue
            if require_label and (p.get("label") != g.get("label")):
                continue

            iou = temporal_iou(g["start"], g["end"], p["start"], p["end"])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None and best_iou >= iou_thresh:
            used.add(best_j)
            tp += 1

    fp = max(0, len(pred_spans) - len(used))
    fn = max(0, len(gt_spans) - tp)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-9, (prec + rec))
    return prec, rec, f1


# ---------- label collapse (same as before) ----------

def get_label_collapse_map(preset: str) -> Optional[Dict[str, str]]:
    p = (preset or "none").lower().strip()
    if p in ("none", "off", "false", "0", ""):
        return None
    if p == "demo3":
        return {
            "open": "toggle",
            "close": "toggle",
            "pick": "manipulate",
            "place": "manipulate",
            "pour": "manipulate",
            "wipe": "manipulate",
            "move": "move",
        }
    raise ValueError(f"Unknown --label_collapse preset: {preset}")


def apply_label_map(spans: List[Dict[str, Any]], label_map: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not label_map:
        return spans
    out = []
    for s in spans:
        lab = s.get("label", "UNKNOWN")
        out.append({"start": s["start"], "end": s["end"], "label": label_map.get(lab, lab)})
    return out


def count_labels(spans: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in spans:
        lab = s.get("label", "UNKNOWN")
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_root", type=str, required=True)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--label_collapse", type=str, default="none")
    args = ap.parse_args()

    root = Path(args.episodes_root)
    label_files = sorted(root.rglob("event_labels.json"))
    if not label_files:
        print(f"No event_labels.json found under: {root}")
        return

    label_map = get_label_collapse_map(args.label_collapse)

    agg = {
        "raw": {"label_match": {"tp": 0, "fp": 0, "fn": 0}, "span_only": {"tp": 0, "fp": 0, "fn": 0}},
        "collapsed": {"label_match": {"tp": 0, "fp": 0, "fn": 0}, "span_only": {"tp": 0, "fp": 0, "fn": 0}},
        "episodes": 0,
        "total_gt": 0,
        "total_pred": 0,
    }

    raw_gt_counts: Dict[str, int] = {}
    raw_pr_counts: Dict[str, int] = {}
    col_gt_counts: Dict[str, int] = {}
    col_pr_counts: Dict[str, int] = {}

    for lf in label_files:
        ep_dir = lf.parent

        gt_data = load_json(lf)
        gt_spans = normalize_gt(gt_data)
        if not gt_spans:
            continue

        pred_spans = load_pred_spans_for_episode(ep_dir)

        agg["episodes"] += 1
        agg["total_gt"] += len(gt_spans)
        agg["total_pred"] += len(pred_spans)

        # Raw metrics
        tp, fp, fn = greedy_match(gt_spans, pred_spans, require_label=True, iou_thresh=args.iou)
        agg["raw"]["label_match"]["tp"] += tp
        agg["raw"]["label_match"]["fp"] += fp
        agg["raw"]["label_match"]["fn"] += fn

        tp, fp, fn = greedy_match(gt_spans, pred_spans, require_label=False, iou_thresh=args.iou)
        agg["raw"]["span_only"]["tp"] += tp
        agg["raw"]["span_only"]["fp"] += fp
        agg["raw"]["span_only"]["fn"] += fn

        for k, v in count_labels(gt_spans).items():
            raw_gt_counts[k] = raw_gt_counts.get(k, 0) + v
        for k, v in count_labels(pred_spans).items():
            raw_pr_counts[k] = raw_pr_counts.get(k, 0) + v

        # Collapsed metrics
        if label_map:
            gt2 = apply_label_map(gt_spans, label_map)
            pr2 = apply_label_map(pred_spans, label_map)

            tp, fp, fn = greedy_match(gt2, pr2, require_label=True, iou_thresh=args.iou)
            agg["collapsed"]["label_match"]["tp"] += tp
            agg["collapsed"]["label_match"]["fp"] += fp
            agg["collapsed"]["label_match"]["fn"] += fn

            tp, fp, fn = greedy_match(gt2, pr2, require_label=False, iou_thresh=args.iou)
            agg["collapsed"]["span_only"]["tp"] += tp
            agg["collapsed"]["span_only"]["fp"] += fp
            agg["collapsed"]["span_only"]["fn"] += fn

            for k, v in count_labels(gt2).items():
                col_gt_counts[k] = col_gt_counts.get(k, 0) + v
            for k, v in count_labels(pr2).items():
                col_pr_counts[k] = col_pr_counts.get(k, 0) + v

    print("\n=== EVAL DASHBOARD SUMMARY ===")
    print(f"Episodes root: {root}")
    print(f"IoU threshold: {args.iou:.2f}")
    print(f"Episodes evaluated: {agg['episodes']}")
    print(f"Total GT spans: {agg['total_gt']} | Total predicted spans: {agg['total_pred']}")

    for key in ["label_match", "span_only"]:
        tp, fp, fn = agg["raw"][key]["tp"], agg["raw"][key]["fp"], agg["raw"][key]["fn"]
        prec, rec, f1 = prf(tp, fp, fn)
        print(f"\n-- RAW {key.upper()} --")
        print(f"TP={tp}  FP={fp}  FN={fn}")
        print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    print("\n-- RAW label distribution (GT vs Pred) --")
    print(f"GT  : {dict(sorted(raw_gt_counts.items(), key=lambda kv: kv[0]))}")
    print(f"Pred: {dict(sorted(raw_pr_counts.items(), key=lambda kv: kv[0]))}")

    if label_map:
        for key in ["label_match", "span_only"]:
            tp, fp, fn = agg["collapsed"][key]["tp"], agg["collapsed"][key]["fp"], agg["collapsed"][key]["fn"]
            prec, rec, f1 = prf(tp, fp, fn)
            print(f"\n-- COLLAPSED {key.upper()} --")
            print(f"TP={tp}  FP={fp}  FN={fn}")
            print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

        print("\n-- COLLAPSED label distribution (GT vs Pred) --")
        print(f"GT  : {dict(sorted(col_gt_counts.items(), key=lambda kv: kv[0]))}")
        print(f"Pred: {dict(sorted(col_pr_counts.items(), key=lambda kv: kv[0]))}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()