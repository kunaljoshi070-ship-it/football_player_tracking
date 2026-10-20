"""Single-class detection metrics in plain NumPy (no extra dependencies).

Boxes are ``(x1, y1, x2, y2)``. Matching is greedy by descending score; each prediction takes the
best still-unmatched ground-truth box with IoU >= threshold (COCO-style). AP is the area under the
precision envelope using all recall change points (VOC2010-style all-point interpolation), so it
is close to, but not identical to, the COCO 101-point value.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape ``(len(a), len(b))``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_image(gt: np.ndarray, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """True-positive flag for each prediction (in input order) of one image."""
    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    tp = np.zeros(len(boxes), dtype=bool)
    gt = np.asarray(gt).reshape(-1, 4)
    if len(gt) == 0 or len(boxes) == 0:
        return tp
    ious = iou_matrix(boxes, gt)
    taken = np.zeros(len(gt), dtype=bool)
    for p in np.argsort(-scores, kind="stable"):
        candidates = np.where(taken, -1.0, ious[p])
        g = int(np.argmax(candidates))
        if candidates[g] >= iou_thr:
            tp[p], taken[g] = True, True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP over a set of predictions; ``nan`` if there is no ground truth."""
    if n_gt == 0:
        return float("nan")
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp_sorted = np.asarray(tp, dtype=np.float64)[order]
    ctp, cfp = np.cumsum(tp_sorted), np.cumsum(1.0 - tp_sorted)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class DetectionEvaluator:
    """Accumulate per-image predictions and report AP at several IoU thresholds plus P/R at a score."""

    def __init__(self, iou_thresholds: Sequence[float] = (0.5, 0.75)):
        self.iou_thresholds = tuple(iou_thresholds)
        self._scores: List[np.ndarray] = []
        self._tp: Dict[float, List[np.ndarray]] = {t: [] for t in self.iou_thresholds}
        self.n_gt = 0
        self.n_images = 0

    def add_image(self, gt_xyxy: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray) -> None:
        gt = np.asarray(gt_xyxy).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        self.n_gt += len(gt)
        self.n_images += 1
        self._scores.append(scores)
        for t in self.iou_thresholds:
            self._tp[t].append(match_image(gt, boxes_xyxy, scores, t))

    def summary(self, report_score: float = 0.5) -> dict:
        scores = np.concatenate(self._scores) if self._scores else np.zeros(0)
        out: dict = {"images": self.n_images, "n_gt": self.n_gt, "n_pred": int(len(scores))}
        for t in self.iou_thresholds:
            tp = np.concatenate(self._tp[t]) if self._tp[t] else np.zeros(0, bool)
            out[f"ap{int(round(t * 100))}"] = average_precision(scores, tp, self.n_gt)
        if self.iou_thresholds and 0.5 in self.iou_thresholds:
            tp = np.concatenate(self._tp[0.5]) if self._tp[0.5] else np.zeros(0, bool)
            sel = scores >= report_score
            n_tp, n_fp = int(np.sum(tp & sel)), int(np.sum(~tp & sel))
            precision = n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan")
            recall = n_tp / self.n_gt if self.n_gt else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if n_tp else 0.0
            out.update({"report_score": report_score, "iou_for_pr": 0.5, "tp": n_tp, "fp": n_fp,
                        "fn": self.n_gt - n_tp, "precision": precision, "recall": recall, "f1": f1})
        return out
