"""A deliberately simple IoU tracker: greedy association, optional linear motion prediction, coasting.

Per frame: predict each live track's box (last matched box + smoothed per-frame velocity * frames since
the last match), match detections to tracks greedily by descending IoU (>= ``iou_thr``), start a new
track for every unmatched detection, and keep an unmatched track alive for up to ``max_age`` frames.
While a track is unmatched it can still be *output* at its predicted position (``coast_output``), which
is what bridges short detection gaps. Boxes are ``(x1, y1, x2, y2)``. No appearance model, no Kalman
filter; identities are therefore image-plane track ids, not player identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .metrics import iou_matrix


def greedy_match(iou: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    """One-to-one pairs ``(row, col)`` taken in descending IoU order while IoU >= ``thr``."""
    iou = np.asarray(iou, dtype=np.float64)
    if iou.size == 0:
        return []
    pairs: List[Tuple[int, int]] = []
    used_r, used_c = set(), set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        r, c = np.unravel_index(flat, iou.shape)
        if iou[r, c] < thr:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return pairs


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    vel: np.ndarray
    score: float
    hits: int = 1
    missed: int = 0


class IouTracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 5, min_hits: int = 1, use_velocity: bool = True,
                 vel_alpha: float = 0.5, coast_output: bool = True):
        if not 0.0 < iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in (0, 1], got {iou_thr}")
        if max_age < 0 or min_hits < 1:
            raise ValueError("max_age must be >= 0 and min_hits >= 1")
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.use_velocity, self.vel_alpha, self.coast_output = use_velocity, vel_alpha, coast_output
        self.tracks: List[_Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: Optional[np.ndarray] = None):
        """Process one frame. Returns ``(ids int64 (K,), boxes float32 (K, 4), scores float32 (K,), coasted bool (K,))``."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.ones(len(boxes)) if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)
        predicted = np.array([t.box + t.vel * (t.missed + 1) for t in self.tracks]).reshape(-1, 4)
        pairs = greedy_match(iou_matrix(predicted, boxes), self.iou_thr) if len(self.tracks) and len(boxes) else []
        matched_t = {ti for ti, _ in pairs}
        matched_d = {di for _, di in pairs}
        for ti, di in pairs:
            t = self.tracks[ti]
            if self.use_velocity:
                t.vel = self.vel_alpha * (boxes[di] - t.box) / (t.missed + 1) + (1 - self.vel_alpha) * t.vel
            t.box, t.score, t.hits, t.missed = boxes[di].copy(), float(scores[di]), t.hits + 1, 0
        out_ids, out_boxes, out_scores, out_coasted = [], [], [], []
        for ti, t in enumerate(self.tracks):
            if ti in matched_t:
                if t.hits >= self.min_hits:
                    out_ids.append(t.track_id); out_boxes.append(t.box); out_scores.append(t.score); out_coasted.append(False)
            else:
                t.missed += 1
                if self.coast_output and t.missed <= self.max_age and t.hits >= self.min_hits:
                    out_ids.append(t.track_id); out_boxes.append(t.box + t.vel * t.missed)
                    out_scores.append(t.score); out_coasted.append(True)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        for di in range(len(boxes)):
            if di not in matched_d:
                self.tracks.append(_Track(self._next_id, boxes[di].copy(), np.zeros(4), float(scores[di])))
                if self.min_hits <= 1:
                    out_ids.append(self._next_id); out_boxes.append(boxes[di]); out_scores.append(float(scores[di])); out_coasted.append(False)
                self._next_id += 1
        return (np.asarray(out_ids, dtype=np.int64), np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4),
                np.asarray(out_scores, dtype=np.float32), np.asarray(out_coasted, dtype=bool))
