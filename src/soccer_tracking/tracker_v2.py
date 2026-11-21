"""Tracker v2 (Phase II, WS-C1): a ByteTrack-style two-threshold IoU tracker. Phase I's ``IouTracker`` is untouched.

Differences from ``IouTracker`` (all intended to cut the false positives that coasting created):

* Two-stage association. Stage 1 matches every live track (predicted box) to the *high-score* detections
  (score >= ``det_high``). Stage 2 matches the tracks still unmatched, but only *confirmed* ones, to the
  *low-score* detections (``det_low`` <= score < ``det_high``). Low-score boxes can therefore only continue a
  confirmed track; they never start one.
* Confirmation. A new track (from an unmatched high-score box) is tentative; it is output only once it has
  ``min_hits`` consecutive matches. A tentative track that misses a frame is deleted.
* No coasting output. A confirmed track that goes unmatched is kept alive internally for up to ``max_age``
  frames (so it can be re-matched) but nothing is output for it while it is unmatched.
* Motion model as in Phase I: last box + smoothed per-frame velocity * frames since the last match.

Input to ``update`` is every detection above the score floor (not pre-filtered at 0.5); the tracker applies
``det_low`` / ``det_high`` itself. Return value matches ``IouTracker.update``:
``(ids int64 (K,), boxes float32 (K, 4), scores float32 (K,), coasted bool (K,))`` (``coasted`` is always False).
Matched low-score boxes are output (as in ByteTrack), which is why ``det_low`` is part of the grid.
No appearance model: ids are image-plane track ids, not player identities.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence

import numpy as np

from .metrics import iou_matrix
from .tracker import greedy_match

# Pre-registered grid (docs/phase2_amendment_tracker_v2.md): 2 x 2 x 2 x 3 x 2 = 48 settings.
GRID_V2: Dict[str, tuple] = {"det_high": (0.5, 0.6), "det_low": (0.1, 0.3), "match_iou": (0.3, 0.4),
                             "max_age": (5, 15, 30), "min_hits": (2, 3)}


def grid_v2() -> Iterator[dict]:
    keys = list(GRID_V2)
    for values in itertools.product(*GRID_V2.values()):
        yield dict(zip(keys, values))


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    vel: np.ndarray
    score: float
    hits: int = 1
    missed: int = 0


class ByteIouTracker:
    def __init__(self, det_high: float = 0.6, det_low: float = 0.1, match_iou: float = 0.4, max_age: int = 15,
                 min_hits: int = 2, use_velocity: bool = True, vel_alpha: float = 0.5):
        if not 0.0 < det_low < det_high <= 1.0:
            raise ValueError(f"need 0 < det_low < det_high <= 1, got det_low={det_low}, det_high={det_high}")
        if not 0.0 < match_iou <= 1.0:
            raise ValueError(f"match_iou must be in (0, 1], got {match_iou}")
        if max_age < 0 or min_hits < 1:
            raise ValueError("max_age must be >= 0 and min_hits >= 1")
        self.det_high, self.det_low, self.match_iou = det_high, det_low, match_iou
        self.max_age, self.min_hits = max_age, min_hits
        self.use_velocity, self.vel_alpha = use_velocity, vel_alpha
        self.tracks: List[_Track] = []
        self._next_id = 1

    def _confirmed(self, t: _Track) -> bool:
        return t.hits >= self.min_hits

    def update(self, boxes: np.ndarray, scores: np.ndarray):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        keep = scores >= self.det_low
        boxes, scores = boxes[keep], scores[keep]
        high = np.flatnonzero(scores >= self.det_high)
        low = np.flatnonzero(scores < self.det_high)
        n = len(self.tracks)
        pred = np.array([t.box + t.vel * (t.missed + 1) for t in self.tracks]).reshape(-1, 4)
        confirmed_before = [self._confirmed(t) for t in self.tracks]

        matched: Dict[int, int] = {}                                  # track index -> detection index

        def stage(track_idx: Sequence[int], det_idx: np.ndarray) -> None:
            track_idx = np.asarray(track_idx, dtype=int)
            if len(track_idx) == 0 or len(det_idx) == 0:
                return
            for r, c in greedy_match(iou_matrix(pred[track_idx], boxes[det_idx]), self.match_iou):
                matched[int(track_idx[r])] = int(det_idx[c])

        stage(range(n), high)
        stage([i for i in range(n) if i not in matched and confirmed_before[i]], low)

        out_ids, out_boxes, out_scores = [], [], []
        survivors: List[_Track] = []
        for ti, t in enumerate(self.tracks):
            if ti in matched:
                di = matched[ti]
                if self.use_velocity:
                    t.vel = self.vel_alpha * (boxes[di] - t.box) / (t.missed + 1) + (1 - self.vel_alpha) * t.vel
                t.box, t.score, t.hits, t.missed = boxes[di].copy(), float(scores[di]), t.hits + 1, 0
                if self._confirmed(t):
                    out_ids.append(t.track_id); out_boxes.append(t.box); out_scores.append(t.score)
                survivors.append(t)
            elif confirmed_before[ti]:
                t.missed += 1
                if t.missed <= self.max_age:
                    survivors.append(t)                               # kept alive, not output
            # tentative and unmatched: dropped
        used = set(matched.values())
        for di in high:
            if int(di) in used:
                continue
            t = _Track(self._next_id, boxes[di].copy(), np.zeros(4), float(scores[di]))
            self._next_id += 1
            survivors.append(t)
            if self._confirmed(t):
                out_ids.append(t.track_id); out_boxes.append(t.box); out_scores.append(t.score)
        self.tracks = survivors
        return (np.asarray(out_ids, dtype=np.int64), np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4),
                np.asarray(out_scores, dtype=np.float32), np.zeros(len(out_ids), dtype=bool))


def build_arm_v2(dets, params: dict):
    """Per-frame ``(ids, boxes)`` and score arrays from cached detections (all scores above the floor) with tracker v2."""
    tracker = ByteIouTracker(**params)
    preds, scores = [], []
    for b, s in dets:
        ids, boxes, sc, _ = tracker.update(np.asarray(b), np.asarray(s))
        preds.append((ids, boxes))
        scores.append(sc)
    return preds, scores


def rank_candidates(rows: Sequence[dict]) -> List[dict]:
    """Stage-1 ranking on NumPy-reference mean HOTA (rows: ``{'params', 'hota', 'idsw'}``); ties: lower IDSW, smaller max_age."""
    return sorted(rows, key=lambda r: (-r["hota"], r["idsw"], r["params"]["max_age"]))


def pick_final(rows: Sequence[dict], tol: float = 0.005) -> dict:
    """Stage-2 rule on TrackEval mean HOTA: best HOTA; among settings within ``tol`` of it, lower IDSW, then smaller max_age."""
    best = max(r["hota"] for r in rows)
    close = [r for r in rows if r["hota"] >= best - tol]
    return min(close, key=lambda r: (r["idsw"], r["params"]["max_age"]))
