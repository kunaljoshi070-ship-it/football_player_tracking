"""Tracker v3 (Phase II, WS-C2/C3/C4): Deep-EIoU online tracking + GTA-Link offline stitching +
Track-Level Kit-Color Official Filter & Soft Roster Prior.

Pure NumPy + SciPy (zero torch dependency at tracking time). Protocol: docs/phase2_amendment_tracker_v3.md.
Phase I ``IouTracker`` and Tracker v2 ``ByteIouTracker`` are untouched.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import iou_matrix

# Pre-registered compact grid (docs/phase2_amendment_tracker_v3.md): 2 x 2 x 2 x 2 = 16 settings.
GRID_V3: Dict[str, tuple] = {
    "det_high": (0.65, 0.75),
    "merge_cos_thresh": (0.25, 0.35),
    "official_lab_thresh": (28.0, 38.0),
    "roster_k": (None, 24),
}

DEFAULT_V3_PARAMS: Dict[str, object] = {
    "det_high": 0.70,
    "det_low": 0.30,
    "expand_ratios": (0.5, 0.7, 0.9),
    "prox_thresh": 0.85,
    "app_weight": 0.50,
    "app_thresh": 0.45,
    "max_age": 45,
    "min_hits": 3,
    "emb_alpha": 0.90,
    "use_gta": True,
    "max_gap_frames": 300,
    "max_speed_px": 12.0,
    "base_dist_px": 60.0,
    "merge_cos_thresh": 0.30,
    "team_color_gate": 35.0,
    "min_track_len": 12,
    "use_official_filter": True,
    "official_lab_thresh": 33.0,
    "gk_margin": 0.14,
    "ref_min_len": 40,
    "roster_k": None,
}


def grid_v3() -> Iterator[dict]:
    keys = list(GRID_V3)
    for values in itertools.product(*GRID_V3.values()):
        p = dict(DEFAULT_V3_PARAMS)
        p.update(dict(zip(keys, values)))
        yield p


def expand_boxes(boxes: np.ndarray, ratio: float) -> np.ndarray:
    """Expand xyxy boxes symmetrically around their center by ``ratio`` (w' = w*(1+ratio), h' = h*(1+ratio))."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0 or ratio <= 0.0:
        return b.copy()
    w = np.maximum(0.0, b[:, 2] - b[:, 0])
    h = np.maximum(0.0, b[:, 3] - b[:, 1])
    dw = 0.5 * ratio * w
    dh = 0.5 * ratio * h
    return np.stack([b[:, 0] - dw, b[:, 1] - dh, b[:, 2] + dw, b[:, 3] + dh], axis=1)


def hungarian_match(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    """Exact minimum-cost one-to-one assignment keeping only pairs with ``cost <= max_cost``."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    large = max(1e6, float(max_cost) + 1e5)
    safe = np.where(np.isfinite(cost) & (cost <= max_cost), cost, large)
    rows, cols = linear_sum_assignment(safe)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-8 else v


@dataclass
class Tracklet:
    track_id: int
    frames: List[int] = field(default_factory=list)
    boxes: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    embs: List[np.ndarray] = field(default_factory=list)
    labs: List[np.ndarray] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    def mean_emb(self) -> Optional[np.ndarray]:
        if not self.embs:
            return None
        m = np.mean(np.stack(self.embs, axis=0), axis=0)
        return _l2_normalize(m.astype(np.float64))

    def median_lab(self) -> Optional[np.ndarray]:
        if not self.labs:
            return None
        return np.median(np.stack(self.labs, axis=0), axis=0).astype(np.float64)

    def median_center(self) -> np.ndarray:
        arr = np.stack(self.boxes, axis=0)
        cx = 0.5 * (arr[:, 0] + arr[:, 2])
        cy = 0.5 * (arr[:, 1] + arr[:, 3])
        return np.array([float(np.median(cx)), float(np.median(cy))], dtype=np.float64)


@dataclass
class _LiveTrack:
    track_id: int
    box: np.ndarray
    score: float
    emb: Optional[np.ndarray] = None
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    history: List[Tuple[int, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]] = field(default_factory=list)


class DeepEIoUTracker:
    """Motion-agnostic online tracker using Iterative Scale-Up Expansion IoU, Hungarian matching,
    and optional L2-normalized appearance embeddings (inspired by Deep-EIoU / GTATrack Section 4.4)."""

    def __init__(
        self,
        det_high: float = 0.70,
        det_low: float = 0.30,
        expand_ratios: Sequence[float] = (0.5, 0.7, 0.9),
        prox_thresh: float = 0.85,
        app_weight: float = 0.50,
        app_thresh: float = 0.45,
        max_age: int = 45,
        min_hits: int = 3,
        emb_alpha: float = 0.90,
    ):
        if not 0.0 < det_low < det_high <= 1.0:
            raise ValueError(f"need 0 < det_low < det_high <= 1, got det_low={det_low}, det_high={det_high}")
        self.det_high = float(det_high)
        self.det_low = float(det_low)
        self.expand_ratios = tuple(float(e) for e in expand_ratios)
        self.prox_thresh = float(prox_thresh)
        self.app_weight = float(app_weight)
        self.app_thresh = float(app_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.emb_alpha = float(emb_alpha)
        self.tracks: List[_LiveTrack] = []
        self.finished: Dict[int, Tracklet] = {}
        self._next_id = 1
        self._frame_idx = 0

    def _cost_matrix(
        self,
        t_indices: Sequence[int],
        d_indices: np.ndarray,
        boxes: np.ndarray,
        embs: Optional[np.ndarray],
        valid_emb: Optional[np.ndarray],
        ratio: float,
        use_app: bool,
    ) -> np.ndarray:
        t_boxes = np.stack([self.tracks[i].box for i in t_indices], axis=0)
        d_boxes = boxes[d_indices]
        eiou = iou_matrix(expand_boxes(t_boxes, ratio), expand_boxes(d_boxes, ratio))
        c_spatial = 1.0 - eiou
        cost = np.where(c_spatial <= self.prox_thresh, c_spatial, 1e6)

        if use_app and self.app_weight > 0.0 and embs is not None:
            for r, ti in enumerate(t_indices):
                te = self.tracks[ti].emb
                if te is None:
                    continue
                for c, di in enumerate(d_indices):
                    if cost[r, c] > self.prox_thresh:
                        continue
                    if valid_emb is not None and not valid_emb[di]:
                        continue
                    de = embs[di]
                    if np.linalg.norm(de) < 1e-8:
                        continue
                    c_app = float(np.clip(1.0 - float(np.dot(te, de)), 0.0, 1.0))
                    if c_app > self.app_thresh:
                        cost[r, c] = 1e6
                    else:
                        cost[r, c] = (1.0 - self.app_weight) * c_spatial[r, c] + self.app_weight * c_app
        return cost

    def _commit_track(self, t: _LiveTrack) -> None:
        if not t.confirmed or not t.history:
            return
        tl = Tracklet(track_id=t.track_id)
        for f, b, sc, emb, lab in t.history:
            tl.frames.append(f)
            tl.boxes.append(b)
            tl.scores.append(sc)
            if emb is not None:
                tl.embs.append(emb)
            if lab is not None:
                tl.labs.append(lab)
        self.finished[t.track_id] = tl

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        embs: Optional[np.ndarray] = None,
        labs: Optional[np.ndarray] = None,
        valid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self._frame_idx
        self._frame_idx += 1

        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        keep = scores >= self.det_low
        boxes, scores = boxes[keep], scores[keep]
        embs_k = np.asarray(embs, dtype=np.float64)[keep] if embs is not None else None
        labs_k = np.asarray(labs, dtype=np.float64)[keep] if labs is not None else None
        valid_k = np.asarray(valid, dtype=bool)[keep] if valid is not None else None

        high = np.flatnonzero(scores >= self.det_high)
        low = np.flatnonzero(scores < self.det_high)

        matched: Dict[int, int] = {}
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high = list(high.tolist())

        # Stage 1: Iterative Scale-Up EIoU + Appearance on high-score detections
        for ratio in self.expand_ratios:
            if not unmatched_tracks or not unmatched_high:
                break
            d_arr = np.asarray(unmatched_high, dtype=int)
            cost = self._cost_matrix(unmatched_tracks, d_arr, boxes, embs_k, valid_k, ratio, use_app=True)
            pairs = hungarian_match(cost, max_cost=self.prox_thresh)
            if not pairs:
                continue
            used_r, used_c = set(), set()
            for r, c in pairs:
                matched[unmatched_tracks[r]] = int(d_arr[c])
                used_r.add(r)
                used_c.add(c)
            unmatched_tracks = [t for r, t in enumerate(unmatched_tracks) if r not in used_r]
            unmatched_high = [d for c, d in enumerate(unmatched_high) if c not in used_c]

        # Stage 2: ByteTrack low-score continuation for remaining confirmed tracks (spatial EIoU only)
        conf_rem = [ti for ti in unmatched_tracks if self.tracks[ti].confirmed]
        if conf_rem and len(low) > 0:
            low_ratio = self.expand_ratios[0] if self.expand_ratios else 0.5
            cost_low = self._cost_matrix(conf_rem, low, boxes, None, None, low_ratio, use_app=False)
            for r, c in hungarian_match(cost_low, max_cost=min(0.75, self.prox_thresh)):
                matched[conf_rem[r]] = int(low[c])

        out_ids, out_boxes, out_scores = [], [], []
        survivors: List[_LiveTrack] = []
        for ti, t in enumerate(self.tracks):
            if ti in matched:
                di = matched[ti]
                t.box = boxes[di].copy()
                t.score = float(scores[di])
                t.hits += 1
                t.missed = 0
                de = None
                if embs_k is not None and (valid_k is None or valid_k[di]) and np.linalg.norm(embs_k[di]) > 1e-8:
                    de = _l2_normalize(embs_k[di])
                    t.emb = de if t.emb is None else _l2_normalize(self.emb_alpha * t.emb + (1.0 - self.emb_alpha) * de)
                dl = None
                if labs_k is not None and (valid_k is None or valid_k[di]) and np.all(np.isfinite(labs_k[di])):
                    dl = labs_k[di].copy()
                if t.hits >= self.min_hits:
                    t.confirmed = True
                t.history.append((f, t.box.copy(), t.score, de, dl))
                if t.confirmed:
                    out_ids.append(t.track_id)
                    out_boxes.append(t.box)
                    out_scores.append(t.score)
                survivors.append(t)
            elif t.confirmed:
                t.missed += 1
                if t.missed <= self.max_age:
                    survivors.append(t)
                else:
                    self._commit_track(t)
            # unconfirmed and unmatched: dropped

        used_dets = set(matched.values())
        for di in high.tolist():
            if di in used_dets:
                continue
            de = None
            if embs_k is not None and (valid_k is None or valid_k[di]) and np.linalg.norm(embs_k[di]) > 1e-8:
                de = _l2_normalize(embs_k[di])
            dl = None
            if labs_k is not None and (valid_k is None or valid_k[di]) and np.all(np.isfinite(labs_k[di])):
                dl = labs_k[di].copy()
            t = _LiveTrack(
                track_id=self._next_id,
                box=boxes[di].copy(),
                score=float(scores[di]),
                emb=de,
                hits=1,
                missed=0,
                confirmed=(self.min_hits <= 1),
            )
            t.history.append((f, t.box.copy(), t.score, de, dl))
            self._next_id += 1
            survivors.append(t)
            if t.confirmed:
                out_ids.append(t.track_id)
                out_boxes.append(t.box)
                out_scores.append(t.score)

        self.tracks = survivors
        return (
            np.asarray(out_ids, dtype=np.int64),
            np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(out_scores, dtype=np.float32),
        )

    def finalize_tracklets(self) -> Dict[int, Tracklet]:
        """Commit all remaining confirmed live tracks and return ``{track_id: Tracklet}``."""
        for t in self.tracks:
            self._commit_track(t)
        self.tracks = []
        return dict(self.finished)


def _box_center(b: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float64)


def _lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Perceptual Lab distance weighting chroma (a, b) twice as strongly as lightness (L) for shadow robustness."""
    d = a - b
    return float(np.sqrt(0.25 * d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))


def gta_link_connect(
    tracklets: Dict[int, Tracklet],
    max_gap_frames: int = 300,
    max_speed_px: float = 12.0,
    base_dist_px: float = 60.0,
    merge_cos_thresh: float = 0.30,
    team_color_gate: float = 35.0,
    min_track_len: int = 12,
) -> Dict[int, Tracklet]:
    """Offline Global Tracklet Association (Connector): iteratively merges non-overlapping tracklets
    using mean appearance cosine distance, gap-scaled spatial plausibility, and a same-team kit-color gate."""
    # Deep-copy into mutable list of tracklets
    active: Dict[int, Tracklet] = {
        tid: Tracklet(
            track_id=tid,
            frames=list(tl.frames),
            boxes=[b.copy() for b in tl.boxes],
            scores=list(tl.scores),
            embs=[e.copy() for e in tl.embs],
            labs=[l.copy() for l in tl.labs],
        )
        for tid, tl in tracklets.items()
        if len(tl.frames) > 0
    }

    def pair_cost(a: Tracklet, b: Tracklet) -> Optional[float]:
        # Ensure a precedes b strictly without frame overlap
        if a.end_frame >= b.start_frame:
            return None
        gap = b.start_frame - a.end_frame
        if gap > max_gap_frames:
            return None
        dist = float(np.linalg.norm(_box_center(b.boxes[0]) - _box_center(a.boxes[-1])))
        max_dist = base_dist_px + max_speed_px * gap
        if dist > max_dist:
            return None
        # Same-team kit-color gate (soccer-specific addition)
        if team_color_gate > 0.0:
            la, lb = a.median_lab(), b.median_lab()
            if la is not None and lb is not None and _lab_distance(la, lb) > team_color_gate:
                return None
        ea, eb = a.mean_emb(), b.mean_emb()
        if ea is not None and eb is not None:
            cos_d = float(np.clip(1.0 - float(np.dot(ea, eb)), 0.0, 1.0))
            if cos_d > merge_cos_thresh:
                return None
            # Small spatial tie-breaker favors nearby merges among equal-appearance candidates
            return cos_d + 0.05 * (dist / max(1.0, max_dist))
        # Pure spatial fallback when no embeddings are provided
        if gap <= 30 and dist <= max_dist * 0.5:
            return dist / max(1.0, max_dist)
        return None

    while True:
        ids = sorted(active.keys(), key=lambda k: active[k].start_frame)
        best_cost = float("inf")
        best_pair: Optional[Tuple[int, int]] = None
        for i in range(len(ids)):
            ta = active[ids[i]]
            for j in range(i + 1, len(ids)):
                tb = active[ids[j]]
                if tb.start_frame - ta.end_frame > max_gap_frames:
                    # Sorted by start_frame: later tracklets will only have even larger start_frame
                    break
                c = pair_cost(ta, tb)
                if c is not None and c < best_cost:
                    best_cost = c
                    best_pair = (ids[i], ids[j])
        if best_pair is None:
            break
        ia, ib = best_pair
        ta, tb = active[ia], active.pop(ib)
        ta.frames.extend(tb.frames)
        ta.boxes.extend(tb.boxes)
        ta.scores.extend(tb.scores)
        ta.embs.extend(tb.embs)
        ta.labs.extend(tb.labs)

    if min_track_len > 1:
        active = {tid: tl for tid, tl in active.items() if len(tl.frames) >= min_track_len}
    return active


def _deterministic_two_means(points: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic 2-means clustering in weighted Lab space, seeded with the two farthest points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    w = np.array([0.5, 1.0, 1.0], dtype=np.float64)
    wpts = pts * w
    # Farthest pair initialization (deterministic)
    diff = wpts[:, None, :] - wpts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    i0, i1 = np.unravel_index(int(np.argmax(d2)), d2.shape)
    c0, c1 = pts[i0].copy(), pts[i1].copy()
    for _ in range(iters):
        d0 = np.sum(((pts - c0) * w) ** 2, axis=1)
        d1 = np.sum(((pts - c1) * w) ** 2, axis=1)
        m0 = d0 <= d1
        if not np.any(m0) or np.all(m0):
            break
        nc0, nc1 = pts[m0].mean(axis=0), pts[~m0].mean(axis=0)
        if np.allclose(nc0, c0) and np.allclose(nc1, c1):
            break
        c0, c1 = nc0, nc1
    return c0, c1


def filter_officials_and_roster(
    tracklets: Dict[int, Tracklet],
    frame_width: int,
    frame_height: int,
    official_lab_thresh: float = 33.0,
    gk_margin: float = 0.14,
    ref_min_len: int = 40,
) -> Tuple[Dict[int, Tracklet], Dict[int, bool]]:
    """Track-level kit-color official filter with goalmouth goalkeeper exemption.

    1. Fits 2 outfield team kit centroids in torso Lab space on long midfield tracklets.
    2. Exempts goalkeepers whose median x lies in the left/right ``gk_margin`` goalmouth zones.
    3. Drops non-goalkeeper tracklets whose torso Lab distance to both team centroids exceeds
       ``official_lab_thresh`` (match officials / sideline staff) or whose median y lies outside
       the pitch vertical envelope of reference tracks.
    Returns ``(kept_tracklets, is_team_or_gk_map)``.
    """
    if not tracklets:
        return {}, {}

    # Gather long midfield reference tracklets for team kit clustering and pitch y-envelope
    ref_labs: List[np.ndarray] = []
    ref_ys: List[float] = []
    for tl in tracklets.values():
        if len(tl.frames) < ref_min_len:
            continue
        med_c = tl.median_center()
        if gk_margin * frame_width <= med_c[0] <= (1.0 - gk_margin) * frame_width:
            lab = tl.median_lab()
            if lab is not None and np.all(np.isfinite(lab)):
                ref_labs.append(lab)
                ref_ys.append(float(med_c[1]))

    # Relax length floor if clip/window is short
    if len(ref_labs) < 6:
        ref_labs, ref_ys = [], []
        for tl in tracklets.values():
            med_c = tl.median_center()
            if gk_margin * frame_width <= med_c[0] <= (1.0 - gk_margin) * frame_width:
                lab = tl.median_lab()
                if lab is not None and np.all(np.isfinite(lab)):
                    ref_labs.append(lab)
                    ref_ys.append(float(med_c[1]))

    if len(ref_labs) < 4:
        return dict(tracklets), {tid: True for tid in tracklets}

    c0, c1 = _deterministic_two_means(np.stack(ref_labs, axis=0))
    y_lo = float(np.percentile(ref_ys, 2)) - 0.08 * frame_height
    y_hi = float(np.percentile(ref_ys, 98)) + 0.08 * frame_height

    kept: Dict[int, Tracklet] = {}
    is_core: Dict[int, bool] = {}
    for tid, tl in tracklets.items():
        med_c = tl.median_center()
        is_gk = (med_c[0] < gk_margin * frame_width) or (med_c[0] > (1.0 - gk_margin) * frame_width)
        if is_gk:
            kept[tid] = tl
            is_core[tid] = True
            continue
        if med_c[1] < y_lo or med_c[1] > y_hi:
            continue
        lab = tl.median_lab()
        if lab is not None and np.all(np.isfinite(lab)):
            d = min(_lab_distance(lab, c0), _lab_distance(lab, c1))
            if d > official_lab_thresh:
                continue
        kept[tid] = tl
        is_core[tid] = True
    return kept, is_core


def tracklets_to_frames(
    tracklets: Dict[int, Tracklet],
    n_frames: int,
    roster_k: Optional[int] = None,
    is_core: Optional[Dict[int, bool]] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
    """Convert ``{track_id: Tracklet}`` back to per-frame ``(ids, boxes)`` and ``scores`` lists,
    optionally applying a soft per-frame roster cap ranked by ``(is_core, track_length, score)``."""
    per_f: List[List[Tuple[int, np.ndarray, float, int, int]]] = [[] for _ in range(n_frames)]
    for tid, tl in tracklets.items():
        tlen = len(tl.frames)
        core = 1 if (is_core is None or is_core.get(tid, True)) else 0
        for f, b, sc in zip(tl.frames, tl.boxes, tl.scores):
            if 0 <= f < n_frames:
                per_f[f].append((tid, b, sc, core, tlen))

    preds: List[Tuple[np.ndarray, np.ndarray]] = []
    scores: List[np.ndarray] = []
    for f in range(n_frames):
        items = per_f[f]
        if not items:
            preds.append((np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32)))
            scores.append(np.zeros(0, dtype=np.float32))
            continue
        if roster_k is not None and len(items) > roster_k:
            items = sorted(items, key=lambda x: (-x[3], -x[4], -x[2], x[0]))[:roster_k]
        items = sorted(items, key=lambda x: x[0])
        ids_f = np.asarray([x[0] for x in items], dtype=np.int64)
        boxes_f = np.asarray([x[1] for x in items], dtype=np.float32).reshape(-1, 4)
        sc_f = np.asarray([x[2] for x in items], dtype=np.float32)
        preds.append((ids_f, boxes_f))
        scores.append(sc_f)
    return preds, scores


def run_tracker_v3(
    dets: Sequence[Tuple[np.ndarray, np.ndarray]],
    features: Optional[Sequence[Dict[str, np.ndarray]]],
    width: int,
    height: int,
    params: Optional[dict] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
    """Full Tracker v3 pipeline: DeepEIoU online -> optional GTA-Link offline -> optional Official Filter + Soft Roster."""
    cfg = dict(DEFAULT_V3_PARAMS)
    if params:
        cfg.update(params)

    tracker = DeepEIoUTracker(
        det_high=float(cfg["det_high"]),
        det_low=float(cfg["det_low"]),
        expand_ratios=tuple(cfg["expand_ratios"]),
        prox_thresh=float(cfg["prox_thresh"]),
        app_weight=float(cfg["app_weight"]),
        app_thresh=float(cfg["app_thresh"]),
        max_age=int(cfg["max_age"]),
        min_hits=int(cfg["min_hits"]),
        emb_alpha=float(cfg["emb_alpha"]),
    )
    for f, (b, s) in enumerate(dets):
        if features is not None:
            feat = features[f]
            tracker.update(b, s, feat["emb"], feat["torso_lab"], feat["valid"])
        else:
            tracker.update(b, s)

    tracklets = tracker.finalize_tracklets()
    if cfg.get("use_gta", True):
        tracklets = gta_link_connect(
            tracklets,
            max_gap_frames=int(cfg["max_gap_frames"]),
            max_speed_px=float(cfg["max_speed_px"]),
            base_dist_px=float(cfg["base_dist_px"]),
            merge_cos_thresh=float(cfg["merge_cos_thresh"]),
            team_color_gate=float(cfg["team_color_gate"]),
            min_track_len=int(cfg["min_track_len"]),
        )
    is_core = None
    if cfg.get("use_official_filter", True):
        tracklets, is_core = filter_officials_and_roster(
            tracklets,
            frame_width=width,
            frame_height=height,
            official_lab_thresh=float(cfg["official_lab_thresh"]),
            gk_margin=float(cfg["gk_margin"]),
            ref_min_len=int(cfg["ref_min_len"]),
        )
    roster_k = cfg.get("roster_k", None)
    return tracklets_to_frames(tracklets, len(dets), roster_k=roster_k, is_core=is_core)


def pick_final_v3(rows: Sequence[dict], tol: float = 0.005) -> dict:
    """Selection rule (docs/phase2_amendment_tracker_v3.md): best TrackEval mean HOTA; within tol -> lowest FP, then lowest IDSW."""
    best = max(r["hota"] for r in rows)
    close = [r for r in rows if r["hota"] >= best - tol]
    return min(close, key=lambda r: (r["fp"], r["idsw"]))
