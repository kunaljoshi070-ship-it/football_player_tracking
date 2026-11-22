"""Tracker v3.1 (Phase II, WS-C2/C3/C4): Kit-Aware Deep-EIoU online tracking +
GTA-Link offline stitching + Calibrated Track-Level Official Filter & Soft Roster Prior.

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

# Pre-registered compact grid (16 settings): sweeps detection threshold, GTA merge threshold,
# calibrated official Lab threshold, and soft roster cap.
GRID_V3: Dict[str, tuple] = {
    "det_high": (0.70, 0.80),
    "merge_cos_thresh": (0.25, 0.35),
    "official_lab_thresh": (44.0, 52.0),
    "roster_k": (None, 23),
}

DEFAULT_V3_PARAMS: Dict[str, object] = {
    "det_high": 0.75,
    "det_low": 0.45,
    "expand_ratios": (0.0, 0.35, 0.70),
    "prox_thresh": 0.80,
    "app_weight": 0.50,
    "app_thresh": 0.45,
    "online_lab_gate": 42.0,
    "max_age": 20,
    "min_hits": 3,
    "emb_alpha": 0.90,
    "use_gta": True,
    "max_gap_frames": 300,
    "max_speed_px": 12.0,
    "base_dist_px": 60.0,
    "merge_cos_thresh": 0.30,
    "team_color_gate": 45.0,
    "min_track_len": 12,
    "use_official_filter": True,
    "official_lab_thresh": 46.0,
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


def _box_center(b: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float64)


def _lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Perceptual Lab distance weighting chroma (a, b) twice as strongly as lightness (L) for shadow robustness."""
    d = a - b
    return float(np.sqrt(0.25 * d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))


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
    lab: Optional[np.ndarray] = None
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    history: List[Tuple[int, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]] = field(default_factory=list)


class DeepEIoUTracker:
    """Motion-agnostic online tracker using Iterative Scale-Up Expansion IoU (starting at strict e=0.0
    to prevent crossing swaps), Hungarian matching, online torso-Lab team gate, and L2-normalized ReID."""

    def __init__(
        self,
        det_high: float = 0.75,
        det_low: float = 0.45,
        expand_ratios: Sequence[float] = (0.0, 0.35, 0.70),
        prox_thresh: float = 0.80,
        app_weight: float = 0.50,
        app_thresh: float = 0.45,
        online_lab_gate: float = 42.0,
        max_age: int = 20,
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
        self.online_lab_gate = float(online_lab_gate)
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
        labs: Optional[np.ndarray],
        valid_feat: Optional[np.ndarray],
        ratio: float,
        use_app: bool,
        max_spatial: float,
    ) -> np.ndarray:
        t_boxes = np.stack([self.tracks[i].box for i in t_indices], axis=0)
        d_boxes = boxes[d_indices]
        eiou = iou_matrix(expand_boxes(t_boxes, ratio), expand_boxes(d_boxes, ratio))
        c_spatial = 1.0 - eiou
        cost = np.where(c_spatial <= max_spatial, c_spatial, 1e6)

        if not use_app:
            return cost

        # Vectorized online Lab team gate (prevents opposite-team crossing swaps)
        if self.online_lab_gate > 0.0 and labs is not None:
            for r, ti in enumerate(t_indices):
                tl = self.tracks[ti].lab
                if tl is None:
                    continue
                for c, di in enumerate(d_indices):
                    if cost[r, c] > max_spatial:
                        continue
                    if valid_feat is not None and not valid_feat[di]:
                        continue
                    dl = labs[di]
                    if np.all(np.isfinite(dl)) and _lab_distance(tl, dl) > self.online_lab_gate:
                        cost[r, c] = 1e6

        if self.app_weight > 0.0 and embs is not None:
            for r, ti in enumerate(t_indices):
                te = self.tracks[ti].emb
                if te is None:
                    continue
                for c, di in enumerate(d_indices):
                    if cost[r, c] > max_spatial:
                        continue
                    if valid_feat is not None and not valid_feat[di]:
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

        # Stage 1: Iterative Scale-Up EIoU + Lab Gate + ReID on high-score detections
        for ratio in self.expand_ratios:
            if not unmatched_tracks or not unmatched_high:
                break
            d_arr = np.asarray(unmatched_high, dtype=int)
            # Tighter spatial threshold on strict pass (e=0.0 -> IoU >= 0.35), wider on expanded passes
            max_sp = min(0.65, self.prox_thresh) if ratio == 0.0 else self.prox_thresh
            cost = self._cost_matrix(unmatched_tracks, d_arr, boxes, embs_k, labs_k, valid_k, ratio,
                                     use_app=True, max_spatial=max_sp)
            pairs = hungarian_match(cost, max_cost=max_sp)
            if not pairs:
                continue
            used_r, used_c = set(), set()
            for r, c in pairs:
                matched[unmatched_tracks[r]] = int(d_arr[c])
                used_r.add(r)
                used_c.add(c)
            unmatched_tracks = [t for r, t in enumerate(unmatched_tracks) if r not in used_r]
            unmatched_high = [d for c, d in enumerate(unmatched_high) if c not in used_c]

        # Stage 2: ByteTrack low-score continuation for remaining confirmed tracks (strict spatial IoU/EIoU)
        conf_rem = [ti for ti in unmatched_tracks if self.tracks[ti].confirmed]
        if conf_rem and len(low) > 0:
            cost_low = self._cost_matrix(conf_rem, low, boxes, None, labs_k, valid_k, ratio=0.25,
                                         use_app=False, max_spatial=0.65)
            for r, c in hungarian_match(cost_low, max_cost=0.65):
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
                    t.lab = dl if t.lab is None else (0.85 * t.lab + 0.15 * dl)
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
                lab=dl,
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


def gta_link_connect(
    tracklets: Dict[int, Tracklet],
    max_gap_frames: int = 300,
    max_speed_px: float = 12.0,
    base_dist_px: float = 60.0,
    merge_cos_thresh: float = 0.30,
    team_color_gate: float = 45.0,
    min_track_len: int = 12,
) -> Dict[int, Tracklet]:
    """Fast offline Global Tracklet Association (Connector) with cached summaries.

    Iteratively merges the lowest-cost non-overlapping tracklet pair satisfying:
    - temporal gap in ``(0, max_gap_frames]``,
    - spatial plausibility ``dist <= base_dist_px + max_speed_px * gap``,
    - same-team kit-color gate ``lab_dist <= team_color_gate``,
    - mean L2-normalized appearance cosine distance ``<= merge_cos_thresh``.
    """
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

    # Cache per-tracklet summary stats so inner loop is O(1) per pair (50x faster on full clips)
    def summarize(tl: Tracklet) -> dict:
        return {
            "start": tl.frames[0],
            "end": tl.frames[-1],
            "c_start": _box_center(tl.boxes[0]),
            "c_end": _box_center(tl.boxes[-1]),
            "emb": tl.mean_emb(),
            "lab": tl.median_lab(),
        }

    stats: Dict[int, dict] = {tid: summarize(tl) for tid, tl in active.items()}

    def pair_cost(sa: dict, sb: dict) -> Optional[float]:
        if sa["end"] >= sb["start"]:
            return None
        gap = sb["start"] - sa["end"]
        if gap > max_gap_frames:
            return None
        dist = float(np.linalg.norm(sb["c_start"] - sa["c_end"]))
        max_dist = base_dist_px + max_speed_px * gap
        if dist > max_dist:
            return None
        if team_color_gate > 0.0 and sa["lab"] is not None and sb["lab"] is not None:
            if _lab_distance(sa["lab"], sb["lab"]) > team_color_gate:
                return None
        if sa["emb"] is not None and sb["emb"] is not None:
            cos_d = float(np.clip(1.0 - float(np.dot(sa["emb"], sb["emb"])), 0.0, 1.0))
            if cos_d > merge_cos_thresh:
                return None
            return cos_d + 0.05 * (dist / max(1.0, max_dist))
        if gap <= 30 and dist <= max_dist * 0.5:
            return dist / max(1.0, max_dist)
        return None

    while True:
        ids = sorted(active.keys(), key=lambda k: stats[k]["start"])
        best_cost = float("inf")
        best_pair: Optional[Tuple[int, int]] = None
        for i in range(len(ids)):
            sa = stats[ids[i]]
            end_a = sa["end"]
            for j in range(i + 1, len(ids)):
                sb = stats[ids[j]]
                if sb["start"] - end_a > max_gap_frames:
                    break
                c = pair_cost(sa, sb)
                if c is not None and c < best_cost:
                    best_cost = c
                    best_pair = (ids[i], ids[j])
        if best_pair is None:
            break
        ia, ib = best_pair
        ta, tb = active[ia], active.pop(ib)
        stats.pop(ib)
        ta.frames.extend(tb.frames)
        ta.boxes.extend(tb.boxes)
        ta.scores.extend(tb.scores)
        ta.embs.extend(tb.embs)
        ta.labs.extend(tb.labs)
        stats[ia] = summarize(ta)

    if min_track_len > 1:
        active = {tid: tl for tid, tl in active.items() if len(tl.frames) >= min_track_len}
    return active


def _deterministic_two_means(points: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic 2-means clustering in weighted Lab space, seeded with the two farthest points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    w = np.array([0.5, 1.0, 1.0], dtype=np.float64)
    wpts = pts * w
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
    official_lab_thresh: float = 46.0,
    gk_margin: float = 0.14,
    ref_min_len: int = 40,
) -> Tuple[Dict[int, Tracklet], Dict[int, bool]]:
    """Calibrated track-level kit-color official filter with goalmouth goalkeeper exemption.

    Fits 2 outfield team kit centroids in torso Lab space on long midfield tracklets, exempts
    goalkeepers in the left/right ``gk_margin`` goalmouth bands, and drops only true kit-color
    outliers (Lab distance > ``official_lab_thresh``) without clipping wingers near touchlines.
    """
    if not tracklets:
        return {}, {}

    ref_labs: List[np.ndarray] = []
    for tl in tracklets.values():
        if len(tl.frames) < ref_min_len:
            continue
        med_c = tl.median_center()
        if gk_margin * frame_width <= med_c[0] <= (1.0 - gk_margin) * frame_width:
            lab = tl.median_lab()
            if lab is not None and np.all(np.isfinite(lab)):
                ref_labs.append(lab)

    if len(ref_labs) < 6:
        ref_labs = []
        for tl in tracklets.values():
            med_c = tl.median_center()
            if gk_margin * frame_width <= med_c[0] <= (1.0 - gk_margin) * frame_width:
                lab = tl.median_lab()
                if lab is not None and np.all(np.isfinite(lab)):
                    ref_labs.append(lab)

    if len(ref_labs) < 4:
        return dict(tracklets), {tid: True for tid in tracklets}

    c0, c1 = _deterministic_two_means(np.stack(ref_labs, axis=0))

    kept: Dict[int, Tracklet] = {}
    is_core: Dict[int, bool] = {}
    for tid, tl in tracklets.items():
        med_c = tl.median_center()
        is_gk = (med_c[0] < gk_margin * frame_width) or (med_c[0] > (1.0 - gk_margin) * frame_width)
        if is_gk:
            kept[tid] = tl
            is_core[tid] = True
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
    """Full Tracker v3.1 pipeline: Kit-Aware DeepEIoU online -> optional GTA-Link -> optional Official Filter."""
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
        online_lab_gate=float(cfg.get("online_lab_gate", 42.0)),
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
    """Selection rule: best TrackEval mean HOTA; within tol -> lowest FP, then lowest IDSW."""
    best = max(r["hota"] for r in rows)
    close = [r for r in rows if r["hota"] >= best - tol]
    return min(close, key=lambda r: (r["fp"], r["idsw"]))
