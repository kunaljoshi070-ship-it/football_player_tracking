"""C4 roster cap (Phase II). Protocol: docs/phase2_amendment_c4_roster.md. Tracker v2 itself is untouched."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# 2 modes x 3 caps + the no-cap comparison arm = 7
GRID_C4 = [("none", None)] + [(m, k) for m in ("det", "out") for k in (22, 24, 26)]


def cap_detections(dets, k: Optional[int]):
    """Keep the k highest-scoring detections of each frame (original order preserved). ``k=None`` changes nothing."""
    out = []
    for b, s in dets:
        b, s = np.asarray(b).reshape(-1, 4), np.asarray(s).reshape(-1)
        if k is None or len(s) <= k:
            out.append((b, s))
            continue
        keep = np.sort(np.argsort(-s, kind="stable")[:k])
        out.append((b[keep], s[keep]))
    return out


def cap_output(preds, scores, k: Optional[int]):
    """Output at most k tracks per frame, ranked by age (frames since first output) then score. Ages count dropped tracks too."""
    first, new_p, new_s = {}, [], []
    for f, ((ids, boxes), sc) in enumerate(zip(preds, scores)):
        ids = np.asarray(ids).reshape(-1)
        boxes = np.asarray(boxes).reshape(-1, 4)
        sc = np.asarray(sc).reshape(-1)
        for i in ids.tolist():
            first.setdefault(i, f)
        if k is None or len(ids) <= k:
            new_p.append((ids, boxes)); new_s.append(sc)
            continue
        age = np.array([f - first[i] for i in ids.tolist()])
        keep = np.sort(np.lexsort((-sc, -age))[:k])             # primary key is the last one: age, then score
        new_p.append((ids[keep], boxes[keep])); new_s.append(sc[keep])
    return new_p, new_s


def build_arm_roster(dets, params: dict, mode: str, k: Optional[int]):
    """Tracker v2 with the roster cap. ``mode`` is 'none', 'det' or 'out'."""
    from .tracker_v2 import build_arm_v2

    if mode == "none":
        return build_arm_v2(dets, params)
    if mode == "det":
        return build_arm_v2(cap_detections(dets, k), params)
    if mode == "out":
        preds, sc = build_arm_v2(dets, params)
        return cap_output(preds, sc, k)
    raise ValueError(f"unknown roster mode {mode!r}")


def pick_final_c4(rows: Sequence[dict], tol: float = 0.005) -> dict:
    """Rows: ``{'mode', 'k', 'hota', 'idsw'}`` (TrackEval). Best HOTA; within tol: lower IDSW, then larger k (None = largest)."""
    best = max(r["hota"] for r in rows)
    close = [r for r in rows if r["hota"] >= best - tol]
    return min(close, key=lambda r: (r["idsw"], -(10 ** 6 if r["k"] is None else r["k"])))