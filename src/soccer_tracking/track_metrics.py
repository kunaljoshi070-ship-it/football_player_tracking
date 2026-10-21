"""Tracking measures with explicit definitions (all on contiguous frames of one segment).

Per frame, predictions are matched to ground-truth players at IoU >= ``iou_thr``: for predictions that
carry ids, the previous frame's (player, id) pairs are kept first if still valid (CLEAR-MOT style), then
remaining pairs are matched greedily by IoU.

* TP / FP / FN, precision, recall, F1: counted over all frames.
* MODA = 1 - (FN + FP) / n_gt (detection accuracy); MOTA = 1 - (FN + FP + IDSW) / n_gt (needs ids).
* Fragmentation (``gaps``): for each ground-truth player, the number of interruptions, i.e. maximal runs
  of unmatched frames lying *between* two matched frames. ``frag_per_track`` = gaps / number of players.
  Misses before the first or after the last match are not interruptions. Defined for any output.
* IDSW (ids only): times a player's matched prediction id differs from that player's previously matched id.
* ``ids_per_track`` (ids only): mean number of distinct prediction ids matched to a player (1 is ideal).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import iou_matrix
from .tracker import greedy_match

GtFrame = Tuple[np.ndarray, np.ndarray]                      # (ids (G,), boxes_xyxy (G, 4))
PredFrame = Tuple[Optional[np.ndarray], np.ndarray]          # (ids (P,) or None, boxes_xyxy (P, 4))


def score_sequence(gt_seq: Sequence[GtFrame], pred_seq: Sequence[PredFrame], iou_thr: float = 0.5) -> Dict[str, object]:
    """Raw counts for one segment; combine several segments with :func:`summarize`."""
    if len(gt_seq) != len(pred_seq):
        raise ValueError(f"{len(gt_seq)} ground-truth frames but {len(pred_seq)} prediction frames")
    has_ids = all(p[0] is not None for p in pred_seq)
    tp = fp = fn = idsw = n_gt = 0
    matched_seq: Dict[int, List[bool]] = {}
    last_pid: Dict[int, int] = {}
    seen_pids: Dict[int, set] = {}
    prev: Dict[int, int] = {}
    for (gids, gboxes), (pids, pboxes) in zip(gt_seq, pred_seq):
        gboxes = np.asarray(gboxes, dtype=np.float64).reshape(-1, 4)
        pboxes = np.asarray(pboxes, dtype=np.float64).reshape(-1, 4)
        n_g, n_p = len(gboxes), len(pboxes)
        iou = iou_matrix(gboxes, pboxes) if n_g and n_p else np.zeros((n_g, n_p))
        pairs: Dict[int, int] = {}
        used: set = set()
        if has_ids and n_p:
            index = {int(pid): j for j, pid in enumerate(pids)}
            for gi, gid in enumerate(gids):
                pj = index.get(prev.get(int(gid), -1))
                if pj is not None and pj not in used and iou[gi, pj] >= iou_thr:
                    pairs[gi] = pj
                    used.add(pj)
        if n_g and n_p:
            cand = iou.copy()
            for gi in pairs:
                cand[gi, :] = -1.0
            for pj in used:
                cand[:, pj] = -1.0
            for gi, pj in greedy_match(cand, iou_thr):
                pairs[gi] = pj
        m = len(pairs)
        tp, fp, fn, n_gt = tp + m, fp + n_p - m, fn + n_g - m, n_gt + n_g
        new_prev: Dict[int, int] = {}
        for gi, gid in enumerate(gids):
            gid = int(gid)
            matched_seq.setdefault(gid, []).append(gi in pairs)
            if gi in pairs and has_ids:
                pid = int(pids[pairs[gi]])
                if gid in last_pid and last_pid[gid] != pid:
                    idsw += 1
                last_pid[gid] = pid
                seen_pids.setdefault(gid, set()).add(pid)
                new_prev[gid] = pid
        prev = new_prev
    gaps = 0
    for seq in matched_seq.values():
        runs = sum(1 for i, v in enumerate(seq) if v and (i == 0 or not seq[i - 1]))
        gaps += max(0, runs - 1)
    return {"n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn, "gaps": gaps, "n_tracks": len(matched_seq),
            "idsw": idsw if has_ids else None,
            "ids_sum": sum(len(s) for s in seen_pids.values()) if has_ids else None,
            "ids_tracks": len(seen_pids) if has_ids else None}


def summarize(parts: Sequence[Dict[str, object]]) -> Dict[str, Optional[float]]:
    """Add up per-segment counts and compute the reported measures."""
    tot = {k: sum(p[k] for p in parts) for k in ("n_gt", "tp", "fp", "fn", "gaps", "n_tracks")}
    has_ids = all(p["idsw"] is not None for p in parts)
    tp, fp, fn, n_gt = tot["tp"], tot["fp"], tot["fn"], tot["n_gt"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / n_gt if n_gt else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    out: Dict[str, Optional[float]] = {
        "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
        "moda": 1 - (fn + fp) / n_gt if n_gt else float("nan"), "gaps": tot["gaps"], "n_tracks": tot["n_tracks"],
        "frag_per_track": tot["gaps"] / tot["n_tracks"] if tot["n_tracks"] else float("nan"),
        "idsw": None, "mota": None, "ids_per_track": None}
    if has_ids:
        idsw = sum(p["idsw"] for p in parts)
        out["idsw"] = idsw
        out["mota"] = 1 - (fn + fp + idsw) / n_gt if n_gt else float("nan")
        n_t = sum(p["ids_tracks"] for p in parts)
        out["ids_per_track"] = sum(p["ids_sum"] for p in parts) / n_t if n_t else float("nan")
    return out
