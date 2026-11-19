"""HOTA, DetA, AssA, LocA, FP, FN and ID switches in NumPy (+ SciPy for the assignment step).

This is a *reference* implementation written from the published HOTA definition (Luiten et al., 2021) and
the structure of TrackEval's HOTA metric. It is used for fast sanity tests and quick development
scoring. Numbers that go into the report or are compared with GTATrack must come from TrackEval
(see ``mot_export.py`` for the layout it reads); ``hota_eval.py`` cross-checks the two when TrackEval is installed.

Inputs per sequence are lists with one entry per frame: ``(ids int (N,), boxes xyxy float (N, 4))``.
Similarity is box IoU. HOTA is averaged over alpha = 0.05, 0.10, ..., 0.95.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .metrics import iou_matrix

Frame = Tuple[np.ndarray, np.ndarray]
ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo(float).eps


def _remap(seq: Sequence[Frame]):
    """Map arbitrary integer ids to 0..K-1 (ids are only meaningful within one sequence)."""
    uniq = sorted({int(i) for ids, _ in seq for i in np.asarray(ids).reshape(-1)})
    lut = {v: k for k, v in enumerate(uniq)}
    return [np.array([lut[int(i)] for i in np.asarray(ids).reshape(-1)], dtype=int) for ids, _ in seq], len(uniq)


def hota_sequence(gt: Sequence[Frame], pred: Sequence[Frame]) -> Dict[str, object]:
    """Raw per-alpha counts for one sequence; combine sequences with :func:`combine`."""
    from scipy.optimize import linear_sum_assignment

    if len(gt) != len(pred):
        raise ValueError(f"{len(gt)} ground-truth frames but {len(pred)} prediction frames")
    gids, n_g = _remap(gt)
    pids, n_p = _remap(pred)
    sims = []
    for (_, gb), (_, pb) in zip(gt, pred):
        gb, pb = np.asarray(gb, float).reshape(-1, 4), np.asarray(pb, float).reshape(-1, 4)
        sims.append(iou_matrix(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb))))
    potential = np.zeros((n_g, n_p))
    g_count, p_count = np.zeros(n_g), np.zeros(n_p)
    for gi, pi, s in zip(gids, pids, sims):
        if len(gi) and len(pi):
            denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
            iou = np.divide(s, denom, out=np.zeros_like(s), where=denom > EPS)
            potential[gi[:, None], pi[None, :]] += iou
        g_count[gi] += 1
        p_count[pi] += 1
    align = potential / np.maximum(g_count[:, None] + p_count[None, :] - potential, EPS)
    n_gt_dets = int(sum(len(g) for g in gids))
    n_pr_dets = int(sum(len(p) for p in pids))
    out = {"n_gt": n_gt_dets, "n_pred": n_pr_dets, "tp": [], "loc_sum": [], "ass_num": []}
    for a in ALPHAS:
        matches = np.zeros((n_g, n_p))
        tp, loc = 0, 0.0
        for gi, pi, s in zip(gids, pids, sims):
            if not len(gi) or not len(pi):
                continue
            score = align[gi[:, None], pi[None, :]] * s
            r, c = linear_sum_assignment(-score)
            ok = s[r, c] >= a - EPS
            r, c = r[ok], c[ok]
            tp += len(r)
            loc += float(s[r, c].sum())
            matches[gi[r], pi[c]] += 1
        ass_a = matches / np.maximum(g_count[:, None] + p_count[None, :] - matches, 1)
        out["tp"].append(tp)
        out["loc_sum"].append(loc)
        out["ass_num"].append(float((matches * ass_a).sum()))
    return out


def id_switches(gt: Sequence[Frame], pred: Sequence[Frame], iou_thr: float = 0.5) -> int:
    """ID switches with CLEAR-style bookkeeping (a player's matched id differs from its previous matched id)."""
    from .tracker import greedy_match

    last: Dict[int, int] = {}
    prev: Dict[int, int] = {}
    n = 0
    for (gi, gb), (pi, pb) in zip(gt, pred):
        gb, pb = np.asarray(gb, float).reshape(-1, 4), np.asarray(pb, float).reshape(-1, 4)
        pairs: Dict[int, int] = {}
        if len(gb) and len(pb):
            iou = iou_matrix(gb, pb)
            index = {int(p): j for j, p in enumerate(pi)}
            used = set()
            for g, gid in enumerate(gi):
                j = index.get(prev.get(int(gid), -1))
                if j is not None and j not in used and iou[g, j] >= iou_thr:
                    pairs[g] = j
                    used.add(j)
            cand = iou.copy()
            for g in pairs:
                cand[g, :] = -1.0
            for j in used:
                cand[:, j] = -1.0
            for g, j in greedy_match(cand, iou_thr):
                pairs[g] = j
        new_prev = {}
        for g, j in pairs.items():
            gid, pid = int(gi[g]), int(pi[j])
            if gid in last and last[gid] != pid:
                n += 1
            last[gid] = pid
            new_prev[gid] = pid
        prev = new_prev
    return n


def combine(parts: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Sum the counts of several sequences (pooled, not a mean of per-sequence HOTA) and compute the measures."""
    n_gt = sum(p["n_gt"] for p in parts)
    n_pr = sum(p["n_pred"] for p in parts)
    tp = np.sum([p["tp"] for p in parts], axis=0).astype(float)
    loc = np.sum([p["loc_sum"] for p in parts], axis=0)
    ass = np.sum([p["ass_num"] for p in parts], axis=0)
    fn, fp = n_gt - tp, n_pr - tp
    deta = tp / np.maximum(1, tp + fn + fp)
    assa = ass / np.maximum(1, tp)
    loca = np.divide(loc, tp, out=np.ones_like(loc), where=tp > 0)
    hota = np.sqrt(deta * assa)
    mid = int(np.argmin(np.abs(ALPHAS - 0.5)))
    return {"HOTA": float(hota.mean()), "DetA": float(deta.mean()), "AssA": float(assa.mean()), "LocA": float(loca.mean()),
            "HOTA_at_0.5": float(hota[mid]), "FN_at_0.5": float(fn[mid]), "FP_at_0.5": float(fp[mid]),
            "n_gt": int(n_gt), "n_pred": int(n_pr)}


def evaluate(gt: Sequence[Frame], pred: Sequence[Frame]) -> Dict[str, float]:
    """One sequence: HOTA family plus ID switches at IoU 0.5."""
    out = combine([hota_sequence(gt, pred)])
    out["IDSW"] = id_switches(gt, pred)
    return out
