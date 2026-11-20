import numpy as np
from .metrics import iou_matrix


def suppress_static(frames, window=250, step=5, min_frac=0.9, iou_thr=0.8, min_score=0.5):
    """frames: list of (boxes xyxy (N, 4), scores (N,)) for consecutive frames.
    Drops a box if a confident, near-identical box (IoU >= iou_thr) exists in at least min_frac of the
    other probed frames within +-window/2. Players move; static unlabelled people do not."""
    n, out = len(frames), []
    for i, (b, s) in enumerate(frames):
        b = np.asarray(b).reshape(-1, 4)
        if len(b) == 0:
            out.append((b, s))
            continue
        probes = [j for j in range(max(0, i - window // 2), min(n, i + window // 2 + 1), step) if j != i]
        hits = np.zeros(len(b))
        for j in probes:
            bj, sj = frames[j]
            bj = np.asarray(bj).reshape(-1, 4)[np.asarray(sj) >= min_score]
            if len(bj):
                hits += iou_matrix(b, bj).max(axis=1) >= iou_thr
        static = hits / max(1, len(probes)) >= min_frac
        out.append((b[~static], np.asarray(s)[~static]))
    return out