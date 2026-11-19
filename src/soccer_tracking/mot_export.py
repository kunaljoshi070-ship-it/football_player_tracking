"""Write ground truth and tracker output in the MOTChallenge folder layout that TrackEval reads.

Layout (``root`` = gt folder, ``trackers`` = trackers folder)::

    <root>/<seq>/gt/gt.txt            frame,id,x,y,w,h,1,1,1
    <root>/<seq>/seqinfo.ini
    <root>/seqmap.txt                 header "name" then one sequence per line
    <trackers>/<name>/data/<seq>.txt  frame,id,x,y,w,h,score,-1,-1,-1

**Frame numbers.** MOTChallenge and TrackEval count frames from 1. This project's labels and detections are
0-based, so every exporter adds ``frame_offset`` (default 1). Forgetting this makes TrackEval report nonsense
(or HOTA 0), so ``tests/test_hota_export.py`` pins it. Boxes are exported as ``x, y, w, h`` from ``xyxy`` input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

Frame = Tuple[np.ndarray, np.ndarray]


def _rows(seq: Sequence[Frame], scores: Optional[Sequence[np.ndarray]], offset: int, gt: bool):
    for k, (ids, boxes) in enumerate(seq):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        for j, (i, b) in enumerate(zip(np.asarray(ids).reshape(-1), boxes)):
            w, h = b[2] - b[0], b[3] - b[1]
            if gt:
                tail = "1,1,1"
            else:
                s = 1.0 if scores is None else float(scores[k][j])
                tail = f"{s:.6f},-1,-1,-1"
            yield f"{k + offset},{int(i)},{b[0]:.3f},{b[1]:.3f},{w:.3f},{h:.3f},{tail}\n"


def write_gt(root: Path, seq: str, gt: Sequence[Frame], width: int, height: int, fps: float = 25.0, frame_offset: int = 1) -> Path:
    d = Path(root) / seq
    (d / "gt").mkdir(parents=True, exist_ok=True)
    (d / "gt" / "gt.txt").write_text("".join(_rows(gt, None, frame_offset, True)), encoding="utf-8")
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq}\nimDir=img1\nframeRate={int(round(fps))}\nseqLength={len(gt)}\n"
        f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n", encoding="utf-8")
    return d / "gt" / "gt.txt"


def write_tracker(trackers: Path, name: str, seq: str, pred: Sequence[Frame],
                  scores: Optional[Sequence[np.ndarray]] = None, frame_offset: int = 1) -> Path:
    d = Path(trackers) / name / "data"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{seq}.txt"
    path.write_text("".join(_rows(pred, scores, frame_offset, False)), encoding="utf-8")
    return path


def write_seqmap(root: Path, seqs: Sequence[str]) -> Path:
    path = Path(root) / "seqmap.txt"
    path.write_text("name\n" + "".join(f"{s}\n" for s in seqs), encoding="utf-8")
    return path
