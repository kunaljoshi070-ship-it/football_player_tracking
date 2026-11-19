"""Whole-clip detection in resumable chunks (Phase II, WS-A1).

A full 4-minute clip is ~6,000 frames, about 1.8 h on a T4 at Phase I speed, and Colab sessions disconnect.
Detections are therefore written per chunk of ``chunk`` frames (``<clip>_c<start>.npz``, same layout as the
Phase I segment files) and a rerun only computes the chunks that are missing. The planning, file naming and
assembly are plain NumPy and unit-tested; ``detect_clip`` needs torch and a GPU and is NOT covered by tests.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def plan_chunks(n_frames: int, chunk: int, start: int = 0, end: int = None) -> List[Tuple[int, int]]:
    """``[(chunk_start, length), ...]`` covering frames ``start..end-1`` (the last chunk may be shorter)."""
    end = n_frames if end is None else min(end, n_frames)
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if not 0 <= start < end:
        raise ValueError(f"need 0 <= start < end <= {n_frames}, got start={start}, end={end}")
    return [(s, min(chunk, end - s)) for s in range(start, end, chunk)]


def chunk_path(det_dir: Path, clip_id: str, start: int) -> Path:
    return Path(det_dir) / f"{clip_id}_c{start}.npz"


def chunk_complete(path: Path, length: int) -> bool:
    """True if the file exists, loads, and holds the expected number of frames (a half-written file does not count)."""
    try:
        z = np.load(path)
        return int(z["n_frames"]) == length and len(z["frame"]) == len(z["boxes"]) == len(z["scores"])
    except Exception:
        return False


def missing_chunks(det_dir: Path, clip_id: str, plan: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(s, n) for s, n in plan if not chunk_complete(chunk_path(det_dir, clip_id, s), n)]


def save_chunk(det_dir: Path, clip_id: str, start: int, frames, boxes, scores, n_frames: int) -> Path:
    """Atomic save: write to a temp name, then rename, so an interrupted run never leaves a valid-looking partial file."""
    path = chunk_path(det_dir, clip_id, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, start=start, n_frames=n_frames, frame=np.asarray(frames, dtype=np.int32),
                        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4), scores=np.asarray(scores, dtype=np.float32))
    tmp.replace(path)
    return path


def assemble(det_dir: Path, clip_id: str, plan: Sequence[Tuple[int, int]]):
    """Per-frame ``(boxes, scores)`` for the whole planned range, in order. Raises if any chunk is missing."""
    gone = missing_chunks(det_dir, clip_id, plan)
    if gone:
        raise FileNotFoundError(f"clip {clip_id}: {len(gone)} chunk(s) missing or incomplete, first start {gone[0][0]}; rerun detect")
    out = []
    for s, n in plan:
        z = np.load(chunk_path(det_dir, clip_id, s))
        frame, boxes, scores = z["frame"], z["boxes"], z["scores"]
        order = np.argsort(frame, kind="stable")
        frame, boxes, scores = frame[order], boxes[order], scores[order]
        edges = np.searchsorted(frame, np.arange(n + 1))
        out += [(boxes[edges[i]:edges[i + 1]], scores[edges[i]:edges[i + 1]]) for i in range(n)]
    return out


def detect_clip(model, clip, det_dir: Path, plan, tile, overlap, device, batch_size=2, merge_iou=0.5, drop_edge_px=2.0, log=print) -> int:
    """Compute the missing chunks of ``plan`` for ``clip`` (GPU). Returns the number of chunks written. Untested glue:
    mirrors ``track_eval.cmd_detect`` (sequential, frame-exact decoding), one decode pass per missing chunk."""
    import cv2

    from .detect import detect_tiles, merge_tile_detections
    from .tiles import tile_grid

    windows = tile_grid(clip.width, clip.height, tile[0], tile[1], overlap)
    todo = missing_chunks(det_dir, clip.clip_id, plan)
    log(f"clip {clip.clip_id}: {len(plan) - len(todo)} of {len(plan)} chunks already done, {len(todo)} to run")
    for start, length in todo:
        cap = cv2.VideoCapture(str(clip.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {clip.video_path}")
        frames, boxes_all, scores_all = [], [], []
        t0 = time.time()
        try:
            for i in range(start):
                if not cap.grab():
                    raise RuntimeError(f"decoding stopped at frame {i}")
            for k in range(length):
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"could not decode frame {start + k}")
                if bgr.shape[1] != clip.width or bgr.shape[0] != clip.height:
                    raise ValueError(f"clip {clip.clip_id}: frame size differs from manifest")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                b, s = merge_tile_detections(detect_tiles(model, rgb, windows, device, batch_size), windows,
                                             (clip.width, clip.height), merge_iou, drop_edge_px)
                frames.append(np.full(len(b), k, dtype=np.int32)); boxes_all.append(b); scores_all.append(s)
        finally:
            cap.release()
        save_chunk(det_dir, clip.clip_id, start, np.concatenate(frames), np.concatenate(boxes_all), np.concatenate(scores_all), length)
        log(f"clip {clip.clip_id}: chunk {start}..{start + length - 1} saved, {(time.time() - t0) / length:.2f} s/frame")
    return len(todo)
