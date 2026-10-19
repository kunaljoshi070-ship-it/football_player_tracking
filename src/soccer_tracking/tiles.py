"""Pure-NumPy helpers for tiling full frames and mapping boxes into tiles.

Why tiles: players are small (median about 20 x 41 px in a 3840-wide frame),
and resizing a whole 4096x1080 frame to a detector's default input would
shrink them to a few pixels. Tiles are cut at native scale instead.

Coordinates: a window is ``(x0, y0, x1, y1)`` in full-frame pixels (x1, y1
exclusive). Boxes go in as ``(x, y, w, h)`` and come out as ``(x1, y1, x2, y2)``
relative to the window's top-left corner.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Window = Tuple[int, int, int, int]


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    """Start offsets of tiles along one axis; the last tile is shifted back to
    end exactly at ``length`` (so the final overlap can be larger than ``overlap``)."""
    if tile <= 0:
        raise ValueError(f"tile size must be positive, got {tile}")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must satisfy 0 <= overlap < tile, got {overlap} (tile {tile})")
    if tile >= length:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_grid(width: int, height: int, tile_w: int, tile_h: int, overlap: int) -> List[Window]:
    """Row-major list of windows covering the whole frame. A tile larger than the
    frame along an axis is shrunk to the frame size on that axis."""
    xs = tile_starts(width, tile_w, overlap)
    ys = tile_starts(height, tile_h, overlap)
    tw, th = min(tile_w, width), min(tile_h, height)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def boxes_to_tile(
    boxes_xywh: np.ndarray,
    window: Window,
    min_visible: float = 0.5,
    min_size: float = 4.0,
    frame_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip boxes to ``window`` and keep those mostly inside it.

    A box is kept if at least ``min_visible`` of its area (measured against its
    area inside the frame, when ``frame_size=(w, h)`` is given) lies in the
    window and the clipped box is at least ``min_size`` px in both directions.
    Returns ``(boxes_xyxy_in_tile float32 (M, 4), kept_indices int (M,))``.
    """
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    x0, y0, x1, y1 = window
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    if frame_size is not None:
        fw, fh = frame_size
        ref_w = np.clip(bx2, 0, fw) - np.clip(bx1, 0, fw)
        ref_h = np.clip(by2, 0, fh) - np.clip(by1, 0, fh)
    else:
        ref_w, ref_h = b[:, 2], b[:, 3]
    ref_area = np.clip(ref_w, 0, None) * np.clip(ref_h, 0, None)

    cx1, cx2 = np.clip(bx1, x0, x1), np.clip(bx2, x0, x1)
    cy1, cy2 = np.clip(by1, y0, y1), np.clip(by2, y0, y1)
    cw, ch = cx2 - cx1, cy2 - cy1
    visible = np.divide(cw * ch, ref_area, out=np.zeros_like(ref_area), where=ref_area > 0)

    keep = (visible >= min_visible) & (cw >= min_size) & (ch >= min_size)
    out = np.stack([cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0], axis=1)[keep]
    return out.astype(np.float32), np.flatnonzero(keep)
