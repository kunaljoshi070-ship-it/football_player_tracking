"""Read and validate MOT-format annotation files.

Each non-blank line is ``frame,id,x,y,width,height[,conf,...]`` where
``(x, y)`` is the top-left corner of the box in image pixels.

Frame numbers are returned exactly as written in the file. This module does not
decide whether they are 0-based or 1-based; see ``visualize.py`` for the
explicit conversion to video frame indices.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


class MotFormatError(ValueError):
    """Raised when an annotation row is malformed or has an invalid box."""


@dataclass(frozen=True)
class Detection:
    """One annotated box. ``frame`` is the number as written in the file."""

    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    # 7th column if present. Its meaning depends on the file (detection score,
    # or an "evaluate" flag in MOT ground truth); it is stored, never interpreted.
    conf: Optional[float] = None


def _parse_int(text: str, name: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {text!r}") from None


def _parse_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {text!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {text!r}")
    return value


def parse_mot_line(line: str) -> Detection:
    """Parse and validate one MOT row. Raises ``ValueError`` describing the problem."""
    fields = [f.strip() for f in line.strip().split(",")]
    if len(fields) < 6:
        raise ValueError(
            f"expected at least 6 comma-separated fields "
            f"(frame,id,x,y,width,height), got {len(fields)}"
        )
    frame = _parse_int(fields[0], "frame")
    if frame < 0:
        raise ValueError(f"frame must be >= 0, got {frame}")
    track_id = _parse_int(fields[1], "id")
    x = _parse_float(fields[2], "x")
    y = _parse_float(fields[3], "y")
    width = _parse_float(fields[4], "width")
    height = _parse_float(fields[5], "height")
    if width <= 0 or height <= 0:
        raise ValueError(f"box must have positive width and height, got {width} x {height}")
    conf = _parse_float(fields[6], "conf") if len(fields) > 6 and fields[6] != "" else None
    return Detection(frame, track_id, x, y, width, height, conf)


def load_mot(path: Union[str, Path]) -> Dict[int, List[Detection]]:
    """Load a MOT file into ``{frame_as_written: [Detection, ...]}``.

    Raises ``FileNotFoundError`` for a missing file and ``MotFormatError`` (with
    the line number) for malformed rows, invalid boxes, a repeated (frame, id)
    pair, or a file with no rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MOT annotation file not found: {path}")

    by_frame: Dict[int, List[Detection]] = defaultdict(list)
    seen = set()
    # utf-8-sig tolerates a BOM; text mode normalises CRLF line endings.
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                det = parse_mot_line(line)
            except ValueError as exc:
                raise MotFormatError(f"{path}:{line_no}: {exc}") from None
            key = (det.frame, det.track_id)
            if key in seen:
                raise MotFormatError(
                    f"{path}:{line_no}: duplicate (frame, id) pair {key}"
                )
            seen.add(key)
            by_frame[det.frame].append(det)

    if not by_frame:
        raise MotFormatError(f"{path}: no annotation rows found")
    return dict(sorted(by_frame.items()))
