"""Save an image of one video frame with its annotated MOT boxes and IDs.

Frame numbering (read this first)
---------------------------------
* ``--frame`` is a **0-based video frame index**, as used by OpenCV.
* ``--annotation-base`` says what number the annotation file gives the first
  video frame: ``0`` (default) or ``1`` (the usual MOT convention).
  The annotation frame looked up is ``--frame + --annotation-base``.
* The default is 0 because the SoccerTrack MOT file ``117092.txt`` was
  observed to start at frame 0. Nothing is shifted unless you pass 1, and the
  mapping used is always printed.

Example::

    PYTHONPATH=src python -m soccer_tracking.visualize \\
        --video data/raw/117092.mp4 --annotations data/raw/117092.txt --frame 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .mot import Detection, MotFormatError, load_mot

DEFAULT_OUTPUT_DIR = Path("data") / "processed"  # git-ignored; frames must not be committed


def annotation_frame_for(video_frame: int, annotation_base: int) -> int:
    """Map a 0-based video frame index to the frame number used in the annotation file."""
    if annotation_base not in (0, 1):
        raise ValueError(f"annotation_base must be 0 or 1, got {annotation_base}")
    if video_frame < 0:
        raise ValueError(f"video frame index must be >= 0, got {video_frame}")
    return video_frame + annotation_base


def read_video_frame(video_path: Path, index: int, sequential: bool = False) -> Tuple[np.ndarray, int]:
    """Return ``(BGR image, reported_frame_count)`` for the 0-based frame ``index``.

    ``sequential=True`` decodes from the start (slow, but frame-exact). Otherwise
    OpenCV seeks, which can be imprecise for some codecs.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count > 0 and index >= count:
            raise IndexError(
                f"frame {index} is out of range: video reports {count} frames "
                f"(valid 0-based indices 0..{count - 1})"
            )
        if sequential:
            ok, frame = False, None
            for _ in range(index + 1):
                ok, frame = cap.read()
                if not ok:
                    break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not decode frame {index} from {video_path}")
        return frame, count
    finally:
        cap.release()


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic, reasonably distinct BGR colour per track ID."""
    hsv = np.uint8([[[(track_id * 47) % 180, 230, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_boxes(image: np.ndarray, detections: Iterable[Detection]) -> Tuple[np.ndarray, int, int]:
    """Draw boxes and ID labels on a copy of ``image``.

    Returns ``(annotated, n_drawn, n_clipped)``. Boxes partly outside the image
    are clipped to it (and counted in ``n_clipped``); boxes fully outside are skipped.
    """
    out = image.copy()
    height, width = out.shape[:2]
    thickness = max(1, round(width / 1920))
    font_scale = max(0.4, width / 3840 * 0.9)
    n_drawn = n_clipped = 0
    for det in detections:
        x1, y1 = round(det.x), round(det.y)
        x2, y2 = round(det.x + det.width), round(det.y + det.height)
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(width - 1, x2), min(height - 1, y2)
        if cx1 >= cx2 or cy1 >= cy2:
            n_clipped += 1
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            n_clipped += 1
        color = _color_for_id(det.track_id)
        cv2.rectangle(out, (cx1, cy1), (cx2, cy2), color, thickness)
        label = str(det.track_id)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = cy1 - 3 if cy1 - th - base - 3 >= 0 else cy2 + th + 3  # above box, else below
        tx = min(cx1, max(0, width - tw))
        cv2.rectangle(out, (tx, ty - th - base // 2), (tx + tw, ty + base // 2), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    thickness, cv2.LINE_AA)
        n_drawn += 1
    return out, n_drawn, n_clipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Save one video frame with its MOT boxes and track IDs drawn on it.",
        epilog="--frame is a 0-based video frame index. See --annotation-base.",
    )
    p.add_argument("--video", required=True, type=Path, help="local video file (e.g. data/raw/117092.mp4)")
    p.add_argument("--annotations", required=True, type=Path, help="matching MOT file (e.g. data/raw/117092.txt)")
    p.add_argument("--frame", required=True, type=int, help="0-based video frame index to visualize")
    p.add_argument("--annotation-base", type=int, choices=(0, 1), default=0,
                   help="frame number the annotation file uses for the FIRST video frame "
                        "(default 0, as observed in 117092.txt; use 1 for 1-based MOT files)")
    p.add_argument("--output", type=Path, default=None,
                   help=f"output image path (default: {DEFAULT_OUTPUT_DIR}/<video>_frame<N>.jpg)")
    p.add_argument("--sequential", action="store_true",
                   help="decode from the start instead of seeking: slow but frame-exact")
    return p


def run(args: argparse.Namespace) -> Path:
    annotations = load_mot(args.annotations)
    first, last = min(annotations), max(annotations)
    ann_frame = annotation_frame_for(args.frame, args.annotation_base)
    print(f"annotations: {sum(map(len, annotations.values()))} rows, "
          f"frames {first}..{last} as written in file")
    print(f"frame mapping: video frame {args.frame} (0-based) -> annotation frame {ann_frame} "
          f"(annotation base {args.annotation_base})")
    if ann_frame not in annotations:
        raise KeyError(
            f"no annotations for annotation frame {ann_frame}. The file covers {first}..{last}; "
            f"check --frame and --annotation-base (this mapping assumes the file's first frame "
            f"is {args.annotation_base})."
        )

    image, video_count = read_video_frame(args.video, args.frame, args.sequential)
    height, width = image.shape[:2]
    print(f"video: {width}x{height}, reports {video_count} frames")
    if video_count > 0:
        implied_last = last - args.annotation_base
        if implied_last >= video_count:
            print(f"WARNING: annotations reach video frame {implied_last} but the video reports "
                  f"only {video_count} frames; check that the video and annotations match.",
                  file=sys.stderr)

    annotated, n_drawn, n_clipped = draw_boxes(image, annotations[ann_frame])
    n_total = len(annotations[ann_frame])
    if n_clipped:
        print(f"WARNING: {n_clipped} of {n_total} boxes lie partly/fully outside the "
              f"{width}x{height} frame (clipped or skipped).", file=sys.stderr)

    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.video.stem}_frame{args.frame}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise RuntimeError(f"could not write image: {output}")
    print(f"drew {n_drawn}/{n_total} boxes -> {output}")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, MotFormatError, KeyError, IndexError, RuntimeError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
