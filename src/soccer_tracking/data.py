"""Phase I detector data pipeline: manifest -> extracted frames -> native-scale tiles.

Command line (run from the repository root, ``PYTHONPATH=src``)::

    python -m soccer_tracking.data audit   [--splits train val test] [--verify-hashes]
    python -m soccer_tracking.data extract [--splits train val test] [--stride 25]

Conventions (all explicit, nothing inferred silently)
-----------------------------------------------------
* ``data/manifest.csv`` is the source of truth for clips and splits. Paths in it
  are used as written (relative to the current directory = repo root).
* Annotation frame numbers are 0-based and equal video frame indices (all ten
  clips were measured this way; see ``docs/dataset_card.md``).
* Frames are sampled with ``range(start, end, stride)`` and decoded sequentially
  (frame-exact), then stored as JPEG under ``data/processed/frames/<clip_id>/``
  (git-ignored). One class: ``player`` (label 1; 0 is background by torchvision convention).
* Splits are whole matches. A clip id may appear in only one split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .mot import Detection, MotFormatError, load_mot
from .tiles import Window, boxes_to_tile, tile_grid

SPLITS = ("train", "val", "test", "unused")
DEFAULT_MANIFEST = Path("data") / "manifest.csv"
DEFAULT_FRAMES_ROOT = Path("data") / "processed" / "frames"
# torchvision's RetinaNet default input resize, used only to quantify what resizing would do.
DETECTOR_MIN_SIZE, DETECTOR_MAX_SIZE = 800, 1333


class ManifestError(ValueError):
    """Raised when the manifest is unreadable or fails its checks."""


@dataclass(frozen=True)
class Clip:
    clip_id: str
    split: str
    video_path: Path
    annotation_path: Path
    width: int
    height: int
    fps: float
    frames: int
    sha256: str = ""
    annotation_sha256: str = ""


# --------------------------------------------------------------------------- manifest
def load_manifest(path: Path = DEFAULT_MANIFEST, splits: Optional[Sequence[str]] = None) -> List[Clip]:
    """Read the manifest, optionally keeping only the given splits. Raises ``ManifestError``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    clips: List[Clip] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"clip_id", "split", "local_path", "annotation_path", "width", "height", "fps", "frames"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(f"{path}: missing columns {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                clips.append(Clip(
                    clip_id=row["clip_id"].strip(), split=row["split"].strip(),
                    video_path=Path(row["local_path"].strip()),
                    annotation_path=Path(row["annotation_path"].strip()),
                    width=int(row["width"]), height=int(row["height"]),
                    fps=float(row["fps"]), frames=int(row["frames"]),
                    sha256=(row.get("sha256") or "").strip().lower(),
                    annotation_sha256=(row.get("annotation_sha256") or "").strip().lower(),
                ))
            except (ValueError, KeyError) as exc:
                raise ManifestError(f"{path}:{line_no}: bad row ({exc})") from None
    problems = check_manifest(clips)
    if problems:
        raise ManifestError(f"{path}: " + "; ".join(problems))
    if splits is not None:
        unknown = set(splits) - set(SPLITS)
        if unknown:
            raise ManifestError(f"unknown split(s) requested: {sorted(unknown)}; valid: {SPLITS}")
        clips = [c for c in clips if c.split in splits]
        if not clips:
            raise ManifestError(f"no clips in split(s) {list(splits)}")
    return clips


def check_manifest(clips: Sequence[Clip]) -> List[str]:
    """Leakage and sanity checks on the manifest itself (no files are opened)."""
    problems: List[str] = []
    seen: Dict[str, str] = {}
    for c in clips:
        if c.split not in SPLITS:
            problems.append(f"clip {c.clip_id}: split {c.split!r} not in {SPLITS}")
        if c.clip_id in seen:
            problems.append(f"clip {c.clip_id} listed more than once ({seen[c.clip_id]} and {c.split}): "
                            "a match must belong to exactly one split")
        seen[c.clip_id] = c.split
        if min(c.width, c.height, c.frames) <= 0 or c.fps <= 0:
            problems.append(f"clip {c.clip_id}: width/height/fps/frames must be positive")
    return problems


def check_annotations(clip: Clip, annotations: Dict[int, List[Detection]]) -> List[str]:
    """Annotations must be 0-based and cover exactly the video's frames."""
    first, last = min(annotations), max(annotations)
    problems = []
    if first != 0 or last != clip.frames - 1:
        problems.append(f"clip {clip.clip_id}: annotation frames {first}..{last} do not match "
                        f"0..{clip.frames - 1} (manifest frame count); check numbering")
    return problems


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_hashes(clips: Sequence[Clip]) -> List[str]:
    """Compare file hashes against the manifest (reads every file fully)."""
    problems = []
    for c in clips:
        for label, path, expected in (("video", c.video_path, c.sha256),
                                      ("annotation", c.annotation_path, c.annotation_sha256)):
            if not path.is_file():
                problems.append(f"clip {c.clip_id}: {label} file missing: {path}")
            elif not expected:
                problems.append(f"clip {c.clip_id}: no {label} hash in manifest")
            elif _sha256(path) != expected:
                problems.append(f"clip {c.clip_id}: {label} hash does not match manifest ({path})")
    return problems


# --------------------------------------------------------------------------- frames
def sample_frames(n_frames: int, stride: int, start: int = 0, end: Optional[int] = None) -> range:
    """Frame indices ``range(start, end, stride)`` within ``[0, n_frames)``."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    end = n_frames if end is None else min(end, n_frames)
    if not 0 <= start < end:
        raise ValueError(f"need 0 <= start < end <= {n_frames}, got start={start}, end={end}")
    return range(start, end, stride)


def frame_path(frames_root: Path, clip_id: str, frame: int) -> Path:
    return Path(frames_root) / clip_id / f"{frame:06d}.jpg"


def extract_frames(clip: Clip, out_root: Path = DEFAULT_FRAMES_ROOT, stride: int = 25, start: int = 0,
                   end: Optional[int] = None, jpeg_quality: int = 95, overwrite: bool = False) -> int:
    """Decode the sampled frames sequentially and save them as JPEG. Returns the number written.

    Existing files are kept unless ``overwrite``. Writes ``extract_log.json`` next to the frames.
    """
    wanted = sample_frames(clip.frames, stride, start, end)
    out_dir = Path(out_root) / clip.clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in wanted if overwrite or not frame_path(out_root, clip.clip_id, f).is_file()]
    written = 0
    if todo:
        if not clip.video_path.is_file():
            raise FileNotFoundError(f"video file not found: {clip.video_path}")
        cap = cv2.VideoCapture(str(clip.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {clip.video_path}")
        try:
            reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if reported and reported != clip.frames:
                raise ValueError(f"clip {clip.clip_id}: video reports {reported} frames, manifest says {clip.frames}")
            todo_set = set(todo)
            for i in range(max(todo) + 1):
                if i not in todo_set:
                    if not cap.grab():
                        raise RuntimeError(f"clip {clip.clip_id}: decoding stopped at frame {i}")
                    continue
                ok, img = cap.read()
                if not ok or img is None:
                    raise RuntimeError(f"clip {clip.clip_id}: could not decode frame {i}")
                if img.shape[1] != clip.width or img.shape[0] != clip.height:
                    raise ValueError(f"clip {clip.clip_id}: frame is {img.shape[1]}x{img.shape[0]}, "
                                     f"manifest says {clip.width}x{clip.height}")
                if not cv2.imwrite(str(frame_path(out_root, clip.clip_id, i)), img,
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise RuntimeError(f"could not write {frame_path(out_root, clip.clip_id, i)}")
                written += 1
        finally:
            cap.release()
    log = {"clip_id": clip.clip_id, "video_sha256": clip.sha256, "stride": stride, "start": start,
           "end": end, "jpeg_quality": jpeg_quality, "frames_in_sample": len(wanted),
           "opencv": cv2.__version__, "last_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out_dir / "extract_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return written


# --------------------------------------------------------------------------- dataset
class SoccerTilesDataset:
    """Map-style dataset of native-scale tiles with player boxes (works with ``torch.utils.data.DataLoader``).

    Item ``i`` is ``(image, target)``: ``image`` float32 CHW RGB in [0, 1]; ``target`` has
    ``boxes`` (N, 4) float32 xyxy in tile pixels, ``labels`` (N,) int64 all 1, and ``image_id``.
    Use :meth:`meta` to map tile predictions back to full-frame coordinates.

    Reproducibility: tiles are enumerated in a fixed order (clip, frame, tile). With ``augment=True``
    a horizontal flip is applied with p=0.5, decided by ``default_rng([seed, epoch, index])``, so it is
    identical for any worker count; call :meth:`set_epoch` each epoch to vary it.
    Frames must already be extracted (see ``extract_frames``). With shuffling, each item decodes a full
    frame; the last decoded frame is cached, so ordered access decodes each frame once.
    """

    def __init__(self, clips: Sequence[Clip], frames_root: Path = DEFAULT_FRAMES_ROOT, stride: int = 25,
                 start: int = 0, end: Optional[int] = None, tile: Tuple[int, int] = (1024, 1080),
                 overlap: int = 128, min_visible: float = 0.5, min_size: float = 4.0,
                 augment: bool = False, seed: int = 0):
        self.clips = list(clips)
        self.frames_root = Path(frames_root)
        self.tile, self.overlap = tile, overlap
        self.min_visible, self.min_size = min_visible, min_size
        self.augment, self.seed, self.epoch = augment, seed, 0
        self._annotations: List[Dict[int, np.ndarray]] = []
        self._ids: List[Dict[int, np.ndarray]] = []
        self._windows: List[List[Window]] = []
        self.items: List[Tuple[int, int, int]] = []  # (clip index, frame, tile index)
        missing: List[Path] = []
        for ci, clip in enumerate(self.clips):
            ann = load_mot(clip.annotation_path)
            problems = check_annotations(clip, ann)
            if problems:
                raise ManifestError("; ".join(problems))
            self._annotations.append({f: np.array([[d.x, d.y, d.width, d.height] for d in ds]) for f, ds in ann.items()})
            self._ids.append({f: np.array([d.track_id for d in ds]) for f, ds in ann.items()})
            windows = tile_grid(clip.width, clip.height, tile[0], tile[1], overlap)
            self._windows.append(windows)
            for f in sample_frames(clip.frames, stride, start, end):
                path = frame_path(self.frames_root, clip.clip_id, f)
                if not path.is_file():
                    missing.append(path)
                self.items.extend((ci, f, ti) for ti in range(len(windows)))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} extracted frame(s) missing, first: {missing[0]}. Run "
                "`python -m soccer_tracking.data extract` with the same --stride/--start/--end.")
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache_img: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _frame_image(self, ci: int, frame: int) -> np.ndarray:
        if self._cache_key != (ci, frame):
            clip = self.clips[ci]
            img = cv2.imread(str(frame_path(self.frames_root, clip.clip_id, frame)), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"could not read {frame_path(self.frames_root, clip.clip_id, frame)}")
            if img.shape[1] != clip.width or img.shape[0] != clip.height:
                raise ValueError(f"clip {clip.clip_id} frame {frame}: image is {img.shape[1]}x{img.shape[0]}, "
                                 f"manifest says {clip.width}x{clip.height}")
            self._cache_key, self._cache_img = (ci, frame), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._cache_img  # type: ignore[return-value]

    def _flipped(self, index: int) -> bool:
        return self.augment and np.random.default_rng([self.seed, self.epoch, index]).random() < 0.5

    def get_numpy(self, index: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """``(RGB uint8 HxWx3, boxes float32 (N, 4) xyxy, meta)`` without any torch dependency."""
        ci, frame, ti = self.items[index]
        clip, window = self.clips[ci], self._windows[ci][ti]
        x0, y0, x1, y1 = window
        img = self._frame_image(ci, frame)[y0:y1, x0:x1]
        boxes, kept = boxes_to_tile(self._annotations[ci][frame], window, self.min_visible,
                                    self.min_size, (clip.width, clip.height))
        flipped = self._flipped(index)
        if flipped:
            w = x1 - x0
            img = np.ascontiguousarray(img[:, ::-1])
            boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], axis=1) if len(boxes) else boxes
        meta = {"clip_id": clip.clip_id, "frame": frame, "window": window, "flipped": flipped,
                "track_ids": self._ids[ci][frame][kept]}
        return img, boxes.reshape(-1, 4).astype(np.float32), meta

    def meta(self, index: int) -> dict:
        return self.get_numpy(index)[2]

    def __getitem__(self, index: int):
        import torch  # imported lazily so the rest of this module works without torch

        img, boxes, _ = self.get_numpy(index)
        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        target = {"boxes": torch.from_numpy(boxes).reshape(-1, 4),
                  "labels": torch.ones((len(boxes),), dtype=torch.int64),
                  "image_id": torch.tensor(index, dtype=torch.int64)}
        return image, target


def collate(batch):
    """Detection collate: keep images and targets as tuples (tiles may differ in size)."""
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- audit
def _pct(values: np.ndarray) -> str:
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return f"{p5:.1f} / {p50:.1f} / {p95:.1f}"


def audit(clips: Sequence[Clip], stride: int, tile: Tuple[int, int], overlap: int, min_visible: float,
          min_size: float, out: Callable[[str], None] = print) -> None:
    """Print split sizes, box-size statistics and how well the tiling covers the boxes."""
    by_split: Dict[str, List[Clip]] = {}
    for c in clips:
        by_split.setdefault(c.split, []).append(c)
    for split, cs in by_split.items():
        ws, hs = [], []
        total = covered = full = 0
        n_frames = n_tiles = 0
        for c in cs:
            ann = load_mot(c.annotation_path)
            problems = check_annotations(c, ann)
            if problems:
                raise ManifestError("; ".join(problems))
            arrs = [np.array([[d.x, d.y, d.width, d.height] for d in ds]) for ds in ann.values()]
            allb = np.concatenate(arrs)
            ws.append(allb[:, 2]); hs.append(allb[:, 3])
            windows = tile_grid(c.width, c.height, tile[0], tile[1], overlap)
            n_tiles = len(windows)
            for f in sample_frames(c.frames, stride):
                b = np.asarray([[d.x, d.y, d.width, d.height] for d in ann[f]])
                n_frames += 1
                cov = np.zeros(len(b), bool); fu = np.zeros(len(b), bool)
                for win in windows:
                    cov[boxes_to_tile(b, win, min_visible, min_size, (c.width, c.height))[1]] = True
                    fu[boxes_to_tile(b, win, 0.99, min_size, (c.width, c.height))[1]] = True
                total += len(b); covered += int(cov.sum()); full += int(fu.sum())
        w, h = np.concatenate(ws), np.concatenate(hs)
        scale = min(DETECTOR_MIN_SIZE / min(c.height, c.width), DETECTOR_MAX_SIZE / max(c.height, c.width))
        out(f"[{split}] clips {', '.join(c.clip_id for c in cs)} | boxes {len(w)} | "
            f"frames sampled (stride {stride}) {n_frames} | tiles/frame {n_tiles} (last clip)")
        out(f"  box width  px p5/p50/p95: {_pct(w)}    box height px p5/p50/p95: {_pct(h)}")
        out(f"  boxes with min(w,h) < 16 px: {100 * np.mean(np.minimum(w, h) < 16):.1f}%")
        out(f"  if the full frame were resized to the detector default (scale {scale:.3f}, last clip): "
            f"median box {np.median(w) * scale:.1f} x {np.median(h) * scale:.1f} px")
        out(f"  tiling {tile[0]}x{tile[1]}, overlap {overlap}: boxes kept in >=1 tile "
            f"(>= {min_visible:.0%} visible) {100 * covered / total:.2f}%, fully inside >=1 tile "
            f"{100 * full / total:.2f}%  (checked on the sampled frames)")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase I detector data pipeline utilities.")
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in (("audit", "check the manifest and print box/tiling statistics"),
                       ("extract", "decode sampled frames to JPEG under data/processed/frames")):
        s = sub.add_parser(name, help=text, description=text)
        s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        s.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=SPLITS)
        s.add_argument("--stride", type=int, default=25, help="keep every Nth frame (default 25 = 1 per second at 25 fps)")
    a = sub.choices["audit"]
    a.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    a.add_argument("--overlap", type=int, default=128)
    a.add_argument("--min-visible", type=float, default=0.5)
    a.add_argument("--min-size", type=float, default=4.0)
    a.add_argument("--verify-hashes", action="store_true", help="re-hash every video and label file (slow)")
    e = sub.choices["extract"]
    e.add_argument("--out", type=Path, default=DEFAULT_FRAMES_ROOT)
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--end", type=int, default=None)
    e.add_argument("--jpeg-quality", type=int, default=95)
    e.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        clips = load_manifest(args.manifest, args.splits)
        counts = {s: sum(c.split == s for c in clips) for s in args.splits}
        print(f"manifest {args.manifest}: {counts} (manifest checks passed: unique clip ids, valid splits)")
        if args.command == "audit":
            if args.verify_hashes:
                problems = verify_hashes(clips)
                if problems:
                    raise ManifestError("; ".join(problems))
                print("hashes: all videos and labels match the manifest")
            audit(clips, args.stride, tuple(args.tile), args.overlap, args.min_visible, args.min_size)
        else:
            for c in clips:
                n = extract_frames(c, args.out, args.stride, args.start, args.end, args.jpeg_quality, args.overwrite)
                print(f"clip {c.clip_id} [{c.split}]: wrote {n} new frame(s) -> {args.out / c.clip_id}")
    except (FileNotFoundError, ManifestError, MotFormatError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
