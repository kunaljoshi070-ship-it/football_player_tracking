"""Evaluate a RetinaNet (tiles merged to full frames) on a manifest split and save a JSON record.

Baseline used in Phase I: torchvision's COCO-pretrained RetinaNet used as-is, keeping only the
``person`` class (label 1) as ``player``. Run from the repo root with ``PYTHONPATH=src``::

    python -m soccer_tracking.evaluate --split val --weights coco --stride 250
    python -m soccer_tracking.evaluate --split val --weights runs/finetune/best.pt --stride 125

Frames must already be extracted (``python -m soccer_tracking.data extract``) and ``--stride`` must
be a multiple of the extraction stride. The test split is refused unless ``--allow-test`` is given:
use it once, at the end, never for choosing thresholds or settings.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import (DEFAULT_FRAMES_ROOT, DEFAULT_MANIFEST, Clip, ManifestError, _sha256, check_annotations,
                   frame_path, load_manifest, sample_frames)
from .detect import build_retinanet, detect_tiles, merge_tile_detections
from .metrics import DetectionEvaluator
from .mot import MotFormatError, load_mot
from .tiles import tile_grid


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return "unknown"


def environment_record(manifest: Path, device) -> dict:
    """Versions, hardware, git state and manifest hash, for the reproducibility record."""
    import torch
    import torchvision

    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
        "git_revision": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "manifest_sha256": _sha256(Path(manifest)), "command": " ".join(sys.argv),
    }


def evaluate_model(model, clips: Sequence[Clip], frames_root: Path, stride: int, tile: Tuple[int, int],
                   overlap: int, device, merge_iou: float = 0.5, drop_edge_px: float = 2.0,
                   report_score: float = 0.5, batch_size: int = 2, verbose: bool = True) -> Tuple[dict, List[dict]]:
    """Full-frame evaluation of ``model`` on every ``stride``-th frame of ``clips``.
    Returns ``(results, per_clip)``; ``results`` has AP50/AP75, P/R/F1 at ``report_score`` and timings."""
    evaluator = DetectionEvaluator((0.5, 0.75))
    per_clip: List[dict] = []
    t_all, n_tiles = time.time(), 0
    for clip in clips:
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ManifestError("; ".join(problems))
        windows = tile_grid(clip.width, clip.height, tile[0], tile[1], overlap)
        frames = list(sample_frames(clip.frames, stride))
        missing = [f for f in frames if not frame_path(frames_root, clip.clip_id, f).is_file()]
        if missing:
            raise FileNotFoundError(f"frame {missing[0]} of clip {clip.clip_id} not extracted; the stride must be a "
                                    "multiple of the extraction stride (run `python -m soccer_tracking.data extract`)")
        t_clip = time.time()
        for k, f in enumerate(frames, start=1):
            bgr = cv2.imread(str(frame_path(frames_root, clip.clip_id, f)), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[1] != clip.width or bgr.shape[0] != clip.height:
                raise ValueError(f"clip {clip.clip_id} frame {f}: unreadable or size differs from manifest")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            per_tile = detect_tiles(model, rgb, windows, device, batch_size)
            boxes, scores = merge_tile_detections(per_tile, windows, (clip.width, clip.height), merge_iou, drop_edge_px)
            g = np.array([[d.x, d.y, d.x + d.width, d.y + d.height] for d in ann[f]], dtype=np.float64)
            g[:, [0, 2]] = np.clip(g[:, [0, 2]], 0, clip.width)
            g[:, [1, 3]] = np.clip(g[:, [1, 3]], 0, clip.height)
            evaluator.add_image(g, boxes, scores)
            n_tiles += len(windows)
            if verbose:
                print(f"clip {clip.clip_id}: frame {f} ({k}/{len(frames)}) {len(boxes)} detections, "
                      f"{(time.time() - t_clip) / k:.1f} s/frame", flush=True)
        per_clip.append({"clip_id": clip.clip_id, "frames_evaluated": len(frames), "tiles_per_frame": len(windows)})
    elapsed = time.time() - t_all
    results = evaluator.summary(report_score)
    results["seconds_per_frame"] = elapsed / max(1, evaluator.n_images)
    results["seconds_per_tile"] = elapsed / max(1, n_tiles)
    return results, per_clip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate player detection on a manifest split (full-frame AP, P/R).")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true", help="required to evaluate the held-out test split")
    p.add_argument("--weights", default="coco", help="'coco' (pretrained baseline), 'none' (random init, smoke tests) or a .pt checkpoint from train.py")
    p.add_argument("--stride", type=int, default=250, help="evaluate every Nth frame (multiple of the extraction stride; default 250 is CPU-friendly)")
    p.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--score-thresh", type=float, default=0.05, help="detector score floor (low, so AP sees the full curve)")
    p.add_argument("--nms-thresh", type=float, default=0.5, help="NMS inside the detector, per tile")
    p.add_argument("--merge-iou", type=float, default=0.5, help="NMS across tiles on the full frame")
    p.add_argument("--drop-edge-px", type=float, default=2.0, help="drop boxes cut by an interior tile border; negative disables")
    p.add_argument("--report-score", type=float, default=0.5, help="score at which precision/recall are reported")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="JSON output (default results/baseline_<weights>_<split>.json)")
    return p


def run(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise ValueError("the test split is held out: pass --allow-test only for the single final evaluation")
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    clips = load_manifest(args.manifest, [args.split])
    model = build_retinanet(args.weights, tuple(args.tile), args.score_thresh, args.nms_thresh).to(device)
    results, per_clip = evaluate_model(model, clips, args.frames_root, args.stride, tuple(args.tile), args.overlap,
                                       device, args.merge_iou, args.drop_edge_px, args.report_score, args.batch_size)
    record = {
        "task": "player detection, tiles merged to full frames",
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "label_convention": "label 1 = person (COCO) = player; all other detector classes discarded",
        "clips": per_clip, "results": results, "environment": environment_record(args.manifest, device),
    }
    name = Path(args.weights).stem if Path(args.weights).suffix in (".pt", ".pth") else args.weights
    out = args.out or Path("results") / f"baseline_{name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nAP50 {results.get('ap50', float('nan')):.4f}  AP75 {results.get('ap75', float('nan')):.4f}  "
          f"P {results.get('precision', float('nan')):.3f}  R {results.get('recall', float('nan')):.3f} "
          f"(score >= {args.report_score}, IoU 0.5)  frames {results['images']}  -> {out}")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ManifestError, MotFormatError, RuntimeError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
