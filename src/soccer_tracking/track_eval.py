"""Detector-only vs IoU tracker on fixed contiguous segments (the Phase I controlled comparison).

Protocol (frozen in ``docs/phase1_proposal.md`` before any tracking result existed):
* Segments: 3 per clip, 250 consecutive frames (10 s at 25 fps), starting at frames 500, 2500, 4500.
* Both arms use the *same* saved detections, kept at score >= 0.5. Detector-only outputs them as they
  are; the tracker (``tracker.py``) adds association, ids and short-gap coasting.
* The tracker's settings are chosen on the validation clip only (``tune``): among settings whose F1 is
  within 0.02 of detector-only, the lowest fragmentation per track wins (ties: fewer ID switches, then
  smaller ``max_age``). The test clip is scored once (``evaluate --allow-test``) with those settings.
* Success: fragmentation per track reduced by >= 25% (relative) and F1 not lower by more than 0.02.

Commands (run from the repository root with ``PYTHONPATH=src``)::

    python -m soccer_tracking.track_eval detect   --split val --weights runs/finetune/best.pt   # GPU
    python -m soccer_tracking.track_eval tune     --split val
    python -m soccer_tracking.track_eval evaluate --split val
    python -m soccer_tracking.track_eval detect   --split test --weights runs/finetune/best.pt --allow-test
    python -m soccer_tracking.track_eval evaluate --split test --allow-test                     # once
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .data import DEFAULT_MANIFEST, Clip, ManifestError, _sha256, check_annotations, load_manifest
from .evaluate import _git
from .mot import MotFormatError, load_mot
from .track_metrics import score_sequence, summarize
from .tracker import IouTracker

SEGMENT_STARTS = (500, 2500, 4500)
SEGMENT_LEN = 250
DEFAULT_SCORE = 0.5
DEFAULT_DET_DIR = Path("data") / "processed" / "detections" / "finetune"
GRID = {"iou_thr": (0.2, 0.3, 0.4, 0.5), "max_age": (1, 2, 5, 10), "min_hits": (1, 2), "use_velocity": (False, True)}


# --------------------------------------------------------------------------- data
def det_path(det_dir: Path, clip_id: str, start: int) -> Path:
    return Path(det_dir) / f"{clip_id}_{start}.npz"


def load_detections(path: Path, n_frames: int):
    """Per-frame ``(boxes, scores)`` lists from a saved segment file."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"detections not found: {path} (run the `detect` command first)")
    z = np.load(path)
    if int(z["n_frames"]) != n_frames:
        raise ValueError(f"{path}: file has {int(z['n_frames'])} frames, expected {n_frames}")
    frame, boxes, scores = z["frame"], z["boxes"], z["scores"]
    order = np.argsort(frame, kind="stable")
    frame, boxes, scores = frame[order], boxes[order], scores[order]
    edges = np.searchsorted(frame, np.arange(n_frames + 1))
    return [(boxes[edges[i]:edges[i + 1]], scores[edges[i]:edges[i + 1]]) for i in range(n_frames)]


def gt_sequence(clip: Clip, annotations, start: int, n: int):
    seq = []
    for f in range(start, start + n):
        dets = annotations[f]
        ids = np.array([d.track_id for d in dets], dtype=np.int64)
        b = np.array([[d.x, d.y, d.x + d.width, d.y + d.height] for d in dets], dtype=np.float64)
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, clip.width)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, clip.height)
        seq.append((ids, b))
    return seq


def load_segments(clips: Sequence[Clip], det_dir: Path, starts: Sequence[int], seg_len: int) -> List[dict]:
    segments = []
    for clip in clips:
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ManifestError("; ".join(problems))
        for s in starts:
            if s + seg_len > clip.frames:
                raise ValueError(f"clip {clip.clip_id}: segment {s}..{s + seg_len - 1} exceeds its {clip.frames} frames")
            segments.append({"clip_id": clip.clip_id, "start": s, "gt": gt_sequence(clip, ann, s, seg_len),
                             "dets": load_detections(det_path(det_dir, clip.clip_id, s), seg_len)})
    return segments


# --------------------------------------------------------------------------- arms
def score_detector_only(segments: Sequence[dict], score: float) -> dict:
    parts = [score_sequence(seg["gt"], [(None, b[s >= score]) for b, s in seg["dets"]]) for seg in segments]
    return summarize(parts)


def score_tracker(segments: Sequence[dict], score: float, params: dict) -> dict:
    parts = []
    for seg in segments:
        tracker = IouTracker(**params)
        preds = []
        for b, s in seg["dets"]:
            keep = s >= score
            ids, boxes, _, _ = tracker.update(b[keep], s[keep])
            preds.append((ids, boxes))
        parts.append(score_sequence(seg["gt"], preds))
    return summarize(parts)


def per_segment(segments: Sequence[dict], score: float, params: dict) -> List[dict]:
    """Detector-only and tracker measures for each segment separately (to see whether segments agree)."""
    return [{"clip_id": seg["clip_id"], "start": seg["start"],
             "detector_only": score_detector_only([seg], score), "tracker": score_tracker([seg], score, params)}
            for seg in segments]


def _segment_lines(rows: Sequence[dict]) -> str:
    lines = ["per segment (gaps = interruptions; detector-only -> tracker):"]
    for r in rows:
        d, t = r["detector_only"], r["tracker"]
        lines.append(f"  {r['clip_id']} @ {r['start']:>4}: gaps {d['gaps']} -> {t['gaps']}   F1 {d['f1']:.3f} -> {t['f1']:.3f}   "
                     f"IDSW {t['idsw']}   ids/track {t['ids_per_track']:.2f}")
    fewer = sum(r["tracker"]["gaps"] < r["detector_only"]["gaps"] for r in rows)
    same = sum(r["tracker"]["gaps"] == r["detector_only"]["gaps"] for r in rows)
    lines.append(f"  tracker has fewer gaps in {fewer} of {len(rows)} segments (equal in {same})")
    return "\n".join(lines)


def choose_params(det: dict, results: List[dict], max_f1_drop: float) -> dict:
    """Pre-specified selection rule (see module docstring)."""
    ok = [r for r in results if r["summary"]["f1"] >= det["f1"] - max_f1_drop]
    if ok:
        best = min(ok, key=lambda r: (r["summary"]["frag_per_track"], r["summary"]["idsw"], r["params"]["max_age"]))
        return {"params": best["params"], "constraint_met": True}
    best = max(results, key=lambda r: r["summary"]["f1"])
    return {"params": best["params"], "constraint_met": False}


def criteria(det: dict, trk: dict, min_frag_reduction: float, max_f1_drop: float) -> dict:
    base = det["frag_per_track"]
    reduction = (1 - trk["frag_per_track"] / base) if base > 0 else None
    drop = det["f1"] - trk["f1"]
    return {"frag_reduction_relative": reduction, "f1_drop": drop,
            "min_frag_reduction": min_frag_reduction, "max_f1_drop": max_f1_drop,
            "fragmentation_ok": reduction is not None and reduction >= min_frag_reduction,
            "f1_ok": drop <= max_f1_drop}


def _env(manifest: Path) -> dict:
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform(), "git_revision": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")), "manifest_sha256": _sha256(Path(manifest)),
            "command": " ".join(sys.argv)}


def _fmt(s: dict) -> str:
    ids = "" if s["idsw"] is None else f"  IDSW {s['idsw']}  ids/track {s['ids_per_track']:.2f}  MOTA {s['mota']:.3f}"
    return (f"P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f}  MODA {s['moda']:.3f}  "
            f"gaps {s['gaps']} (frag/track {s['frag_per_track']:.3f}){ids}")


def _guard(args) -> None:
    if args.split == "test" and not args.allow_test:
        raise ValueError("the test split is held out: pass --allow-test only for the single final run")


# --------------------------------------------------------------------------- commands
def cmd_detect(args) -> None:
    _guard(args)
    import torch

    from .detect import build_retinanet, detect_tiles, merge_tile_detections
    from .evaluate import environment_record
    from .tiles import tile_grid

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    clips = load_manifest(args.manifest, [args.split])
    model = build_retinanet(args.weights, tuple(args.tile), args.score_floor, 0.5).to(device)
    out_dir = Path(args.det_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        windows = tile_grid(clip.width, clip.height, args.tile[0], args.tile[1], args.overlap)
        for start in args.starts:
            if start + args.seg_len > clip.frames:
                raise ValueError(f"clip {clip.clip_id}: segment {start}..{start + args.seg_len - 1} exceeds {clip.frames} frames")
            cap = cv2.VideoCapture(str(clip.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV could not open video: {clip.video_path}")
            frames, boxes_all, scores_all = [], [], []
            t0 = time.time()
            try:
                for i in range(start):                      # sequential skip: frame-exact
                    if not cap.grab():
                        raise RuntimeError(f"decoding stopped at frame {i}")
                for k in range(args.seg_len):
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        raise RuntimeError(f"could not decode frame {start + k}")
                    if bgr.shape[1] != clip.width or bgr.shape[0] != clip.height:
                        raise ValueError(f"clip {clip.clip_id}: frame size differs from manifest")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    b, s = merge_tile_detections(detect_tiles(model, rgb, windows, device, args.batch_size), windows,
                                                 (clip.width, clip.height), args.merge_iou, args.drop_edge_px)
                    frames.append(np.full(len(b), k, dtype=np.int32)); boxes_all.append(b); scores_all.append(s)
                    if (k + 1) % 50 == 0:
                        print(f"clip {clip.clip_id} start {start}: {k + 1}/{args.seg_len} frames, {(time.time() - t0) / (k + 1):.2f} s/frame", flush=True)
            finally:
                cap.release()
            np.savez_compressed(det_path(out_dir, clip.clip_id, start), start=start, n_frames=args.seg_len,
                                frame=np.concatenate(frames), boxes=np.concatenate(boxes_all).astype(np.float32),
                                scores=np.concatenate(scores_all).astype(np.float32))
    (out_dir / f"detect_config_{args.split}.json").write_text(json.dumps(
        {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "environment": environment_record(args.manifest, device)}, indent=2) + "\n", encoding="utf-8")
    print(f"saved detections for {len(clips)} clip(s) x {len(args.starts)} segment(s) -> {out_dir}")


def cmd_tune(args) -> None:
    if args.split == "test":
        raise ValueError("the tracker must be tuned on the validation clip, never on the test split")
    segments = load_segments(load_manifest(args.manifest, [args.split]), args.det_dir, args.starts, args.seg_len)
    det = score_detector_only(segments, args.score)
    results = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        results.append({"params": params, "summary": score_tracker(segments, args.score, params)})
    pick = choose_params(det, results, args.max_f1_drop)
    rows = per_segment(segments, args.score, pick["params"])
    record = {"protocol": {"starts": list(args.starts), "seg_len": args.seg_len, "score": args.score,
                           "max_f1_drop": args.max_f1_drop, "grid": {k: list(v) for k, v in GRID.items()}},
              "split": args.split, "detector_only": det, "chosen": pick["params"],
              "constraint_met": pick["constraint_met"], "per_segment": rows, "grid_results": results,
              "environment": _env(args.manifest)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    best = next(r for r in results if r["params"] == pick["params"])
    print(f"detector-only: {_fmt(det)}")
    print(f"chosen {pick['params']} (F1 constraint met: {pick['constraint_met']}):\n  tracker: {_fmt(best['summary'])}")
    print(_segment_lines(rows) + f"\n-> {out}")


def cmd_evaluate(args) -> None:
    _guard(args)
    tuning = json.loads(Path(args.params).read_text(encoding="utf-8"))
    params = tuning["chosen"]
    segments = load_segments(load_manifest(args.manifest, [args.split]), args.det_dir, args.starts, args.seg_len)
    det, trk = score_detector_only(segments, args.score), score_tracker(segments, args.score, params)
    crit = criteria(det, trk, args.min_frag_reduction, args.max_f1_drop)
    rows = per_segment(segments, args.score, params)
    record = {"split": args.split, "protocol": {"starts": list(args.starts), "seg_len": args.seg_len, "score": args.score},
              "tracker_params": params, "params_file": str(args.params), "detector_only": det, "tracker": trk,
              "criteria": crit, "per_segment": rows, "environment": _env(args.manifest)}
    out = Path(args.out or Path("results") / f"tracking_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    red = crit["frag_reduction_relative"]
    print(f"detector-only: {_fmt(det)}\ntracker {params}:\n  {_fmt(trk)}")
    print(_segment_lines(rows))
    print(f"fragmentation reduction {'n/a' if red is None else f'{red:.1%}'} (need >= {args.min_frag_reduction:.0%}): "
          f"{'PASS' if crit['fragmentation_ok'] else 'FAIL'};  F1 drop {crit['f1_drop']:+.3f} (allowed {args.max_f1_drop}): "
          f"{'PASS' if crit['f1_ok'] else 'FAIL'}  -> {out}")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detector-only vs IoU tracker on fixed contiguous segments.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("detect", "tune", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        s.add_argument("--split", choices=("val", "test"), default="val")
        s.add_argument("--det-dir", type=Path, default=DEFAULT_DET_DIR)
        s.add_argument("--starts", type=int, nargs="+", default=list(SEGMENT_STARTS))
        s.add_argument("--seg-len", type=int, default=SEGMENT_LEN)
        if name != "tune":
            s.add_argument("--allow-test", action="store_true", help="required for the held-out test split")
    d = sub.choices["detect"]
    d.add_argument("--weights", required=True, help="fine-tuned .pt checkpoint (or 'coco')")
    d.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    d.add_argument("--overlap", type=int, default=128)
    d.add_argument("--score-floor", type=float, default=0.05, help="keep all detections above this score")
    d.add_argument("--merge-iou", type=float, default=0.5)
    d.add_argument("--drop-edge-px", type=float, default=2.0)
    d.add_argument("--batch-size", type=int, default=2)
    d.add_argument("--device", default="auto")
    d.add_argument("--seed", type=int, default=0)
    for name in ("tune", "evaluate"):
        s = sub.choices[name]
        s.add_argument("--score", type=float, default=DEFAULT_SCORE, help="detection score threshold shared by both arms")
        s.add_argument("--max-f1-drop", type=float, default=0.02)
    sub.choices["tune"].add_argument("--out", type=Path, default=Path("results") / "tracker_tuning.json")
    e = sub.choices["evaluate"]
    e.add_argument("--params", type=Path, default=Path("results") / "tracker_tuning.json")
    e.add_argument("--min-frag-reduction", type=float, default=0.25)
    e.add_argument("--out", type=Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        {"detect": cmd_detect, "tune": cmd_tune, "evaluate": cmd_evaluate}[args.command](args)
    except (FileNotFoundError, ManifestError, MotFormatError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
