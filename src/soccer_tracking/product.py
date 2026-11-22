"""Product CLI Core (Phase II, WS-E): Football Tracker.

Supports two execution modes:
1. ``cached`` (instant, CPU-only, ideal for interactive review and live viva demos):
   Loads precomputed detection chunks (and optional ReID/Lab feature chunks) for a clip segment,
   runs Tracker v3 with user-chosen thresholds, and exports all product outputs.
2. ``live`` (GPU recommended, ~0.35s/frame on T4):
   Validates an input video clip and segment bounds, runs P2-more tiled RetinaNet (FP16) +
   optional fine-tuned OSNet + Tracker v3, and exports all product outputs.

Outputs written to ``--out-dir``:
- ``annotated.mp4``      Browser-playable H.264 annotated video with track IDs, scores, flags, and banner
- ``preview_frame.png``  Crisp annotated middle keyframe preview
- ``trajectories.csv``   Frame-by-frame trajectories in original pixel coordinates
- ``heatmap.png``        Pitch occupancy heat map of player footpoints
- ``summary.json``       Run metadata, track counts, confidence breakdown, and interpretation notice
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .metrics import iou_matrix
from .score_final_test import FROZEN_FINAL_PARAMS
from .tracker_v3 import run_tracker_v3

ALLOWED_CONTAINERS = {".mp4", ".mov", ".avi", ".mkv"}
DISCLAIMER_TEXT = (
    "Image-plane track IDs only (not verified player identities; full-clip val AssA=0.322). "
    "Offline tactical review only."
)


class InputValidationError(ValueError):
    """Raised when an input video or segment fails product validation."""


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    start_frame: int
    end_frame: int


def validate_video_input(
    video_path: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_segment_frames: Optional[int] = 300,
    min_res: Tuple[int, int] = (320, 180),
    max_res: Tuple[int, int] = (4096, 2160),
) -> VideoMetadata:
    """Validate container, readability, resolution, FPS, frame count, and segment bounds."""
    path = Path(video_path)
    if not path.is_file():
        raise InputValidationError(f"Video file not found: {path}")
    if path.suffix.lower() not in ALLOWED_CONTAINERS:
        raise InputValidationError(
            f"Unsupported container '{path.suffix}'; supported formats: {sorted(ALLOWED_CONTAINERS)}"
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputValidationError(f"Could not open video file (corrupt or unsupported codec): {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if width < min_res[0] or height < min_res[1] or width > max_res[0] or height > max_res[1]:
        raise InputValidationError(
            f"Resolution {width}x{height} outside supported bounds "
            f"[{min_res[0]}x{min_res[1]} .. {max_res[0]}x{max_res[1]}]."
        )
    if not (5.0 <= fps <= 120.0):
        raise InputValidationError(f"Invalid FPS ({fps:.2f}); expected between 5 and 120 fps.")
    if total <= 0:
        raise InputValidationError(f"Video reports {total} frames; cannot process empty clip.")

    eff_end = total if end_frame is None else int(end_frame)
    if start_frame < 0 or eff_end <= start_frame or eff_end > total:
        raise InputValidationError(
            f"Invalid segment [{start_frame}, {eff_end}) for video with {total} frames."
        )
    seg_len = eff_end - start_frame
    if max_segment_frames is not None and seg_len > max_segment_frames:
        raise InputValidationError(
            f"Segment length {seg_len} frames exceeds limit of {max_segment_frames} frames "
            f"(increase --max-frames or shorten --start/--end)."
        )
    return VideoMetadata(path=path, width=width, height=height, fps=fps, total_frames=total,
                         start_frame=int(start_frame), end_frame=eff_end)


def compute_track_annotations(
    preds: Sequence[Tuple[np.ndarray, np.ndarray]],
    scores: Sequence[np.ndarray],
    det_high: float = 0.78,
    crowd_iou: float = 0.15,
) -> Tuple[List[List[dict]], dict]:
    """Compute per-detection rows with track confidence tiers and diagnostic flags."""
    lengths: Dict[int, int] = {}
    score_sums: Dict[int, float] = {}
    for (ids, _), sc in zip(preds, scores):
        for tid, s in zip(ids.tolist(), sc.tolist()):
            lengths[tid] = lengths.get(tid, 0) + 1
            score_sums[tid] = score_sums.get(tid, 0.0) + float(s)

    track_tier: Dict[int, str] = {}
    for tid, length in lengths.items():
        mean_sc = score_sums[tid] / max(1, length)
        if length >= 30 and mean_sc >= 0.85:
            track_tier[tid] = "high"
        elif length >= 12 and mean_sc >= 0.70:
            track_tier[tid] = "medium"
        else:
            track_tier[tid] = "low"

    per_frame_rows: List[List[dict]] = []
    crowded_count = 0
    low_score_count = 0
    for f_idx, ((ids, boxes), sc) in enumerate(zip(preds, scores)):
        n = len(ids)
        crowded = np.zeros(n, dtype=bool)
        if n > 1:
            mat = iou_matrix(boxes, boxes)
            np.fill_diagonal(mat, 0.0)
            crowded = mat.max(axis=1) >= crowd_iou
        rows_f = []
        for i in range(n):
            tid = int(ids[i])
            s = float(sc[i])
            flags = []
            if crowded[i]:
                flags.append("crowded")
                crowded_count += 1
            if s < det_high:
                flags.append("low_score")
                low_score_count += 1
            if lengths[tid] < 25:
                flags.append("short_track")
            rows_f.append({
                "frame": f_idx,
                "track_id": tid,
                "x1": float(boxes[i, 0]),
                "y1": float(boxes[i, 1]),
                "x2": float(boxes[i, 2]),
                "y2": float(boxes[i, 3]),
                "score": s,
                "track_confidence": track_tier[tid],
                "track_length": lengths[tid],
                "flags": "|".join(flags),
            })
        per_frame_rows.append(rows_f)

    tier_counts = {k: sum(1 for v in track_tier.values() if v == k) for k in ("high", "medium", "low")}
    stats = {
        "unique_tracks": len(lengths),
        "total_detections": int(sum(len(r) for r in per_frame_rows)),
        "mean_tracks_per_frame": float(np.mean([len(r) for r in per_frame_rows])) if per_frame_rows else 0.0,
        "track_confidence_counts": tier_counts,
        "crowded_box_events": int(crowded_count),
        "low_score_continuation_events": int(low_score_count),
    }
    return per_frame_rows, stats


def _color_for_id(tid: int) -> Tuple[int, int, int]:
    """Deterministic vivid BGR color per track ID."""
    return (int((tid * 67) % 200 + 45), int((tid * 131) % 200 + 45), int((tid * 197) % 200 + 45))


def write_trajectory_csv(out_path: Path, per_frame_rows: Sequence[Sequence[dict]], start_frame: int) -> Path:
    cols = ["frame", "track_id", "x1", "y1", "x2", "y2", "score", "track_confidence", "track_length", "flags"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rows_f in per_frame_rows:
            for r in rows_f:
                row = dict(r)
                row["frame"] = start_frame + int(r["frame"])
                w.writerow(row)
    return out_path


def write_heatmap_png(
    out_path: Path,
    per_frame_rows: Sequence[Sequence[dict]],
    width: int,
    height: int,
    bg_rgb: Optional[np.ndarray] = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = [], []
    for rows_f in per_frame_rows:
        for r in rows_f:
            xs.append(0.5 * (r["x1"] + r["x2"]))
            ys.append(r["y2"])

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    if bg_rgb is not None:
        ax.imshow(bg_rgb, alpha=0.45)
    else:
        ax.set_facecolor("#14361d")
    if xs:
        h, _, _ = np.histogram2d(ys, xs, bins=[54, 128], range=[[0, height], [0, width]])
        h = cv2.GaussianBlur(h.astype(np.float32), (9, 9), 2.0)
        im = ax.imshow(
            h, extent=[0, width, height, 0], origin="upper", cmap="hot", alpha=0.65, aspect="auto"
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Footpoint density")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Football Pitch Occupancy Heat Map (Image Plane)")
    ax.set_xlabel("Image X (px)")
    ax.set_ylabel("Image Y (px)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _draw_annotated_frame(
    frame_bgr: np.ndarray,
    rows_f: Sequence[dict],
    frame_num: int,
    scale: float = 1.0,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if scale < 1.0:
        nw = max(2, int(round(w * scale)) & ~1)
        nh = max(2, int(round(h * scale)) & ~1)
        canvas = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = nw / w, nh / h
    else:
        canvas = frame_bgr.copy()
        nw, nh = w, h
        sx = sy = 1.0

    for r in rows_f:
        x1 = int(round(r["x1"] * sx))
        y1 = int(round(r["y1"] * sy))
        x2 = int(round(r["x2"] * sx))
        y2 = int(round(r["y2"] * sy))
        color = _color_for_id(r["track_id"])
        thickness = 2 if r["track_confidence"] == "high" else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        tag = f"#{r['track_id']} {r['score']:.2f}"
        if r["flags"]:
            tag += " *"
        cv2.putText(canvas, tag, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (nw, 28), (18, 22, 20), -1)
    banner = f"Frame {frame_num} | Active tracks: {len(rows_f)} | {DISCLAIMER_TEXT}"
    cv2.putText(canvas, banner, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 245, 238), 1, cv2.LINE_AA)
    return canvas


def write_annotated_video(
    out_path: Path,
    frames_bgr: Sequence[np.ndarray],
    per_frame_rows: Sequence[Sequence[dict]],
    fps: float,
    start_frame: int,
    max_width: int = 1920,
    preview_path: Optional[Path] = None,
) -> Path:
    """Render annotated video (downscaled to <= max_width for smooth browser playback) and
    transcode to H.264 (yuv420p + faststart) when ffmpeg / imageio-ffmpeg is available."""
    if not frames_bgr:
        raise RuntimeError("No frames available to render annotated video.")
    h, w = frames_bgr[0].shape[:2]
    scale = min(1.0, max_width / float(w)) if max_width and w > max_width else 1.0
    out_w = max(2, int(round(w * scale)) & ~1)
    out_h = max(2, int(round(h * scale)) & ~1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = _find_ffmpeg()
    raw_path = out_path.with_name(out_path.stem + "_raw.mp4") if ffmpeg_exe else out_path

    # Try H264/avc1 directly if ffmpeg is absent, falling back to mp4v
    writer = None
    codecs = ("mp4v",) if ffmpeg_exe else ("avc1", "H264", "mp4v")
    for cc in codecs:
        cand = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*cc), float(fps), (out_w, out_h))
        if cand.isOpened():
            writer = cand
            break
        cand.release()
    if writer is None:
        raise RuntimeError(f"OpenCV VideoWriter failed to open: {raw_path}")

    mid_idx = len(frames_bgr) // 2
    try:
        for f_idx, (frame, rows_f) in enumerate(zip(frames_bgr, per_frame_rows)):
            canvas = _draw_annotated_frame(frame, rows_f, start_frame + f_idx, scale=scale)
            writer.write(canvas)
            if preview_path is not None and f_idx == mid_idx:
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), canvas)
    finally:
        writer.release()

    if ffmpeg_exe and raw_path != out_path:
        try:
            subprocess.run(
                [
                    ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", str(raw_path),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out_path),
                ],
                check=True,
                timeout=120,
            )
            raw_path.unlink(missing_ok=True)
        except Exception:
            # Fallback to raw mp4 if libx264 fails
            if raw_path.is_file() and not out_path.is_file():
                raw_path.replace(out_path)
    return out_path


def _decode_segment(video_path: Path, start: int, end: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InputValidationError(f"Could not open video: {video_path}")
    frames: List[np.ndarray] = []
    try:
        for i in range(start):
            if not cap.grab():
                raise InputValidationError(f"Video ended early at frame {i} before start={start}.")
        for i in range(start, end):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise InputValidationError(f"Could not decode frame {i}.")
            frames.append(bgr)
    finally:
        cap.release()
    return frames


def run_cached_analysis(
    video_path: Path,
    det_dir: Path,
    clip_id: str,
    out_dir: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = 150,
    tracker_overrides: Optional[dict] = None,
    max_segment_frames: int = 600,
) -> dict:
    """Instant CPU-only product pipeline using cached detection (and optional feature) chunks."""
    from .features import assemble_features, missing_feature_chunks
    from .fullclip import assemble
    from .phase2 import plan_file

    t0 = time.time()
    meta = validate_video_input(video_path, start_frame, end_frame, max_segment_frames=max_segment_frames)
    pf = plan_file(det_dir, clip_id)
    if not pf.is_file():
        raise FileNotFoundError(f"No cached detection plan found for clip {clip_id} in {det_dir}")
    info = json.loads(pf.read_text(encoding="utf-8"))
    plan_start, plan_end = int(info["start"]), int(info["end"])
    if meta.start_frame < plan_start or meta.end_frame > plan_end:
        raise InputValidationError(
            f"Requested [{meta.start_frame}, {meta.end_frame}) outside cached range [{plan_start}, {plan_end})."
        )
    plan = [tuple(p) for p in info["plan"]]
    all_dets = assemble(det_dir, clip_id, plan)
    lo, hi = meta.start_frame - plan_start, meta.end_frame - plan_start
    dets = all_dets[lo:hi]

    feats = None
    if not missing_feature_chunks(det_dir, clip_id, plan):
        all_feats, _ = assemble_features(det_dir, clip_id, plan, all_dets)
        feats = all_feats[lo:hi]

    params = dict(FROZEN_FINAL_PARAMS)
    if tracker_overrides:
        params.update(tracker_overrides)
    preds, scores = run_tracker_v3(dets, feats, meta.width, meta.height, params)
    per_frame_rows, stats = compute_track_annotations(preds, scores, det_high=float(params["det_high"]))

    frames_bgr = _decode_segment(meta.path, meta.start_frame, meta.end_frame)
    out_dir = Path(out_dir)
    prev_p = out_dir / "preview_frame.png"
    vid_p = write_annotated_video(
        out_dir / "annotated.mp4", frames_bgr, per_frame_rows, meta.fps, meta.start_frame, preview_path=prev_p
    )
    csv_p = write_trajectory_csv(out_dir / "trajectories.csv", per_frame_rows, meta.start_frame)
    bg_rgb = cv2.cvtColor(frames_bgr[len(frames_bgr) // 2], cv2.COLOR_BGR2RGB)
    hm_p = write_heatmap_png(out_dir / "heatmap.png", per_frame_rows, meta.width, meta.height, bg_rgb=bg_rgb)

    summary = {
        "mode": "cached",
        "clip_id": clip_id,
        "video": str(meta.path),
        "resolution": f"{meta.width}x{meta.height}",
        "fps": meta.fps,
        "segment": {"start": meta.start_frame, "end": meta.end_frame, "frames": meta.end_frame - meta.start_frame},
        "features_used": feats is not None,
        "tracker_params": params,
        "stats": stats,
        "runtime_seconds": round(time.time() - t0, 2),
        "disclaimer": DISCLAIMER_TEXT,
        "outputs": {"video": str(vid_p), "preview": str(prev_p), "csv": str(csv_p), "heatmap": str(hm_p)},
    }
    sum_p = out_dir / "summary.json"
    sum_p.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    summary["outputs"]["summary"] = str(sum_p)
    return summary


def run_live_analysis(
    video_path: Path,
    weights_path: Path,
    out_dir: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = 150,
    reid_weights: Optional[Path] = None,
    tracker_overrides: Optional[dict] = None,
    max_segment_frames: int = 250,
    batch_size: int = 5,
    device: str = "auto",
) -> dict:
    """Live segment inference pipeline: validates input -> P2-more FP16 -> optional OSNet -> Tracker v3 -> outputs."""
    import torch

    from .detect import build_retinanet, detect_tiles, merge_tile_detections
    from .features import AppearanceExtractor, crop_box_rgb, extract_torso_lab
    from .tiles import tile_grid

    t0 = time.time()
    meta = validate_video_input(video_path, start_frame, end_frame, max_segment_frames=max_segment_frames)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model = build_retinanet(str(weights_path), (1024, 1080), score_thresh=0.05, nms_thresh=0.5).to(dev)
    windows = tile_grid(meta.width, meta.height, 1024, 1080, 128)
    extractor = AppearanceExtractor(backbone="osnet" if reid_weights else "auto", device=str(dev),
                                    reid_weights=str(reid_weights) if reid_weights else None)

    frames_bgr = _decode_segment(meta.path, meta.start_frame, meta.end_frame)
    dets, feats = [], []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        b, s = merge_tile_detections(detect_tiles(model, rgb, windows, dev, batch_size=batch_size), windows,
                                     (meta.width, meta.height), merge_iou=0.5, drop_edge_px=2.0)
        dets.append((b, s))
        keep_idx = np.flatnonzero(s >= 0.25)
        embs = np.zeros((len(s), extractor.emb_dim), dtype=np.float32)
        labs = np.full((len(s), 3), np.nan, dtype=np.float32)
        valid = np.zeros(len(s), dtype=bool)
        if len(keep_idx):
            crops = [crop_box_rgb(rgb, b[i]) for i in keep_idx]
            embs[keep_idx] = extractor.extract_batch(crops, batch_size=128)
            for i in keep_idx:
                labs[i] = extract_torso_lab(rgb, b[i])
            valid[keep_idx] = True
        feats.append({"emb": embs, "torso_lab": labs, "valid": valid})

    params = dict(FROZEN_FINAL_PARAMS)
    if tracker_overrides:
        params.update(tracker_overrides)
    preds, scores = run_tracker_v3(dets, feats, meta.width, meta.height, params)
    per_frame_rows, stats = compute_track_annotations(preds, scores, det_high=float(params["det_high"]))

    out_dir = Path(out_dir)
    prev_p = out_dir / "preview_frame.png"
    vid_p = write_annotated_video(
        out_dir / "annotated.mp4", frames_bgr, per_frame_rows, meta.fps, meta.start_frame, preview_path=prev_p
    )
    csv_p = write_trajectory_csv(out_dir / "trajectories.csv", per_frame_rows, meta.start_frame)
    bg_rgb = cv2.cvtColor(frames_bgr[len(frames_bgr) // 2], cv2.COLOR_BGR2RGB)
    hm_p = write_heatmap_png(out_dir / "heatmap.png", per_frame_rows, meta.width, meta.height, bg_rgb=bg_rgb)

    summary = {
        "mode": "live",
        "video": str(meta.path),
        "weights": str(weights_path),
        "reid_backbone": extractor.backbone_used,
        "device": str(dev),
        "resolution": f"{meta.width}x{meta.height}",
        "fps": meta.fps,
        "segment": {"start": meta.start_frame, "end": meta.end_frame, "frames": meta.end_frame - meta.start_frame},
        "tracker_params": params,
        "stats": stats,
        "runtime_seconds": round(time.time() - t0, 2),
        "disclaimer": DISCLAIMER_TEXT,
        "outputs": {"video": str(vid_p), "preview": str(prev_p), "csv": str(csv_p), "heatmap": str(hm_p)},
    }
    sum_p = out_dir / "summary.json"
    sum_p.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    summary["outputs"]["summary"] = str(sum_p)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Football Tracker CLI.")
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("cached", "live"):
        s = sub.add_parser(name)
        s.add_argument("--video", type=Path, required=True)
        s.add_argument("--start", type=int, default=0)
        s.add_argument("--end", type=int, default=150)
        s.add_argument("--det-high", type=float, default=float(FROZEN_FINAL_PARAMS["det_high"]))
        s.add_argument("--det-low", type=float, default=float(FROZEN_FINAL_PARAMS["det_low"]))
        s.add_argument("--merge-cos", type=float, default=float(FROZEN_FINAL_PARAMS["merge_cos_thresh"]))
        s.add_argument("--no-gta", action="store_true")
        s.add_argument("--out-dir", type=Path, required=True)
    c = sub.choices["cached"]
    c.add_argument("--det-dir", type=Path, required=True, help="folder containing <clip>_plan.json and chunks")
    c.add_argument("--clip-id", required=True)
    c.add_argument("--max-frames", type=int, default=600)
    l = sub.choices["live"]
    l.add_argument("--weights", type=Path, required=True, help="P2-more best.pt path")
    l.add_argument("--reid-weights", type=Path, default=None, help="optional fine-tuned OSNet .pt path")
    l.add_argument("--max-frames", type=int, default=250)
    l.add_argument("--batch-size", type=int, default=5)
    l.add_argument("--device", default="auto")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "det_high": args.det_high,
        "det_low": args.det_low,
        "merge_cos_thresh": args.merge_cos,
        "use_gta": not args.no_gta,
    }
    try:
        if args.mode == "cached":
            res = run_cached_analysis(args.video, args.det_dir, args.clip_id, args.out_dir,
                                      args.start, args.end, overrides, args.max_frames)
        else:
            res = run_live_analysis(args.video, args.weights, args.out_dir, args.start, args.end,
                                    args.reid_weights, overrides, args.max_frames, args.batch_size, args.device)
        print(json.dumps(res, indent=2))
    except (InputValidationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
