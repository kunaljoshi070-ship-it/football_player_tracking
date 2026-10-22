"""Error analysis of the player detector at a fixed operating point (score >= 0.5, IoU >= 0.5).

For the same sampled frames as ``evaluate.py`` it matches predictions to ground truth and explains the errors:

* False negatives (missed players) are typed ``merged_with_neighbour`` (a confident box covers the player but
  was matched to someone else), ``low_score`` (a box with IoU >= 0.5 exists but scores below the threshold),
  ``poorly_localized`` (an overlapping box with IoU 0.1-0.5) or ``missed`` (nothing overlaps).
* False positives are typed ``duplicate`` (IoU >= 0.5 with a player who already has a better box),
  ``poorly_localized`` (IoU 0.1-0.5 with a player) or ``no_gt_overlap`` (overlaps no labelled player: an
  unlabelled person such as a referee or substitute, or background).
* Recall is broken down by box height, by crowding (share of the box covered by another player's box) and by
  distance to an interior tile border; precision by score band.

Run from the repo root with ``PYTHONPATH=src``::

    python -m soccer_tracking.errors --split val --weights runs/finetune/best.pt

Counts and typed rows go to ``results/errors_<split>/`` (no pixels, safe to commit). Image panels of sampled errors
are written to ``data/processed/errors_<split>/`` (git-ignored): they contain frames of people, so keep them local.
The test split needs ``--allow-test``; analysing test errors is descriptive only, never a basis for changing the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import DEFAULT_FRAMES_ROOT, DEFAULT_MANIFEST, ManifestError, check_annotations, frame_path, load_manifest, sample_frames
from .detect import build_retinanet, detect_tiles, merge_tile_detections
from .metrics import iou_matrix
from .mot import MotFormatError, load_mot
from .tiles import tile_grid

SIZE_BINS: Tuple[Tuple[float, float], ...] = ((0, 30), (30, 40), (40, 50), (50, 70), (70, 1e9))
FN_TYPES = ("merged_with_neighbour", "low_score", "poorly_localized", "missed")
FP_TYPES = ("duplicate", "poorly_localized", "no_gt_overlap")


def _bin_label(lo: float, hi: float) -> str:
    return f"<{hi:g}" if lo == 0 else (f">={lo:g}" if hi >= 1e8 else f"{lo:g}-{hi:g}")


def assign(gt: np.ndarray, boxes: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching by descending score. Returns ``(gt_of_pred, pred_of_gt)`` with -1 for unmatched."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    gt_of_pred = -np.ones(len(boxes), dtype=int)
    pred_of_gt = -np.ones(len(gt), dtype=int)
    if len(gt) and len(boxes):
        ious = iou_matrix(boxes, gt)
        for p in np.argsort(-scores, kind="stable"):
            cand = np.where(pred_of_gt >= 0, -1.0, ious[p])
            g = int(np.argmax(cand))
            if cand[g] >= thr:
                gt_of_pred[p], pred_of_gt[g] = g, p
    return gt_of_pred, pred_of_gt


def fp_type(box: np.ndarray, gt: np.ndarray) -> str:
    best = float(iou_matrix(np.asarray(box)[None], gt)[0].max()) if len(gt) else 0.0
    return "duplicate" if best >= 0.5 else ("poorly_localized" if best >= 0.1 else "no_gt_overlap")


def fn_type(gt_box: np.ndarray, boxes: np.ndarray, scores: np.ndarray, score_thr: float, iou_thr: float = 0.5) -> str:
    """Why this ground-truth player has no matched confident detection (``boxes`` = all predictions above the floor)."""
    if len(boxes) == 0:
        return "missed"
    ious = iou_matrix(np.asarray(gt_box)[None], boxes)[0]
    if np.any((ious >= iou_thr) & (scores >= score_thr)):
        return "merged_with_neighbour"
    if np.any((ious >= iou_thr) & (scores < score_thr)):
        return "low_score"
    return "poorly_localized" if np.any(ious >= 0.1) else "missed"


def crowding(gt: np.ndarray) -> np.ndarray:
    """For each box, the largest share of its own area covered by another ground-truth box."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    out = np.zeros(len(gt))
    for i in range(len(gt)):
        for j in range(len(gt)):
            if i == j:
                continue
            iw = min(gt[i, 2], gt[j, 2]) - max(gt[i, 0], gt[j, 0])
            ih = min(gt[i, 3], gt[j, 3]) - max(gt[i, 1], gt[j, 1])
            area = (gt[i, 2] - gt[i, 0]) * (gt[i, 3] - gt[i, 1])
            if iw > 0 and ih > 0 and area > 0:
                out[i] = max(out[i], iw * ih / area)
    return out


def border_distance(center_x: np.ndarray, windows: Sequence[Tuple[int, int, int, int]], frame_w: int) -> np.ndarray:
    """Horizontal distance from each x to the nearest *interior* tile border (inf if the frame has one tile)."""
    borders = sorted({w[0] for w in windows if w[0] > 0} | {w[2] for w in windows if w[2] < frame_w})
    if not borders:
        return np.full(len(center_x), np.inf)
    return np.abs(np.asarray(center_x, dtype=np.float64)[:, None] - np.asarray(borders)[None, :]).min(axis=1)


def analyze_frame(frame: int, gt_ids: np.ndarray, gt: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                  windows: Sequence[Tuple[int, int, int, int]], frame_w: int, score_thr: float = 0.5,
                  iou_thr: float = 0.5, near_px: float = 16.0) -> Tuple[List[dict], List[dict]]:
    """Typed ground-truth rows and prediction rows (predictions at or above ``score_thr``) for one frame."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = scores >= score_thr
    conf_boxes, conf_scores = boxes[keep], scores[keep]
    gt_of_pred, pred_of_gt = assign(gt, conf_boxes, conf_scores, iou_thr)
    crowd = crowding(gt)
    dist = border_distance((gt[:, 0] + gt[:, 2]) / 2, windows, frame_w) if len(gt) else np.zeros(0)
    gt_rows, pred_rows = [], []
    for g in range(len(gt)):
        matched = bool(pred_of_gt[g] >= 0)
        gt_rows.append({"frame": frame, "gt_id": int(gt_ids[g]), "box": gt[g].tolist(), "height": float(gt[g, 3] - gt[g, 1]),
                        "width": float(gt[g, 2] - gt[g, 0]), "crowding": float(crowd[g]), "border_dist": float(dist[g]),
                        "near_border": bool(dist[g] < near_px), "matched": matched,
                        "type": "" if matched else fn_type(gt[g], boxes, scores, score_thr, iou_thr)})
    for p in range(len(conf_boxes)):
        matched = bool(gt_of_pred[p] >= 0)
        pred_rows.append({"frame": frame, "box": conf_boxes[p].tolist(), "score": float(conf_scores[p]), "matched": matched,
                          "type": "" if matched else fp_type(conf_boxes[p], gt)})
    return gt_rows, pred_rows


def _rate(rows: Sequence[dict], flag: str = "matched") -> Tuple[int, Optional[float]]:
    return len(rows), (sum(r[flag] for r in rows) / len(rows) if rows else None)


def summarize_errors(gt_rows: Sequence[dict], pred_rows: Sequence[dict], score_thr: float = 0.5, crowd_thr: float = 0.2) -> dict:
    n_gt, tp = len(gt_rows), sum(r["matched"] for r in gt_rows)
    n_pred, fp = len(pred_rows), sum(not r["matched"] for r in pred_rows)
    out: dict = {"n_gt": n_gt, "tp": tp, "fn": n_gt - tp, "n_pred": n_pred, "fp": fp,
                 "precision": (n_pred - fp) / n_pred if n_pred else None, "recall": tp / n_gt if n_gt else None}
    out["fn_types"] = {t: sum(1 for r in gt_rows if r["type"] == t) for t in FN_TYPES}
    out["fp_types"] = {t: sum(1 for r in pred_rows if r["type"] == t) for t in FP_TYPES}
    by_height = []
    for lo, hi in SIZE_BINS:
        n, rec = _rate([r for r in gt_rows if lo <= r["height"] < hi])
        by_height.append({"height_px": _bin_label(lo, hi), "n_gt": n, "recall": rec})
    out["recall_by_height"] = by_height
    for name, pred in (("crowded", lambda r: r["crowding"] >= crowd_thr), ("near_tile_border", lambda r: r["near_border"])):
        yes, no = [r for r in gt_rows if pred(r)], [r for r in gt_rows if not pred(r)]
        out[f"recall_by_{name}"] = {"yes": dict(zip(("n_gt", "recall"), _rate(yes))), "no": dict(zip(("n_gt", "recall"), _rate(no)))}
    edges = sorted({score_thr} | {e for e in (0.7, 0.9) if e > score_thr} | {1.01})
    out["precision_by_score"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = [r for r in pred_rows if lo <= r["score"] < hi]
        out["precision_by_score"].append({"score": f"{lo:g}-{min(hi, 1.0):g}", "n_pred": len(rows), "precision": _rate(rows)[1]})
    return out


# --------------------------------------------------------------------------- panels
def crop_panel(image_rgb: np.ndarray, center: Tuple[float, float], gt_boxes: np.ndarray, pred_boxes: np.ndarray,
               highlight: Sequence[float], label: str, size: int = 160, scale: int = 2) -> np.ndarray:
    """RGB panel of ``size`` px around ``center`` (shifted to stay inside the frame), upscaled by ``scale``.
    Ground truth in yellow, confident predictions in cyan, the error box in red."""
    h, w = image_rgb.shape[:2]
    x0 = int(min(max(center[0] - size / 2, 0), max(w - size, 0)))
    y0 = int(min(max(center[1] - size / 2, 0), max(h - size, 0)))
    crop = image_rgb[y0:y0 + size, x0:x0 + size]
    panel = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

    def draw(b, color, thick):
        p1 = (int((b[0] - x0) * scale), int((b[1] - y0) * scale))
        p2 = (int((b[2] - x0) * scale), int((b[3] - y0) * scale))
        cv2.rectangle(panel, p1, p2, color, thick)

    for b in np.asarray(gt_boxes).reshape(-1, 4):
        draw(b, (255, 220, 0), 1)
    for b in np.asarray(pred_boxes).reshape(-1, 4):
        draw(b, (0, 220, 255), 1)
    draw(highlight, (255, 40, 40), 2)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(panel, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def make_montage(panels: Sequence[np.ndarray], cols: int = 4, pad: int = 4) -> Optional[np.ndarray]:
    if not panels:
        return None
    ph, pw = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    sheet = np.full((rows * (ph + pad) + pad, cols * (pw + pad) + pad, 3), 255, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        sheet[pad + r * (ph + pad): pad + r * (ph + pad) + ph, pad + c * (pw + pad): pad + c * (pw + pad) + pw] = p
    return sheet


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Typed error analysis of the player detector.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--weights", default="coco", help="'coco', 'none' or a fine-tuned .pt checkpoint")
    p.add_argument("--stride", type=int, default=125)
    p.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--score-floor", type=float, default=0.05)
    p.add_argument("--merge-iou", type=float, default=0.5)
    p.add_argument("--drop-edge-px", type=float, default=2.0)
    p.add_argument("--score", type=float, default=0.5, help="operating score threshold")
    p.add_argument("--iou", type=float, default=0.5, help="matching IoU")
    p.add_argument("--n-per-type", type=int, default=8, help="image panels sampled per error type")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path, default=None, help="default results/errors_<split>")
    p.add_argument("--images-dir", type=Path, default=None, help="default data/processed/errors_<split> (git-ignored)")
    return p


def _print_summary(s: dict) -> None:
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"\nground truth {s['n_gt']}: found {s['tp']}, missed {s['fn']} (recall {f(s['recall'])});  "
          f"confident predictions {s['n_pred']}: false {s['fp']} (precision {f(s['precision'])})")
    print("missed players by reason:   " + ", ".join(f"{k} {v}" for k, v in s["fn_types"].items()))
    print("false positives by reason:  " + ", ".join(f"{k} {v}" for k, v in s["fp_types"].items()))
    print("recall by box height (px):  " + ", ".join(f"{r['height_px']}: {f(r['recall'])} (n={r['n_gt']})" for r in s["recall_by_height"]))
    for key, label in (("recall_by_crowded", "crowded (>=20% covered by a neighbour)"), ("recall_by_near_tile_border", "within 16 px of a tile border")):
        y, n = s[key]["yes"], s[key]["no"]
        print(f"recall {label}: {f(y['recall'])} (n={y['n_gt']}) vs otherwise {f(n['recall'])} (n={n['n_gt']})")
    print("precision by score band:    " + ", ".join(f"{r['score']}: {f(r['precision'])} (n={r['n_pred']})" for r in s["precision_by_score"]))


def run(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise ValueError("the test split is held out: pass --allow-test only for a descriptive look, never to change the model")
    import torch

    from .evaluate import environment_record

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    clips = load_manifest(args.manifest, [args.split])
    model = build_retinanet(args.weights, tuple(args.tile), args.score_floor, 0.5).to(device)
    gt_rows: List[dict] = []
    pred_rows: List[dict] = []
    cache: Dict[Tuple[str, int], dict] = {}
    for clip in clips:
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ManifestError("; ".join(problems))
        windows = tile_grid(clip.width, clip.height, args.tile[0], args.tile[1], args.overlap)
        for f in sample_frames(clip.frames, args.stride):
            path = frame_path(args.frames_root, clip.clip_id, f)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[1] != clip.width or bgr.shape[0] != clip.height:
                raise FileNotFoundError(f"{path} missing or wrong size; run `python -m soccer_tracking.data extract --splits {args.split}`")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes, scores = merge_tile_detections(detect_tiles(model, rgb, windows, device, args.batch_size), windows,
                                                  (clip.width, clip.height), args.merge_iou, args.drop_edge_px)
            ids = np.array([d.track_id for d in ann[f]])
            g = np.array([[d.x, d.y, d.x + d.width, d.y + d.height] for d in ann[f]], dtype=np.float64)
            g[:, [0, 2]] = np.clip(g[:, [0, 2]], 0, clip.width)
            g[:, [1, 3]] = np.clip(g[:, [1, 3]], 0, clip.height)
            gr, pr = analyze_frame(f, ids, g, boxes, scores, windows, clip.width, args.score, args.iou)
            for r in gr + pr:
                r["clip_id"] = clip.clip_id
            gt_rows += gr
            pred_rows += pr
            cache[(clip.clip_id, f)] = {"gt": g, "conf": boxes[scores >= args.score]}
            print(f"clip {clip.clip_id} frame {f}: {len(gr)} players, {len(pr)} confident detections", flush=True)

    summary = summarize_errors(gt_rows, pred_rows, args.score)
    _print_summary(summary)

    out_dir = args.out_dir or Path("results") / f"errors_{args.split}"
    img_dir = args.images_dir or Path("data") / "processed" / f"errors_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    errors = [dict(kind="FN", **{k: r[k] for k in ("clip_id", "frame", "type", "height", "crowding", "border_dist")}, score="",
                   x1=r["box"][0], y1=r["box"][1], x2=r["box"][2], y2=r["box"][3]) for r in gt_rows if not r["matched"]]
    errors += [dict(kind="FP", clip_id=r["clip_id"], frame=r["frame"], type=r["type"], height=r["box"][3] - r["box"][1], crowding="",
                    border_dist="", score=r["score"], x1=r["box"][0], y1=r["box"][1], x2=r["box"][2], y2=r["box"][3]) for r in pred_rows if not r["matched"]]
    cols = ["kind", "type", "clip_id", "frame", "x1", "y1", "x2", "y2", "height", "score", "crowding", "border_dist"]
    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(errors)
    record = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "summary": summary,
              "environment": environment_record(args.manifest, device)}
    (out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    clip_by_id = {c.clip_id: c for c in clips}
    for kind, types in (("fp", FP_TYPES), ("fn", FN_TYPES)):
        panels = []
        for t in types:
            rows = [e for e in errors if e["kind"] == kind.upper() and e["type"] == t]
            pick = rng.choice(len(rows), size=min(args.n_per_type, len(rows)), replace=False) if rows else []
            for i in sorted(int(i) for i in pick):
                e = rows[i]
                c = cache[(e["clip_id"], e["frame"])]
                bgr = cv2.imread(str(frame_path(args.frames_root, e["clip_id"], e["frame"])), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                box = [e["x1"], e["y1"], e["x2"], e["y2"]]
                label = f"{kind.upper()} {t[:16]}" + (f" s={e['score']:.2f}" if kind == "fp" else f" h={e['height']:.0f}")
                panels.append(crop_panel(rgb, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), c["gt"], c["conf"], box, label))
        sheet = make_montage(panels)
        if sheet is not None:
            cv2.imwrite(str(img_dir / f"{kind}_examples.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"\ncounts and typed rows -> {out_dir} (summary.json, errors.csv); image panels (keep local) -> {img_dir}")
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
