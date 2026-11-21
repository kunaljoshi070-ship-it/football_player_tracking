"""Phase II command line: full-clip (or windowed) detection in resumable chunks, then HOTA scoring of detector-only or IoU-tracker output.

Roles follow ``docs/phase2_plan.md`` section 3 (GTATrack-aligned). They are separate from the frozen Phase I ``split`` column in
``data/manifest.csv`` (which is only used here for file paths, sizes and hashes).

    train    118575 118576 118577 117092
    val      117093 128058                      all development decisions use only these
    test     128057 132831                      needs --allow-test; run once per frozen configuration
    reserve  118578 132877                      needs --allow-reserve; kept for the live unseen test in the viva

Commands (run from the repo root, ``PYTHONPATH=src`` and TrackEval on PYTHONPATH for ``score``)::

    python -m soccer_tracking.phase2 detect --role val --run p1model --weights /content/drive/MyDrive/soccer_runs/finetune/best.pt \
        --start 1000 --end 2000 --chunk 250            # a window; omit --start/--end for whole clips
    python -m soccer_tracking.phase2 score  --role val --run p1model --arm iou --params results/tracker_tuning.json

``detect`` needs torch and a GPU and is not covered by unit tests. A window is scored against the same frames of the ground truth,
renumbered from 1 for TrackEval. The chunk plan is stored next to the detections so ``score`` cannot disagree with ``detect``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PHASE2_ROLES: Dict[str, List[str]] = {
    "train": ["118575", "118576", "118577", "117092"],
    "val": ["117093", "128058"],
    "test": ["128057", "132831"],
    "reserve": ["118578", "132877"],
}
DEFAULT_DET_ROOT = Path("data") / "processed" / "detections_p2"


def role_of(clip_id: str) -> str:
    for role, ids in PHASE2_ROLES.items():
        if clip_id in ids:
            return role
    raise ValueError(f"clip {clip_id} has no Phase II role")


def check_roles_disjoint() -> None:
    seen: Dict[str, str] = {}
    for role, ids in PHASE2_ROLES.items():
        for c in ids:
            if c in seen:
                raise ValueError(f"clip {c} is in both {seen[c]} and {role}")
            seen[c] = role


def select_clips(role: Optional[str], clips: Optional[Sequence[str]], allow_test: bool, allow_reserve: bool) -> List[str]:
    """Resolve ``--role`` / ``--clip`` to clip ids and enforce the evaluation discipline."""
    check_roles_disjoint()
    if bool(role) == bool(clips):
        raise ValueError("give exactly one of --role or --clip")
    ids = list(PHASE2_ROLES[role]) if role else list(clips)
    for c in ids:
        r = role_of(c)
        if r == "train":
            raise ValueError(f"clip {c} is a training clip: scoring or detecting on it says nothing about generalisation (use val)")
        if r == "test" and not allow_test:
            raise ValueError(f"clip {c} is a held-out test clip: pass --allow-test only for the single final run of a frozen configuration")
        if r == "reserve" and not allow_reserve:
            raise ValueError(f"clip {c} is a reserve clip, kept for the live unseen test: pass --allow-reserve only then")
    return ids

def training_clips(manifest, train_ids=None, val_ids=None):
    """Phase II training and validation Clip objects; test and reserve clips are refused."""
    from .data import ManifestError, load_manifest
    by_id = {c.clip_id: c for c in load_manifest(manifest)}

    def pick(ids, role):
        ids = list(ids) if ids else list(PHASE2_ROLES[role])
        bad = [i for i in ids if i not in PHASE2_ROLES[role]]
        if bad:
            raise ManifestError(f"clip(s) {bad} are not Phase II {role} clips {PHASE2_ROLES[role]}")
        absent = [i for i in ids if i not in by_id]
        if absent:
            raise ManifestError(f"clip(s) {absent} not in the manifest")
        return [by_id[i] for i in ids]

    return pick(train_ids, "train"), pick(val_ids, "val")


def build_arm(dets, arm: str, score: float, params: Optional[dict] = None):
    """Per-frame ``(ids, boxes)`` predictions and score arrays for one arm.
    ``detector``: every detection at or above ``score`` gets a unique id (so AssA is meaningless; read DetA, LocA, FP, FN).
    ``iou``: the Phase I IoU tracker with the given settings (coasted boxes are output, as in Phase I)."""
    from .tracker import IouTracker

    preds, scores = [], []
    if arm == "detector":
        nxt = 1
        for b, s in dets:
            keep = np.asarray(s) >= score
            n = int(keep.sum())
            preds.append((np.arange(nxt, nxt + n, dtype=np.int64), np.asarray(b)[keep].reshape(-1, 4)))
            scores.append(np.asarray(s)[keep])
            nxt += n
    elif arm == "iou":
        if not params:
            raise ValueError("arm 'iou' needs tracker settings (--params results/tracker_tuning.json)")
        tracker = IouTracker(**params)
        for b, s in dets:
            keep = np.asarray(s) >= score
            ids, boxes, sc, _ = tracker.update(np.asarray(b)[keep], np.asarray(s)[keep])
            preds.append((ids, boxes))
            scores.append(sc)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return preds, scores


def plan_file(det_dir: Path, clip_id: str) -> Path:
    return Path(det_dir) / f"{clip_id}_plan.json"


# --------------------------------------------------------------------------- commands
def cmd_detect(args) -> None:
    import torch

    from .data import load_manifest
    from .detect import build_retinanet
    from .evaluate import environment_record
    from .fullclip import detect_clip, plan_chunks

    ids = select_clips(args.role, args.clip, args.allow_test, args.allow_reserve)
    by_id = {c.clip_id: c for c in load_manifest(args.manifest)}
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = build_retinanet(args.weights, tuple(args.tile), args.score_floor, args.nms_thresh).to(device)
    det_dir = Path(args.det_root) / args.run
    for cid in ids:
        clip = by_id[cid]
        plan = plan_chunks(clip.frames, args.chunk, args.start, args.end)
        det_dir.mkdir(parents=True, exist_ok=True)
        plan_file(det_dir, cid).write_text(json.dumps({"clip_id": cid, "start": plan[0][0], "end": plan[-1][0] + plan[-1][1],
                                                       "chunk": args.chunk, "plan": plan}) + "\n", encoding="utf-8")
        detect_clip(model, clip, det_dir, plan, tuple(args.tile), args.overlap, device, args.batch_size, args.merge_iou, args.drop_edge_px)
    (det_dir / "detect_config.json").write_text(json.dumps(
        {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "environment": environment_record(args.manifest, device)}, indent=2) + "\n", encoding="utf-8")
    print(f"detections for {ids} -> {det_dir}")


def cmd_score(args) -> None:
    from .data import check_annotations, load_manifest
    from .fullclip import assemble
    from .hota_eval import evaluate_sequences
    from .mot import load_mot
    from .track_eval import _env, gt_sequence

    ids = select_clips(args.role, args.clip, args.allow_test, args.allow_reserve)
    by_id = {c.clip_id: c for c in load_manifest(args.manifest)}
    params = None
    if args.arm in ("iou", "iou_v2"):
        default = "tracker_v2_tuning.json" if args.arm == "iou_v2" else "tracker_tuning.json"
        params = json.loads(Path(args.params or Path("results") / default).read_text(encoding="utf-8"))["chosen"]
    det_dir = Path(args.det_root) / args.run
    seqs, meta = {}, {}
    for cid in ids:
        clip = by_id[cid]
        pf = plan_file(det_dir, cid)
        if not pf.is_file():
            raise FileNotFoundError(f"no detections for clip {cid} in {det_dir} (run `detect` first)")
        info = json.loads(pf.read_text(encoding="utf-8"))
        plan = [tuple(p) for p in info["plan"]]
        n = info["end"] - info["start"]
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ValueError("; ".join(problems))
        dets = assemble(det_dir, cid, plan)
        if args.static_frac > 0:
            from .postfilter import suppress_static
            dets = suppress_static([(b, s) for b, s in dets], args.static_window, 5, args.static_frac, args.static_iou, args.score)
        if args.arm == "iou_v2":
            from .tracker_v2 import build_arm_v2
            preds, sc = build_arm_v2(dets, params)      # v2 applies det_low/det_high itself
        else:
            preds, sc = build_arm(dets, args.arm, args.score, params)
        name = f"{cid}_{info['start']}_{info['end']}"
        seqs[name] = {"gt": gt_sequence(clip, ann, info["start"], n), "pred": preds, "scores": sc, "width": clip.width, "height": clip.height}
        meta[name] = {"clip_id": cid, "role": role_of(cid), "start": info["start"], "end": info["end"], "frames": n}
    res = evaluate_sequences(seqs, use_trackeval=not args.no_trackeval)
    record = {"run": args.run, "arm": args.arm, "score": args.score, "tracker_params": params, "postfilter": {"static_frac": args.static_frac, "window": args.static_window, "iou": args.static_iou}, "sequences": meta, "results": res,
              "note": "mean_over_seqs is the Challenge-style per-clip mean; pooled sums counts. 'detector' arm ids are unique per box (AssA not meaningful). "
                      "Report TrackEval numbers; 'reference' is the NumPy cross-check.",
              "environment": _env(args.manifest)}
    out = Path(args.out or Path("results") / f"hota_{args.run}_{args.arm}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    best = res["trackeval"] or res["reference"]
    src = "TrackEval" if res["trackeval"] else f"NumPy reference only ({res['trackeval_error']})"
    print(f"[{src}] mean over clips: " + ", ".join(f"{k} {v:.4g}" for k, v in best["mean_over_seqs"].items()) + f"  -> {out}")


def build_parser() -> argparse.ArgumentParser:
    from .data import DEFAULT_MANIFEST

    p = argparse.ArgumentParser(description="Phase II: resumable full-clip detection and HOTA scoring.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("detect", "score"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        s.add_argument("--role", choices=("val", "test", "reserve"))
        s.add_argument("--clip", nargs="+")
        s.add_argument("--run", required=True, help="name of this detection run (a folder under --det-root)")
        s.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT)
        s.add_argument("--allow-test", action="store_true")
        s.add_argument("--allow-reserve", action="store_true")
    d = sub.choices["detect"]
    d.add_argument("--weights", required=True)
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--end", type=int, default=None)
    d.add_argument("--chunk", type=int, default=500)
    d.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    d.add_argument("--overlap", type=int, default=128)
    d.add_argument("--score-floor", type=float, default=0.05)
    d.add_argument("--merge-iou", type=float, default=0.5)
    d.add_argument("--nms-thresh", type=float, default=0.5, help="NMS inside the detector, per tile (Phase I: 0.5)")
    d.add_argument("--drop-edge-px", type=float, default=2.0)
    d.add_argument("--batch-size", type=int, default=2)
    d.add_argument("--device", default="auto")
    d.add_argument("--seed", type=int, default=0)
    e = sub.choices["score"]
    e.add_argument("--arm", choices=("detector", "iou", "iou_v2"), required=True)
    e.add_argument("--params", type=Path, default=None, help="tracker settings JSON (default depends on --arm)")
    e.add_argument("--score", type=float, default=0.5)
    e.add_argument("--no-trackeval", action="store_true", help="NumPy reference only")
    e.add_argument("--static-frac", type=float, default=0.0, help="0 = off; else drop boxes static in this share of +-window/2 (e.g. 0.9)")
    e.add_argument("--static-window", type=int, default=250)
    e.add_argument("--static-iou", type=float, default=0.8)
    e.add_argument("--out", type=Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        {"detect": cmd_detect, "score": cmd_score}[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
