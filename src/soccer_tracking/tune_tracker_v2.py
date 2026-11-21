"""Tracker v2 tuning and confirmation (Phase II, WS-C1). Protocol: docs/phase2_amendment_tracker_v2.md (frozen before any run).

Three commands, all on validation clips only (``select_clips`` refuses train, test and reserve clips)::

    # 1. grid of 48 on the cached tuning window (NumPy reference HOTA, no TrackEval needed)
    python -m soccer_tracking.tune_tracker_v2 sweep    --run p2more --out results/tracker_v2_sweep.json
    # 2. TrackEval on the top 3, apply the tie rule -> results/tracker_v2_tuning.json (key "chosen", same format as Phase I)
    python -m soccer_tracking.tune_tracker_v2 finalize --sweep results/tracker_v2_sweep.json --out results/tracker_v2_tuning.json
    # 3. unseen confirmation window: chosen v2 settings against the inherited Phase I settings (needs a second `detect` run)
    python -m soccer_tracking.tune_tracker_v2 confirm  --run p2more_w2 --params results/tracker_v2_tuning.json

Needs TrackEval on PYTHONPATH for ``finalize`` and ``confirm``. Not covered by unit tests (it needs real cached detections);
the tracker and the selection rules it calls are (tests/test_tracker_v2.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .tracker_v2 import build_arm_v2, grid_v2, pick_final, rank_candidates


def load_sequences(manifest: Path, det_root: Path, run: str, clip_ids: Optional[Sequence[str]]) -> Dict[str, dict]:
    """Cached detections and ground truth of every validation clip that has a plan file in ``det_root/run``."""
    from .data import check_annotations, load_manifest
    from .fullclip import assemble
    from .mot import load_mot
    from .phase2 import PHASE2_ROLES, plan_file, select_clips
    from .track_eval import gt_sequence

    ids = select_clips(None, list(clip_ids), False, False) if clip_ids else list(PHASE2_ROLES["val"])
    by_id = {c.clip_id: c for c in load_manifest(manifest)}
    det_dir = Path(det_root) / run
    seqs: Dict[str, dict] = {}
    for cid in ids:
        pf = plan_file(det_dir, cid)
        if not pf.is_file():
            raise FileNotFoundError(f"no detections for clip {cid} in {det_dir} (run `phase2 detect` first)")
        info = json.loads(pf.read_text(encoding="utf-8"))
        clip, n = by_id[cid], info["end"] - info["start"]
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ValueError("; ".join(problems))
        seqs[f"{cid}_{info['start']}_{info['end']}"] = {
            "gt": gt_sequence(clip, ann, info["start"], n), "dets": assemble(det_dir, cid, [tuple(p) for p in info["plan"]]),
            "width": clip.width, "height": clip.height, "clip_id": cid, "start": info["start"], "end": info["end"]}
    return seqs


def score_params(seqs: Dict[str, dict], preds_fn, use_trackeval: bool) -> dict:
    """``preds_fn(dets) -> (preds, scores)``; returns the dict from ``evaluate_sequences``."""
    from .hota_eval import evaluate_sequences

    arm = {}
    for name, s in seqs.items():
        preds, sc = preds_fn(s["dets"])
        arm[name] = {"gt": s["gt"], "pred": preds, "scores": sc, "width": s["width"], "height": s["height"]}
    return evaluate_sequences(arm, use_trackeval=use_trackeval)


def _env(manifest: Path) -> dict:
    from .track_eval import _env as env

    return env(manifest)


def _write(out: Path, record: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")


def cmd_sweep(args) -> None:
    seqs = load_sequences(args.manifest, args.det_root, args.run, args.clip)
    rows = []
    grid = list(grid_v2())
    for k, params in enumerate(grid, start=1):
        m = score_params(seqs, lambda d, p=params: build_arm_v2(d, p), use_trackeval=False)["reference"]["mean_over_seqs"]
        rows.append({"params": params, "hota": m["HOTA"], "deta": m["DetA"], "assa": m["AssA"], "loca": m["LocA"], "idsw": m["IDSW"]})
        print(f"[{k}/{len(grid)}] {params}  HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} IDSW {m['IDSW']:.1f}", flush=True)
    ranked = rank_candidates(rows)
    record = {"protocol": "docs/phase2_amendment_tracker_v2.md", "run": args.run, "det_root": str(args.det_root),
              "windows": {n: {k: s[k] for k in ("clip_id", "start", "end")} for n, s in seqs.items()},
              "top_k": args.top_k, "candidates": ranked[:args.top_k], "grid_results": rows, "environment": _env(args.manifest)}
    _write(Path(args.out), record)
    print("top candidates (NumPy reference HOTA):")
    for r in ranked[:args.top_k]:
        print(f"  HOTA {r['hota']:.4f} IDSW {r['idsw']:.1f}  {r['params']}")
    print(f"-> {args.out}")


def cmd_finalize(args) -> None:
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    seqs = load_sequences(args.manifest, Path(sweep["det_root"]), sweep["run"], None)
    rescored = []
    for cand in sweep["candidates"]:
        te = score_params(seqs, lambda d, p=cand["params"]: build_arm_v2(d, p), use_trackeval=True)["trackeval"]
        if te is None:
            raise ImportError("TrackEval is required for `finalize` (put it on PYTHONPATH)")
        m = te["mean_over_seqs"]
        rescored.append({"params": cand["params"], "hota": m["HOTA"], "deta": m["DetA"], "assa": m["AssA"], "loca": m["LocA"],
                         "fp": m["FP"], "fn": m["FN"], "idsw": m["IDSW"], "reference_hota": cand["hota"]})
        print(f"TrackEval {cand['params']}: HOTA {m['HOTA']:.4f} AssA {m['AssA']:.4f} FP {m['FP']:.1f} FN {m['FN']:.1f} IDSW {m['IDSW']:.1f}")
    chosen = pick_final(rescored)
    _write(Path(args.out), {"protocol": "docs/phase2_amendment_tracker_v2.md", "tracker": "ByteIouTracker", "chosen": chosen["params"],
                            "tuning_run": sweep["run"], "tuning_windows": sweep["windows"], "rescored_top_k": rescored,
                            "rule": "best TrackEval mean HOTA; within 0.005 -> lower IDSW, then smaller max_age",
                            "environment": _env(args.manifest)})
    print(f"chosen {chosen['params']} -> {args.out}")


def cmd_confirm(args) -> None:
    from .phase2 import build_arm

    v2 = json.loads(Path(args.params).read_text(encoding="utf-8"))["chosen"]
    base = json.loads(Path(args.baseline_params).read_text(encoding="utf-8"))["chosen"]
    seqs = load_sequences(args.manifest, args.det_root, args.run, args.clip)
    res = {}
    for label, fn in (("phase1_iou", lambda d: build_arm(d, "iou", 0.5, base)), ("tracker_v2", lambda d: build_arm_v2(d, v2))):
        r = score_params(seqs, fn, use_trackeval=True)
        if r["trackeval"] is None:
            raise ImportError("TrackEval is required for `confirm` (put it on PYTHONPATH)")
        res[label] = r
    a, b = res["phase1_iou"]["trackeval"]["mean_over_seqs"], res["tracker_v2"]["trackeval"]["mean_over_seqs"]
    d_hota, d_assa = b["HOTA"] - a["HOTA"], b["AssA"] - a["AssA"]
    crit = {"delta_hota": d_hota, "delta_assa": d_assa, "min_delta_hota": 0.01, "hota_ok": d_hota >= 0.01, "assa_ok": d_assa >= 0.0}
    _write(Path(args.out), {"protocol": "docs/phase2_amendment_tracker_v2.md", "run": args.run, "phase1_params": base, "v2_params": v2,
                            "windows": {n: {k: s[k] for k in ("clip_id", "start", "end")} for n, s in seqs.items()},
                            "results": res, "criteria": crit, "environment": _env(args.manifest)})
    for label in res:
        m = res[label]["trackeval"]["mean_over_seqs"]
        print(f"{label:<11} HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} LocA {m['LocA']:.4f} FP {m['FP']:.1f} FN {m['FN']:.1f} IDSW {m['IDSW']:.1f}")
    print(f"delta HOTA {d_hota:+.4f} (need >= +0.01): {'PASS' if crit['hota_ok'] else 'FAIL'};  delta AssA {d_assa:+.4f} (need >= 0): "
          f"{'PASS' if crit['assa_ok'] else 'FAIL'}  -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    from .data import DEFAULT_MANIFEST
    from .phase2 import DEFAULT_DET_ROOT

    p = argparse.ArgumentParser(description="Tracker v2: pre-registered sweep, TrackEval rescoring and confirmation (validation clips only).")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("sweep", "finalize", "confirm"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    for name in ("sweep", "confirm"):
        s = sub.choices[name]
        s.add_argument("--run", required=True, help="detection run folder under --det-root (sweep: tuning window; confirm: unseen window)")
        s.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT)
        s.add_argument("--clip", nargs="+", default=None, help="validation clip ids (default: both validation clips)")
    sw = sub.choices["sweep"]
    sw.add_argument("--top-k", type=int, default=3)
    sw.add_argument("--out", type=Path, default=Path("results") / "tracker_v2_sweep.json")
    fi = sub.choices["finalize"]
    fi.add_argument("--sweep", type=Path, default=Path("results") / "tracker_v2_sweep.json")
    fi.add_argument("--out", type=Path, default=Path("results") / "tracker_v2_tuning.json")
    co = sub.choices["confirm"]
    co.add_argument("--params", type=Path, default=Path("results") / "tracker_v2_tuning.json")
    co.add_argument("--baseline-params", type=Path, default=Path("results") / "tracker_tuning.json")
    co.add_argument("--out", type=Path, default=Path("results") / "tracker_v2_confirm.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        {"sweep": cmd_sweep, "finalize": cmd_finalize, "confirm": cmd_confirm}[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
