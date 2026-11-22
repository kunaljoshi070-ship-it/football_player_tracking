"""Score the frozen Phase II final configuration ONCE on the held-out test split (requires --allow-test).

Frozen config: docs/phase2_final_config.md (2026-11-22).

Command line::

    PYTHONPATH=src:/content/TrackEval python3 -m soccer_tracking.score_final_test \
        --allow-test --run p2more_test \
        --det-root /content/drive/MyDrive/soccer_runs/detections_p2 \
        --out results/p2_final_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .tracker_v3 import DEFAULT_V3_PARAMS, run_tracker_v3

FROZEN_FINAL_PARAMS: Dict[str, object] = {
    **DEFAULT_V3_PARAMS,
    "det_high": 0.78,
    "det_low": 0.45,
    "expand_ratios": (0.0, 0.35, 0.70),
    "prox_thresh": 0.80,
    "app_weight": 0.50,
    "app_thresh": 0.45,
    "online_lab_gate": 42.0,
    "max_age": 20,
    "min_hits": 3,
    "emb_alpha": 0.90,
    "use_gta": True,
    "max_gap_frames": 600,
    "max_speed_px": 12.0,
    "base_dist_px": 60.0,
    "merge_cos_thresh": 0.20,
    "team_color_gate": 45.0,
    "min_track_len": 12,
    "use_official_filter": False,
    "roster_k": None,
}


def run(args: argparse.Namespace) -> dict:
    from .data import check_annotations, load_manifest
    from .features import assemble_features
    from .fullclip import assemble
    from .hota_eval import evaluate_sequences
    from .mot import load_mot
    from .phase2 import plan_file, role_of, select_clips
    from .track_eval import _env, gt_sequence

    ids = select_clips(args.role, args.clip, args.allow_test, args.allow_reserve)
    by_id = {c.clip_id: c for c in load_manifest(args.manifest)}
    det_dir = Path(args.det_root) / args.run

    seqs, meta = {}, {}
    for cid in ids:
        pf = plan_file(det_dir, cid)
        if not pf.is_file():
            raise FileNotFoundError(f"no detections for clip {cid} in {det_dir} (run `phase2 detect` first)")
        info = json.loads(pf.read_text(encoding="utf-8"))
        plan = [tuple(p) for p in info["plan"]]
        clip, n = by_id[cid], info["end"] - info["start"]
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ValueError("; ".join(problems))
        dets = assemble(det_dir, cid, plan)
        feats, _ = assemble_features(det_dir, cid, plan, dets)
        preds, sc = run_tracker_v3(dets, feats, clip.width, clip.height, FROZEN_FINAL_PARAMS)
        name = f"{cid}_{info['start']}_{info['end']}"
        seqs[name] = {
            "gt": gt_sequence(clip, ann, info["start"], n),
            "pred": preds,
            "scores": sc,
            "width": clip.width,
            "height": clip.height,
        }
        meta[name] = {
            "clip_id": cid,
            "role": role_of(cid),
            "resolution": f"{clip.width}x{clip.height}",
            "start": info["start"],
            "end": info["end"],
            "frames": n,
        }

    res = evaluate_sequences(seqs, use_trackeval=not args.no_trackeval)
    record = {
        "protocol": "docs/phase2_final_config.md",
        "run": args.run,
        "detector": "P2-more (runs/p2_more_data/best.pt)",
        "reid": "osnet_x1_0 fine-tuned on 4 train clips (reid_osnet/best_osnet_soccer.pt)",
        "tracker": "Tracker v3 (Kit-Aware Deep-EIoU + GTA-Link)",
        "frozen_params": FROZEN_FINAL_PARAMS,
        "disclosure": "128057 was scored once in Phase I on the frozen Phase I pipeline; 132831 (3840x1504) tests resolution shift.",
        "sequences": meta,
        "results": res,
        "environment": _env(args.manifest),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")

    te = res["trackeval"] or res["reference"]
    src = "TrackEval" if res["trackeval"] else f"NumPy reference ({res['trackeval_error']})"
    print(f"[{src}] Frozen Final Test Results ({args.role or args.clip}):")
    for sname, m in te["per_seq"].items():
        fp_s = f" FP {m['FP']:.0f} FN {m['FN']:.0f} IDF1 {m['IDF1']:.4f}" if "FP" in m else ""
        print(f"  {sname:<18} HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} LocA {m['LocA']:.4f}{fp_s} IDSW {m['IDSW']:.0f}")
    mm = te["mean_over_seqs"]
    fp_s = f" FP {mm['FP']:.1f} FN {mm['FN']:.1f} IDF1 {mm['IDF1']:.4f}" if "FP" in mm else ""
    print(f"  {'MEAN':<18} HOTA {mm['HOTA']:.4f} DetA {mm['DetA']:.4f} AssA {mm['AssA']:.4f} LocA {mm['LocA']:.4f}{fp_s} IDSW {mm['IDSW']:.1f}")
    print(f"-> {out}")
    return record


def build_parser() -> argparse.ArgumentParser:
    from .data import DEFAULT_MANIFEST
    from .phase2 import DEFAULT_DET_ROOT

    p = argparse.ArgumentParser(description="Score the frozen Phase II final config on held-out test clips.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--role", choices=("val", "test", "reserve"), default="test")
    p.add_argument("--clip", nargs="+", default=None)
    p.add_argument("--run", required=True)
    p.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT)
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--allow-reserve", action="store_true")
    p.add_argument("--no-trackeval", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results") / "p2_final_test.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
