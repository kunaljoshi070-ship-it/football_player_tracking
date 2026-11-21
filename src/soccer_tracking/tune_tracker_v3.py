"""Tracker v3 ablation, sweep, and TrackEval finalization (Phase II, WS-C2/C3/C4).

Protocol: docs/phase2_amendment_tracker_v3.md. All commands run on CPU using cached detections + extracted features.

Commands::

    # 1. Incremental ablation on full validation clips (TrackEval)
    PYTHONPATH=src:TrackEval python -m soccer_tracking.tune_tracker_v3 ablate \
        --run p2more_full --det-root /content/drive/MyDrive/soccer_runs/detections_p2 \
        --out results/tracker_v3_ablation.json

    # 2. Compact 16-setting sweep (NumPy reference HOTA, fast on CPU)
    PYTHONPATH=src python -m soccer_tracking.tune_tracker_v3 sweep \
        --run p2more_full --det-root /content/drive/MyDrive/soccer_runs/detections_p2 \
        --out results/tracker_v3_sweep.json

    # 3. TrackEval rescore of top-3 -> results/tracker_v3_tuning.json
    PYTHONPATH=src:TrackEval python -m soccer_tracking.tune_tracker_v3 finalize \
        --sweep results/tracker_v3_sweep.json --out results/tracker_v3_tuning.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .tracker_v3 import DEFAULT_V3_PARAMS, grid_v3, pick_final_v3, run_tracker_v3


def load_sequences_with_features(
    manifest: Path,
    det_root: Path,
    run: str,
    clip_ids: Optional[Sequence[str]] = None,
    require_features: bool = True,
) -> Dict[str, dict]:
    from .data import check_annotations, load_manifest
    from .features import assemble_features
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
            raise FileNotFoundError(f"no detections for clip {cid} in {det_dir}")
        info = json.loads(pf.read_text(encoding="utf-8"))
        plan = [tuple(p) for p in info["plan"]]
        clip, n = by_id[cid], info["end"] - info["start"]
        ann = load_mot(clip.annotation_path)
        problems = check_annotations(clip, ann)
        if problems:
            raise ValueError("; ".join(problems))
        dets = assemble(det_dir, cid, plan)
        feats = None
        if require_features:
            feats, _ = assemble_features(det_dir, cid, plan, dets)
        seqs[f"{cid}_{info['start']}_{info['end']}"] = {
            "gt": gt_sequence(clip, ann, info["start"], n),
            "dets": dets,
            "features": feats,
            "width": clip.width,
            "height": clip.height,
            "clip_id": cid,
            "start": info["start"],
            "end": info["end"],
        }
    return seqs


def _score_v3(seqs: Dict[str, dict], params: dict, use_trackeval: bool) -> dict:
    from .hota_eval import evaluate_sequences

    arm = {}
    for name, s in seqs.items():
        preds, sc = run_tracker_v3(s["dets"], s["features"], s["width"], s["height"], params)
        arm[name] = {"gt": s["gt"], "pred": preds, "scores": sc, "width": s["width"], "height": s["height"]}
    return evaluate_sequences(arm, use_trackeval=use_trackeval)


def _write(out: Path, record: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")


def cmd_ablate(args) -> None:
    from .track_eval import _env

    seqs = load_sequences_with_features(args.manifest, args.det_root, args.run, args.clip, require_features=not args.no_features)
    base = dict(DEFAULT_V3_PARAMS)
    if args.params:
        base.update(json.loads(Path(args.params).read_text(encoding="utf-8"))["chosen"])

    arms = [
        ("v3_eiou_only", {**base, "app_weight": 0.0, "use_gta": False, "use_official_filter": False, "roster_k": None}),
        ("v3_deep_eiou", {**base, "use_gta": False, "use_official_filter": False, "roster_k": None}),
        ("v3_gta", {**base, "use_gta": True, "use_official_filter": False, "roster_k": None}),
        ("v3_full", {**base, "use_gta": True, "use_official_filter": True}),
    ]
    results = {}
    for label, p in arms:
        r = _score_v3(seqs, p, use_trackeval=not args.no_trackeval)
        results[label] = {"params": p, "results": r}
        m = (r["trackeval"] or r["reference"])["mean_over_seqs"]
        fp_s = f" FP {m['FP']:.1f} FN {m['FN']:.1f}" if "FP" in m else ""
        print(f"{label:<14} HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} "
              f"LocA {m['LocA']:.4f}{fp_s} IDSW {m['IDSW']:.1f}", flush=True)

    _write(
        Path(args.out),
        {
            "protocol": "docs/phase2_amendment_tracker_v3.md",
            "run": args.run,
            "det_root": str(args.det_root),
            "sequences": {n: {k: s[k] for k in ("clip_id", "start", "end")} for n, s in seqs.items()},
            "arms": results,
            "environment": _env(args.manifest),
        },
    )
    print(f"-> {args.out}")


def cmd_sweep(args) -> None:
    from .track_eval import _env

    seqs = load_sequences_with_features(args.manifest, args.det_root, args.run, args.clip, require_features=not args.no_features)
    grid = list(grid_v3())
    rows = []
    for k, params in enumerate(grid, start=1):
        m = _score_v3(seqs, params, use_trackeval=False)["reference"]["mean_over_seqs"]
        rows.append({"params": params, "hota": m["HOTA"], "deta": m["DetA"], "assa": m["AssA"],
                     "loca": m["LocA"], "idsw": m["IDSW"]})
        print(f"[{k}/{len(grid)}] det_high={params['det_high']} cos={params['merge_cos_thresh']} "
              f"lab={params['official_lab_thresh']} k={params['roster_k']}  "
              f"HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} IDSW {m['IDSW']:.1f}", flush=True)

    ranked = sorted(rows, key=lambda r: (-r["hota"], r["idsw"]))
    _write(
        Path(args.out),
        {
            "protocol": "docs/phase2_amendment_tracker_v3.md",
            "run": args.run,
            "det_root": str(args.det_root),
            "sequences": {n: {k: s[k] for k in ("clip_id", "start", "end")} for n, s in seqs.items()},
            "top_k": args.top_k,
            "candidates": ranked[:args.top_k],
            "grid_results": rows,
            "environment": _env(args.manifest),
        },
    )
    print(f"-> {args.out}")


def cmd_finalize(args) -> None:
    from .track_eval import _env

    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    seqs = load_sequences_with_features(args.manifest, Path(sweep["det_root"]), sweep["run"], None, require_features=True)
    rescored = []
    for cand in sweep["candidates"]:
        te = _score_v3(seqs, cand["params"], use_trackeval=True)["trackeval"]
        if te is None:
            raise ImportError("TrackEval is required for `finalize` (put it on PYTHONPATH)")
        m = te["mean_over_seqs"]
        rescored.append({
            "params": cand["params"],
            "hota": m["HOTA"], "deta": m["DetA"], "assa": m["AssA"], "loca": m["LocA"],
            "fp": m["FP"], "fn": m["FN"], "idsw": m["IDSW"], "idf1": m["IDF1"],
            "reference_hota": cand["hota"],
        })
        print(f"TrackEval HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} "
              f"FP {m['FP']:.1f} FN {m['FN']:.1f} IDSW {m['IDSW']:.1f}")
    chosen = pick_final_v3(rescored)
    _write(
        Path(args.out),
        {
            "protocol": "docs/phase2_amendment_tracker_v3.md",
            "tracker": "DeepEIoU+GTALink+KitColorFilter",
            "chosen": chosen["params"],
            "tuning_run": sweep["run"],
            "sequences": sweep["sequences"],
            "rescored_top_k": rescored,
            "rule": "best TrackEval mean HOTA; within 0.005 -> lowest FP, then lowest IDSW",
            "environment": _env(args.manifest),
        },
    )
    print(f"chosen -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    from .data import DEFAULT_MANIFEST
    from .phase2 import DEFAULT_DET_ROOT

    p = argparse.ArgumentParser(description="Tracker v3 ablation, sweep, and TrackEval finalization (validation clips only).")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ablate", "sweep", "finalize"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    for name in ("ablate", "sweep"):
        s = sub.choices[name]
        s.add_argument("--run", required=True)
        s.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT)
        s.add_argument("--clip", nargs="+", default=None)
        s.add_argument("--no-features", action="store_true", help="run without extracted embeddings/Lab features (spatial only)")
    ab = sub.choices["ablate"]
    ab.add_argument("--params", type=Path, default=None, help="optional tracker_v3_tuning.json to ablate around")
    ab.add_argument("--no-trackeval", action="store_true")
    ab.add_argument("--out", type=Path, default=Path("results") / "tracker_v3_ablation.json")
    sw = sub.choices["sweep"]
    sw.add_argument("--top-k", type=int, default=3)
    sw.add_argument("--out", type=Path, default=Path("results") / "tracker_v3_sweep.json")
    fi = sub.choices["finalize"]
    fi.add_argument("--sweep", type=Path, default=Path("results") / "tracker_v3_sweep.json")
    fi.add_argument("--out", type=Path, default=Path("results") / "tracker_v3_tuning.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        {"ablate": cmd_ablate, "sweep": cmd_sweep, "finalize": cmd_finalize}[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
