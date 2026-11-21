"""C4 roster-cap tuning and confirmation (validation clips only). Protocol: docs/phase2_amendment_c4_roster.md.

    python -m soccer_tracking.tune_roster sweep    --run p2more --det-root <dir> --out results/c4_sweep.json
    python -m soccer_tracking.tune_roster finalize --sweep results/c4_sweep.json --out results/c4_tuning.json
    python -m soccer_tracking.tune_roster confirm  --run p2more_w3 --det-root <dir> --params results/c4_tuning.json

``finalize`` and ``confirm`` need TrackEval on PYTHONPATH. Not unit-tested (needs real cached detections).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .roster import GRID_C4, build_arm_roster, pick_final_c4
from .tune_tracker_v2 import _env, _write, load_sequences, score_params

V2_PARAMS = Path("results") / "tracker_v2_tuning.json"


def _v2(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["chosen"]


def cmd_sweep(args) -> None:
    v2 = _v2(args.v2_params)
    seqs = load_sequences(args.manifest, args.det_root, args.run, args.clip)
    rows = []
    for n, (mode, k) in enumerate(GRID_C4, start=1):
        m = score_params(seqs, lambda d, mo=mode, kk=k: build_arm_roster(d, v2, mo, kk), False)["reference"]["mean_over_seqs"]
        rows.append({"mode": mode, "k": k, "hota": m["HOTA"], "deta": m["DetA"], "assa": m["AssA"], "idsw": m["IDSW"]})
        print(f"[{n}/{len(GRID_C4)}] {mode} k={k}  HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} IDSW {m['IDSW']:.1f}", flush=True)
    ranked = sorted(rows, key=lambda r: (-r["hota"], r["idsw"]))
    cands = ranked[:3] + [r for r in rows if r["mode"] == "none" and r not in ranked[:3]]
    _write(Path(args.out), {"protocol": "docs/phase2_amendment_c4_roster.md", "run": args.run, "det_root": str(args.det_root),
                            "v2_params": v2, "windows": {n: {k: s[k] for k in ("clip_id", "start", "end")} for n, s in seqs.items()},
                            "candidates": cands, "grid_results": rows, "environment": _env(args.manifest)})
    print(f"-> {args.out}")


def cmd_finalize(args) -> None:
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    seqs = load_sequences(args.manifest, Path(sweep["det_root"]), sweep["run"], None)
    rescored = []
    for c in sweep["candidates"]:
        te = score_params(seqs, lambda d, mo=c["mode"], kk=c["k"]: build_arm_roster(d, sweep["v2_params"], mo, kk), True)["trackeval"]
        if te is None:
            raise ImportError("TrackEval is required for `finalize` (put it on PYTHONPATH)")
        m = te["mean_over_seqs"]
        rescored.append({"mode": c["mode"], "k": c["k"], "hota": m["HOTA"], "deta": m["DetA"], "assa": m["AssA"],
                         "fp": m["FP"], "fn": m["FN"], "idsw": m["IDSW"], "reference_hota": c["hota"]})
        print(f"TrackEval {c['mode']} k={c['k']}: HOTA {m['HOTA']:.4f} AssA {m['AssA']:.4f} FP {m['FP']:.1f} FN {m['FN']:.1f} IDSW {m['IDSW']:.1f}")
    chosen = pick_final_c4(rescored)
    _write(Path(args.out), {"protocol": "docs/phase2_amendment_c4_roster.md", "chosen": {"mode": chosen["mode"], "k": chosen["k"]},
                            "v2_params": sweep["v2_params"], "rescored": rescored, "c4_rejected": chosen["mode"] == "none",
                            "rule": "best TrackEval mean HOTA; within 0.005 -> lower IDSW, then larger k", "environment": _env(args.manifest)})
    print(f"chosen {chosen['mode']} k={chosen['k']}" + ("  (C4 rejected: no cap won)" if chosen["mode"] == "none" else "") + f" -> {args.out}")


def cmd_confirm(args) -> None:
    tuning = json.loads(Path(args.params).read_text(encoding="utf-8"))
    if tuning["chosen"]["mode"] == "none":
        raise ValueError("C4 was rejected at tuning (no cap won); there is nothing to confirm")
    v2, ch = tuning["v2_params"], tuning["chosen"]
    seqs = load_sequences(args.manifest, args.det_root, args.run, args.clip)
    res = {}
    for label, mode, k in (("v2_no_cap", "none", None), ("v2_with_cap", ch["mode"], ch["k"])):
        r = score_params(seqs, lambda d, mo=mode, kk=k: build_arm_roster(d, v2, mo, kk), True)
        if r["trackeval"] is None:
            raise ImportError("TrackEval is required for `confirm` (put it on PYTHONPATH)")
        res[label] = r
    a, b = (res[x]["trackeval"]["mean_over_seqs"] for x in ("v2_no_cap", "v2_with_cap"))
    crit = {"delta_hota": b["HOTA"] - a["HOTA"], "delta_assa": b["AssA"] - a["AssA"], "delta_fp": b["FP"] - a["FP"],
            "min_delta_hota": 0.005, "hota_ok": b["HOTA"] - a["HOTA"] >= 0.005, "assa_ok": b["AssA"] >= a["AssA"], "fp_ok": b["FP"] <= a["FP"]}
    _write(Path(args.out), {"protocol": "docs/phase2_amendment_c4_roster.md", "run": args.run, "chosen": ch, "v2_params": v2,
                            "windows": {n: {k: s[k] for k in ("clip_id", "start", "end")} for n, s in seqs.items()},
                            "results": res, "criteria": crit, "environment": _env(args.manifest)})
    for label in res:
        m = res[label]["trackeval"]["mean_over_seqs"]
        print(f"{label:<12} HOTA {m['HOTA']:.4f} DetA {m['DetA']:.4f} AssA {m['AssA']:.4f} FP {m['FP']:.1f} FN {m['FN']:.1f} IDSW {m['IDSW']:.1f}")
    print(f"dHOTA {crit['delta_hota']:+.4f} (need >= +0.005): {'PASS' if crit['hota_ok'] else 'FAIL'}; dAssA {crit['delta_assa']:+.4f}: "
          f"{'PASS' if crit['assa_ok'] else 'FAIL'}; dFP {crit['delta_fp']:+.1f}: {'PASS' if crit['fp_ok'] else 'FAIL'} -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    from .data import DEFAULT_MANIFEST
    from .phase2 import DEFAULT_DET_ROOT

    p = argparse.ArgumentParser(description="C4 roster cap: pre-registered sweep, TrackEval rescoring and confirmation (validation only).")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("sweep", "finalize", "confirm"):
        sub.add_parser(name).add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    for name in ("sweep", "confirm"):
        s = sub.choices[name]
        s.add_argument("--run", required=True)
        s.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT)
        s.add_argument("--clip", nargs="+", default=None)
    sub.choices["sweep"].add_argument("--v2-params", type=Path, default=V2_PARAMS)
    sub.choices["sweep"].add_argument("--out", type=Path, default=Path("results") / "c4_sweep.json")
    f = sub.choices["finalize"]
    f.add_argument("--sweep", type=Path, default=Path("results") / "c4_sweep.json")
    f.add_argument("--out", type=Path, default=Path("results") / "c4_tuning.json")
    c = sub.choices["confirm"]
    c.add_argument("--params", type=Path, default=Path("results") / "c4_tuning.json")
    c.add_argument("--out", type=Path, default=Path("results") / "c4_confirm.json")
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