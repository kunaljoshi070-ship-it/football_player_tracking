"""Score tracker output with TrackEval (the reference tool) and with our NumPy HOTA, side by side.

TrackEval is MIT-licensed (github.com/JonathonLuiten/TrackEval). It is not on PyPI under a stable name: clone it and
put the folder on ``PYTHONPATH`` (it needs numpy and scipy). Numbers for the report and for the GTATrack comparison
must come from TrackEval; ``hota.py`` is the cross-check.

Run the one-off agreement test first (about a second, no data needed)::

    PYTHONPATH=src:TrackEval python -m soccer_tracking.hota_eval selftest

What is reported (to match how GTATrack's leaderboard numbers look: FN/FP end in .5, i.e. means of integer counts over two videos):
* ``HOTA, DetA, AssA, LocA``: mean over the 19 alpha values, from TrackEval's HOTA metric;
* ``FP, FN, IDSW``: integer counts from TrackEval's CLEAR metric (IoU 0.5), ``IDF1`` from Identity;
* a **per-sequence mean** (the Challenge protocol) as well as TrackEval's pooled ``COMBINED_SEQ`` value.
The structure of TrackEval's result dictionary is written from memory of its source: ``parse_trackeval`` is unit-tested only on
a mock of that structure, so the selftest on a real install is what confirms it.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import hota as ref
from .mot_export import write_gt, write_seqmap, write_tracker

HOTA_KEYS = ("HOTA", "DetA", "AssA", "LocA")


def _numpy_compat() -> None:
    """TrackEval's older code uses ``np.float``/``np.int``/``np.bool``, removed in NumPy >= 1.24 (Colab ships NumPy 2.x).
    Restoring the aliases is harmless and avoids an AttributeError inside TrackEval."""
    for name, real in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if name not in np.__dict__:
            setattr(np, name, real)


def import_trackeval():
    _numpy_compat()
    try:
        import trackeval  # type: ignore
    except ImportError as exc:
        raise ImportError("TrackEval is not importable: clone https://github.com/JonathonLuiten/TrackEval and add it to "
                          "PYTHONPATH (needs numpy and scipy)") from exc
    return trackeval


def run_trackeval(gt_root: Path, trackers_root: Path, tracker: str, seqs: Sequence[str]) -> dict:
    """Run TrackEval (HOTA, CLEAR, Identity; pedestrian class) on the layout written by ``mot_export`` and return its raw result dict."""
    trackeval = import_trackeval()
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "PRINT_ONLY_COMBINED": False,
                     "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root), "TRACKERS_TO_EVAL": [tracker],
                   "SEQMAP_FILE": str(Path(gt_root) / "seqmap.txt"), "SKIP_SPLIT_FOL": True, "CLASSES_TO_EVAL": ["pedestrian"],
                   "PRINT_CONFIG": False})
    evaluator = trackeval.Evaluator(eval_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    return res


def _one(entry: dict) -> Dict[str, float]:
    """Flatten one TrackEval per-sequence entry (``entry['pedestrian']``) into plain floats."""
    h, c, i = entry["HOTA"], entry["CLEAR"], entry["Identity"]
    out = {k: float(np.mean(h[k])) for k in HOTA_KEYS}
    if max(out.values()) > 1.0 + 1e-6:
        raise ValueError(f"HOTA values {out} are not fractions in [0, 1]; TrackEval's output scale is not what this parser assumes")
    out.update({"FP": float(c["CLR_FP"]), "FN": float(c["CLR_FN"]), "IDSW": float(c["IDSW"]), "IDF1": float(i["IDF1"])})
    return out


def parse_trackeval(res: dict, tracker: str, seqs: Sequence[str]) -> dict:
    """``{'per_seq': {seq: metrics}, 'mean_over_seqs': metrics, 'pooled': metrics}`` from TrackEval's raw result dict."""
    node = res["MotChallenge2DBox"][tracker]
    per_seq = {s: _one(node[s]["pedestrian"]) for s in seqs}
    keys = next(iter(per_seq.values())).keys()
    mean = {k: float(np.mean([per_seq[s][k] for s in seqs])) for k in keys}
    return {"per_seq": per_seq, "mean_over_seqs": mean, "pooled": _one(node["COMBINED_SEQ"]["pedestrian"])}


def evaluate_sequences(seqs: Dict[str, dict], work: Optional[Path] = None, use_trackeval: bool = True,
                       tracker_name: str = "mine") -> dict:
    """``seqs[name] = {'gt': frames, 'pred': frames, 'scores': per-frame score arrays or None, 'width': int, 'height': int}``
    (frames are ``(ids, boxes_xyxy)`` lists, 0-based). Returns reference (NumPy) results always and TrackEval results if available."""
    names = list(seqs)
    parts = {n: ref.hota_sequence(seqs[n]["gt"], seqs[n]["pred"]) for n in names}
    per_seq = {}
    for n in names:
        r = ref.combine([parts[n]])
        r["IDSW"] = ref.id_switches(seqs[n]["gt"], seqs[n]["pred"])
        per_seq[n] = r
    keys = ("HOTA", "DetA", "AssA", "LocA", "IDSW")
    out = {"reference": {"per_seq": per_seq, "mean_over_seqs": {k: float(np.mean([per_seq[n][k] for n in names])) for k in keys},
                         "pooled": ref.combine(list(parts.values()))}, "trackeval": None, "trackeval_error": None}
    if not use_trackeval:
        return out
    tmp = None
    if work is None:
        tmp = tempfile.TemporaryDirectory()
        work = Path(tmp.name)
    try:
        gt_root, trk_root = Path(work) / "gt", Path(work) / "trackers"
        for n in names:
            s = seqs[n]
            write_gt(gt_root, n, s["gt"], s["width"], s["height"])
            write_tracker(trk_root, tracker_name, n, s["pred"], s.get("scores"))
        write_seqmap(gt_root, names)
        out["trackeval"] = parse_trackeval(run_trackeval(gt_root, trk_root, tracker_name, names), tracker_name, names)
    except ImportError as exc:
        out["trackeval_error"] = str(exc)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return out


# --------------------------------------------------------------------------- self-test
def _scene(n: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 9)
    x0, vx = 150.0 * ids, rng.uniform(-3, 3, len(ids))
    gt = []
    for f in range(n):
        x = x0 + vx * f
        gt.append((ids, np.stack([x, np.full(len(ids), 200.0), x + 20, np.full(len(ids), 245.0)], 1)))
    pred = []
    for f, (i, b) in enumerate(gt):
        keep = rng.random(len(i)) > 0.08                                   # misses
        jitter = rng.normal(0, 1.5, b.shape)                               # localisation noise
        pid = np.where(i % 3 == 0, i + (f // 25) * 100, i)                 # some ids change over time
        bb = np.vstack([(b + jitter)[keep], [[900 + f % 5, 400, 920, 445]]])   # one false positive per frame
        pred.append((np.append(pid[keep], 999), bb))
    return gt, pred, [np.full(len(p[0]), 0.9) for p in pred]


def selftest(tol: float = 0.01) -> int:
    seqs = {}
    for name, seed in (("synthA", 1), ("synthB", 2)):
        gt, pred, sc = _scene(120, seed)
        seqs[name] = {"gt": gt, "pred": pred, "scores": sc, "width": 1200, "height": 800}
    res = evaluate_sequences(seqs)
    if res["trackeval"] is None:
        print(f"TrackEval not available: {res['trackeval_error']}\nreference only: {res['reference']['mean_over_seqs']}")
        return 2
    bad = 0
    print(f"{'sequence':<9}{'metric':<7}{'reference':>11}{'TrackEval':>11}{'diff':>9}")
    for n in seqs:
        for k in HOTA_KEYS:
            a, b = res["reference"]["per_seq"][n][k], res["trackeval"]["per_seq"][n][k]
            flag = "" if abs(a - b) <= tol else "  <-- MISMATCH"
            bad += bool(flag)
            print(f"{n:<9}{k:<7}{a:>11.4f}{b:>11.4f}{a - b:>9.4f}{flag}")
        print(f"{n:<9}{'IDSW':<7}{res['reference']['per_seq'][n]['IDSW']:>11}{res['trackeval']['per_seq'][n]['IDSW']:>11.0f}   (informational: different matching rules)")
    print("AGREE within tolerance" if not bad else f"{bad} metric(s) disagree: do not trust either until resolved")
    return 1 if bad else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="TrackEval wrapper utilities.")
    p.add_argument("command", choices=("selftest",))
    p.add_argument("--tol", type=float, default=0.01)
    args = p.parse_args(argv)
    return selftest(args.tol)


if __name__ == "__main__":
    sys.exit(main())
