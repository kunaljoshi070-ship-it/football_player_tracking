"""Patch src/soccer_tracking/train.py for Phase II (idempotent, fails loudly if the file is not the expected Phase I text).

Adds: --train-clips / --val-clips (explicit clip ids, roles enforced by phase2.training_clips) and --select-metric
(ap50 [default, Phase I behaviour], ap75 or mean) for early stopping and the best checkpoint. Defaults leave Phase I behaviour unchanged.

    python tools/apply_train_patch.py            # edits src/soccer_tracking/train.py in place
    git diff src/soccer_tracking/train.py        # review it
"""
import sys
from pathlib import Path

PATH = Path("src/soccer_tracking/train.py")
EDITS = [
("""    p.add_argument("--seed", type=int, default=0)
""",
"""    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-clips", nargs="+", default=None, help="Phase II: explicit training clip ids (role 'train')")
    p.add_argument("--val-clips", nargs="+", default=None, help="Phase II: explicit validation clip ids (role 'val')")
    p.add_argument("--select-metric", choices=("ap50", "ap75", "mean"), default="ap50",
                   help="validation metric for early stopping and best.pt (mean = (AP50 + AP75) / 2)")
"""),
("""def run(args: argparse.Namespace) -> dict:
""",
"""def select_value(val: dict, metric: str) -> float:
    \"\"\"The validation number that decides 'best' and early stopping.\"\"\"
    if metric == "ap50":
        return val["ap50"]
    if metric == "ap75":
        return val["ap75"]
    return 0.5 * (val["ap50"] + val["ap75"])


def run(args: argparse.Namespace) -> dict:
"""),
("""    train_clips = load_manifest(args.manifest, ["train"])
    val_clips = load_manifest(args.manifest, ["val"])
""",
"""    if args.train_clips or args.val_clips:
        from .phase2 import training_clips
        train_clips, val_clips = training_clips(args.manifest, args.train_clips, args.val_clips)
    else:
        train_clips = load_manifest(args.manifest, ["train"])
        val_clips = load_manifest(args.manifest, ["val"])
"""),
("""        improved = val["ap50"] > best + 1e-4
        if improved:
            best, bad = val["ap50"], 0
""",
"""        sel = select_value(val, args.select_metric)
        improved = sel > best + 1e-4
        if improved:
            best, bad = sel, 0
"""),
("""    best_row = max(rows, key=lambda r: float(r["val_ap50"])) if rows else {}
""",
"""    best_row = max(rows, key=lambda r: select_value({"ap50": float(r["val_ap50"]), "ap75": float(r["val_ap75"])}, args.select_metric)) if rows else {}
"""),
]


def apply(text: str) -> str:
    if "--select-metric" in text:
        raise SystemExit("train.py already patched")
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"expected exactly one occurrence of:\n{old}\nfound {text.count(old)}; train.py is not the expected Phase I file")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    PATH.write_text(apply(PATH.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"patched {PATH}")
