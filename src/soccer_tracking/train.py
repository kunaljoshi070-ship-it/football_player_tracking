"""Fine-tune a COCO-pretrained RetinaNet (ResNet-50 FPN) on the Phase I train tiles (one class: player).

Meant for a GPU (see README, "Training on a hosted GPU"); it also runs on CPU for smoke tests::

    PYTHONPATH=src python -m soccer_tracking.train --run-name finetune --epochs 8

Outputs in ``runs/<run-name>/``: ``config_<utc>.json`` (all settings + environment), ``iterations.csv``
and ``metrics.csv`` (learning curves), ``curves.png``, ``last.pt`` / ``best.pt`` (git-ignored) and
``summary.json``. Early stopping and "best" are decided on validation AP50 only; the test split is
never touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .data import DEFAULT_FRAMES_ROOT, DEFAULT_MANIFEST, ManifestError, SoccerTilesDataset, collate, load_manifest
from .detect import adapt_head, build_retinanet
from .evaluate import environment_record, evaluate_model
from .mot import MotFormatError

ITER_COLS = ["iteration", "epoch", "loss", "classification", "bbox_regression", "lr"]
EPOCH_COLS = ["epoch", "train_loss", "train_classification", "train_bbox_regression", "val_ap50", "val_ap75",
              "val_precision", "val_recall", "val_f1", "lr_end", "seconds"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune RetinaNet on the Phase I train split.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--run-name", default="finetune")
    p.add_argument("--init", choices=("coco", "none"), default="coco", help="'none' = random init (smoke tests only)")
    p.add_argument("--stride", type=int, default=25, help="train on every Nth extracted frame")
    p.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4, help="lower to 2 if the GPU runs out of memory")
    p.add_argument("--lr", type=float, default=0.005, help="peak SGD learning rate")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--clip-grad", type=float, default=10.0, help="max gradient norm")
    p.add_argument("--patience", type=int, default=3, help="stop after this many epochs without val AP50 improvement")
    p.add_argument("--val-stride", type=int, default=125, help="validate on every Nth frame (multiple of the extraction stride)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="mixed precision (CUDA only)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=20, help="write an iterations.csv row every N iterations")
    p.add_argument("--max-iters-per-epoch", type=int, default=None, help="cap iterations per epoch (smoke tests)")
    p.add_argument("--resume", type=Path, default=None, help="continue from a last.pt in the same run directory")
    p.add_argument("--overwrite", action="store_true", help="allow a non-empty run directory without --resume")
    return p


def _read_csv(path: Path) -> list:
    """Rows of a CSV file as dicts ([] if the file does not exist); the file is always closed."""
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, cols: Sequence[str], row: dict) -> None:
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cols))
        if new:
            writer.writeheader()
        writer.writerow(row)


def lr_at(iteration: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 1% of the peak."""
    warm = min(1.0, (iteration + 1) / max(1, warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, iteration / max(1, total))))
    return peak * warm * max(0.01, cosine)


def plot_curves(out_dir: Path) -> Optional[Path]:
    """Save ``curves.png``: training loss per logged iteration and validation AP per epoch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping curves.png (the CSV files have the data)")
        return None
    its, eps = _read_csv(out_dir / "iterations.csv"), _read_csv(out_dir / "metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if its:
        axes[0].plot([float(r["iteration"]) for r in its], [float(r["loss"]) for r in its], label="total")
        axes[0].plot([float(r["iteration"]) for r in its], [float(r["classification"]) for r in its], label="classification", alpha=0.6)
        axes[0].plot([float(r["iteration"]) for r in its], [float(r["bbox_regression"]) for r in its], label="box regression", alpha=0.6)
    axes[0].set(title="Training loss", xlabel="iteration", ylabel="loss")
    axes[0].legend()
    if eps:
        e = [int(r["epoch"]) for r in eps]
        axes[1].plot(e, [float(r["val_ap50"]) for r in eps], marker="o", label="val AP50")
        axes[1].plot(e, [float(r["val_ap75"]) for r in eps], marker="o", label="val AP75")
    axes[1].set(title="Validation", xlabel="epoch", ylabel="AP")
    axes[1].legend()
    fig.tight_layout()
    path = out_dir / "curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader

    out_dir = Path(args.out_dir) / args.run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume and not args.overwrite:
        raise ValueError(f"{out_dir} is not empty; use --resume, --overwrite or another --run-name "
                         "(mixing runs would corrupt the learning curves)")
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    use_amp = bool(args.amp and device.type == "cuda")
    tile = tuple(args.tile)

    train_clips = load_manifest(args.manifest, ["train"])
    val_clips = load_manifest(args.manifest, ["val"])
    dataset = SoccerTilesDataset(train_clips, args.frames_root, stride=args.stride, tile=tile,
                                 overlap=args.overlap, augment=True, seed=args.seed)
    gen = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, generator=gen)
    iters_per_epoch = len(loader) if not args.max_iters_per_epoch else min(len(loader), args.max_iters_per_epoch)
    total_iters = args.epochs * iters_per_epoch

    model = adapt_head(build_retinanet(args.init, tile, score_thresh=0.05), 2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, iteration, best, bad = 0, 0, -1.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, iteration, best, bad = ckpt["epoch"], ckpt["iteration"], ckpt["best_ap50"], ckpt["bad_epochs"]
        print(f"resumed from {args.resume}: starting epoch {start_epoch + 1}, iteration {iteration}")

    config = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "train_clips": [c.clip_id for c in train_clips], "val_clips": [c.clip_id for c in val_clips],
              "train_tiles": len(dataset), "iterations_per_epoch": iters_per_epoch, "amp_used": use_amp,
              "trainable_parameters": int(sum(p.numel() for p in params)),
              "notes": "COCO_V1 weights, classification output layer re-initialised for 2 classes; torchvision "
                       "default frozen BatchNorm and trainable backbone layers; GPU runs are not bit-exact reproducible",
              "environment": environment_record(args.manifest, device)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"config_{stamp}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"train tiles {len(dataset)} ({len(train_clips)} clips), {iters_per_epoch} iterations/epoch, device {device}, amp {use_amp}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        dataset.set_epoch(epoch)
        gen.manual_seed(args.seed + epoch)
        t0 = time.time()
        sums = {"loss": 0.0, "classification": 0.0, "bbox_regression": 0.0}
        n = 0
        lr = args.lr
        for batch_i, (images, targets) in enumerate(loader):
            if args.max_iters_per_epoch and batch_i >= args.max_iters_per_epoch:
                break
            lr = lr_at(iteration, total_iters, args.lr, args.warmup_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at iteration {iteration}; lower --lr or check the data")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            vals = {"loss": loss.item(), "classification": losses["classification"].item(),
                    "bbox_regression": losses["bbox_regression"].item()}
            for k in sums:
                sums[k] += vals[k]
            n += 1
            iteration += 1
            if iteration % args.log_every == 0 or batch_i == 0:
                _append_row(out_dir / "iterations.csv", ITER_COLS, {"iteration": iteration, "epoch": epoch + 1, "lr": lr, **vals})
                print(f"epoch {epoch + 1} iter {batch_i + 1}/{iters_per_epoch} loss {vals['loss']:.4f} "
                      f"lr {lr:.5f} {(time.time() - t0) / n:.2f} s/iter", flush=True)

        val, _ = evaluate_model(model, val_clips, args.frames_root, args.val_stride, tile, args.overlap, device, verbose=False)
        row = {"epoch": epoch + 1, "train_loss": sums["loss"] / max(1, n),
               "train_classification": sums["classification"] / max(1, n),
               "train_bbox_regression": sums["bbox_regression"] / max(1, n), "val_ap50": val["ap50"],
               "val_ap75": val["ap75"], "val_precision": val.get("precision", float("nan")),
               "val_recall": val.get("recall", float("nan")), "val_f1": val.get("f1", float("nan")),
               "lr_end": lr, "seconds": time.time() - t0}
        _append_row(out_dir / "metrics.csv", EPOCH_COLS, row)
        improved = val["ap50"] > best + 1e-4
        if improved:
            best, bad = val["ap50"], 0
        else:
            bad += 1
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "iteration": iteration, "best_ap50": best, "bad_epochs": bad}
        torch.save(state, out_dir / "last.pt")
        if improved:
            torch.save(state, out_dir / "best.pt")
        print(f"epoch {epoch + 1}: train loss {row['train_loss']:.4f}  val AP50 {val['ap50']:.4f} AP75 {val['ap75']:.4f} "
              f"P {row['val_precision']:.3f} R {row['val_recall']:.3f}{'  (best)' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"early stopping: no val AP50 improvement for {bad} epochs")
            break

    rows = _read_csv(out_dir / "metrics.csv")
    best_row = max(rows, key=lambda r: float(r["val_ap50"])) if rows else {}
    summary = {"best_epoch": int(best_row["epoch"]) if best_row else None, "best_val_ap50": best,
               "epochs_run": len(rows), "iterations": iteration, "run_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot = plot_curves(out_dir)
    print(f"done: {summary}" + (f"; curves -> {plot}" if plot else ""))
    return summary


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
