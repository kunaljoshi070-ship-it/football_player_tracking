"""Domain-adapted OSNet ReID fine-tuning on SoccerTrack training clips (Phase II, WS-C2).

Trains ``osnet_x1_0`` with Batch-Hard Triplet Loss + Label-Smoothed Cross-Entropy (Bag-of-Tricks ReID)
exclusively on the 4 Phase II training clips (118575, 118576, 118577, 117092; 88 unique player identities).
Validation, test, and reserve clips are never touched here.

Command line (Colab T4, ~12-15 min)::

    PYTHONPATH=src:/content/deep-person-reid python3 -m soccer_tracking.train_reid \
        --out-dir /content/drive/MyDrive/soccer_runs/reid_osnet \
        --stride 25 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import DEFAULT_FRAMES_ROOT, DEFAULT_MANIFEST, frame_path
from .features import crop_box_rgb
from .mot import load_mot
from .phase2 import PHASE2_ROLES, training_clips


def batch_hard_triplet_loss(embs, labels, margin: float = 0.3):
    """Batch-hard triplet loss on L2-normalized embeddings (PyTorch tensor ``(B, D)``)."""
    import torch
    import torch.nn.functional as F

    embs = F.normalize(embs, p=2, dim=1)
    # Cosine distance in [0, 2]
    dist = torch.clamp(1.0 - torch.matmul(embs, embs.t()), min=0.0)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff = ~same
    # Hardest positive: max distance among same-id samples
    pos_dist = (dist * same.float()).max(dim=1).values
    # Hardest negative: min distance among different-id samples
    neg_dist = (dist + same.float() * 1e6).min(dim=1).values
    valid = diff.any(dim=1) & (same.sum(dim=1) > 1)
    if not valid.any():
        return dist.sum() * 0.0
    loss = F.relu(pos_dist[valid] - neg_dist[valid] + margin)
    return loss.mean()


def collect_train_crops(
    manifest: Path,
    frames_root: Path,
    cache_dir: Path,
    stride: int = 25,
    min_box_h: float = 18.0,
    crop_wh: Tuple[int, int] = (128, 256),
    log=print,
) -> Tuple[List[Tuple[Path, int]], Dict[int, str]]:
    """Extract & cache labeled player crops from the 4 Phase II training clips only.

    Uses pre-extracted frames under ``frames_root`` if present, otherwise decodes directly from
    each training clip's ``video_path``. Returns ``([(crop_path, class_id), ...], id_to_label)``.
    """
    train_clips, _ = training_clips(manifest, PHASE2_ROLES["train"], PHASE2_ROLES["val"])
    cache_dir = Path(cache_dir)
    meta_file = cache_dir / f"index_s{stride}.json"
    if meta_file.is_file():
        saved = json.loads(meta_file.read_text(encoding="utf-8"))
        items = [(Path(p), int(cid)) for p, cid in saved["items"]]
        if items and items[0][0].is_file():
            log(f"reusing {len(items)} cached ReID crops from {cache_dir}")
            return items, {int(k): v for k, v in saved["id_to_label"].items()}

    cache_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[Path, int]] = []
    label_to_id: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}

    for clip in train_clips:
        ann = load_mot(clip.annotation_path)
        wanted = list(range(0, clip.frames, stride))
        have_jpegs = all(frame_path(frames_root, clip.clip_id, f).is_file() for f in wanted)
        clip_out = cache_dir / clip.clip_id
        clip_out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_saved = 0

        def _process_frame(f_idx: int, rgb: np.ndarray) -> None:
            nonlocal n_saved
            for det in ann.get(f_idx, []):
                if det.height < min_box_h:
                    continue
                key = f"{clip.clip_id}_{det.track_id}"
                if key not in label_to_id:
                    cid = len(label_to_id)
                    label_to_id[key] = cid
                    id_to_label[cid] = key
                cid = label_to_id[key]
                out_p = clip_out / f"f{f_idx:06d}_id{det.track_id:03d}.jpg"
                if not out_p.is_file():
                    box = [det.x, det.y, det.x + det.width, det.y + det.height]
                    crop = crop_box_rgb(rgb, box, out_wh=crop_wh)
                    cv2.imwrite(str(out_p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                items.append((out_p, cid))
                n_saved += 1

        if have_jpegs:
            for f_idx in wanted:
                bgr = cv2.imread(str(frame_path(frames_root, clip.clip_id, f_idx)), cv2.IMREAD_COLOR)
                if bgr is not None:
                    _process_frame(f_idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            wanted_set = set(wanted)
            cap = cv2.VideoCapture(str(clip.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"could not open training video: {clip.video_path}")
            try:
                for i in range(max(wanted) + 1):
                    if i not in wanted_set:
                        if not cap.grab():
                            break
                        continue
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        break
                    _process_frame(i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
        log(f"train clip {clip.clip_id}: {n_saved} crops extracted in {time.time() - t0:.1f}s")

    meta_file.write_text(
        json.dumps({"items": [[str(p), c] for p, c in items], "id_to_label": id_to_label}) + "\n",
        encoding="utf-8",
    )
    return items, id_to_label


def _build_pk_batches(by_class: Dict[int, List[int]], p: int, k: int, rng: random.Random) -> List[List[int]]:
    """Construct balanced PK batches (``p`` identities x ``k`` crops per batch)."""
    classes = [c for c, idxs in by_class.items() if len(idxs) >= 2]
    pools = {}
    for c in classes:
        arr = list(by_class[c])
        rng.shuffle(arr)
        pools[c] = arr

    batches: List[List[int]] = []
    avail = list(classes)
    while len(avail) >= p:
        chosen = rng.sample(avail, p)
        batch: List[int] = []
        for c in chosen:
            if len(pools[c]) >= k:
                batch.extend(pools[c][:k])
                pools[c] = pools[c][k:]
            else:
                batch.extend(rng.choices(by_class[c], k=k))
                pools[c] = []
            if len(pools[c]) < k:
                avail.remove(c)
        batches.append(batch)
    return batches


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn as nn
    import torchreid  # type: ignore

    from .evaluate import environment_record

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    items, id_to_label = collect_train_crops(
        args.manifest, args.frames_root, args.cache_dir, stride=args.stride
    )
    num_classes = len(id_to_label)
    by_class: Dict[int, List[int]] = {}
    for idx, (_, cid) in enumerate(items):
        by_class.setdefault(cid, []).append(idx)
    print(f"ReID train set: {len(items)} crops across {num_classes} player identities (4 train clips), device={device}")

    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    history = []
    best_loss = float("inf")
    rng = random.Random(args.seed)

    def _load_batch(indices: List[int]):
        imgs, lbls = [], []
        for idx in indices:
            p, cid = items[idx]
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rng.random() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
            # Mild brightness jitter (preserve jersey hue)
            if rng.random() < 0.5:
                alpha = rng.uniform(0.85, 1.15)
                rgb = np.clip(rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            # Random erasing (occlusion simulation)
            if rng.random() < 0.4:
                h, w = rgb.shape[:2]
                eh, ew = rng.randint(h // 6, h // 3), rng.randint(w // 4, w // 2)
                ey, ex = rng.randint(0, h - eh), rng.randint(0, w - ew)
                rgb[ey:ey + eh, ex:ex + ew] = 127
            imgs.append(rgb)
            lbls.append(cid)
        x = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).to(device, non_blocking=True).float().div_(255.0)
        x = (x - mean) / std
        y = torch.tensor(lbls, dtype=torch.long, device=device)
        return x, y

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = _build_pk_batches(by_class, args.pk_p, args.pk_k, rng)
        t0 = time.time()
        sum_loss = sum_ce = sum_tri = 0.0
        for batch in batches:
            x, y = _load_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                # Extract pre-classifier features via model.featuremaps + global_avgpool + fc
                v = model.featuremaps(x)
                v = model.global_avgpool(v)
                v = v.view(v.size(0), -1)
                if model.fc is not None:
                    v = model.fc(v)
                l_ce = ce_loss(logits, y)
                l_tri = batch_hard_triplet_loss(v, y, margin=args.margin)
                loss = l_ce + args.tri_weight * l_tri
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_loss += float(loss.item())
            sum_ce += float(l_ce.item())
            sum_tri += float(l_tri.item())
        scheduler.step()
        nb = max(1, len(batches))
        row = {
            "epoch": epoch,
            "batches": nb,
            "loss": sum_loss / nb,
            "ce_loss": sum_ce / nb,
            "tri_loss": sum_tri / nb,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs} loss={row['loss']:.4f} (ce={row['ce_loss']:.4f}, tri={row['tri_loss']:.4f}) "
              f"lr={row['lr']:.6f} ({row['seconds']:.1f}s)", flush=True)
        # Save backbone weights (excluding classifier head so FeatureExtractor loads cleanly)
        backbone_sd = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("classifier.")}
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(backbone_sd, out_dir / "best_osnet_soccer.pt")
        torch.save(backbone_sd, out_dir / "last_osnet_soccer.pt")

    summary = {
        "backbone": "osnet_x1_0",
        "train_clips": list(PHASE2_ROLES["train"]),
        "num_identities": num_classes,
        "num_crops": len(items),
        "epochs": args.epochs,
        "best_loss": best_loss,
        "weights_path": str(out_dir / "best_osnet_soccer.pt"),
        "history": history,
        "environment": environment_record(args.manifest, device),
    }
    (out_dir / "reid_train_log.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved fine-tuned OSNet weights -> {out_dir / 'best_osnet_soccer.pt'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune OSNet ReID on SoccerTrack training clips.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    p.add_argument("--cache-dir", type=Path, default=Path("/content/reid_crops"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/reid_osnet"))
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--pk-p", type=int, default=16, help="identities per batch")
    p.add_argument("--pk-k", type=int, default=4, help="crops per identity")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--tri-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
