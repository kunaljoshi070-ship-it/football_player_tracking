"""Chunked appearance (OSNet / torchvision) and green-masked torso Lab color feature extraction (Phase II, WS-C2).

Extracts per-detection embeddings and torso color vectors directly from cached detection chunks
(``<clip>_c<start>.npz``) in a single sequential video decode pass per missing chunk, saving resumable
``<clip>_f<start>.npz`` files next to the detections.

Command line (run on Colab T4, ~15-25 min for both full validation clips)::

    PYTHONPATH=src python -m soccer_tracking.features extract \
        --role val --run p2more_full \
        --det-root /content/drive/MyDrive/soccer_runs/detections_p2 \
        --backbone auto --min-score 0.25 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .fullclip import chunk_path


def feature_chunk_path(det_dir: Path, clip_id: str, start: int) -> Path:
    return Path(det_dir) / f"{clip_id}_f{start}.npz"


def feature_chunk_complete(path: Path, length: int) -> bool:
    try:
        z = np.load(path)
        return (int(z["n_frames"]) == length
                and len(z["frame"]) == len(z["det_idx"]) == len(z["emb"]) == len(z["torso_lab"]))
    except Exception:
        return False


def missing_feature_chunks(det_dir: Path, clip_id: str, plan: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(s, n) for s, n in plan if not feature_chunk_complete(feature_chunk_path(det_dir, clip_id, s), n)]


def save_feature_chunk(det_dir: Path, clip_id: str, start: int, frames, det_indices, embs, torso_labs,
                       n_frames: int, emb_dim: int) -> Path:
    """Atomic save of a feature chunk aligned to a detection chunk."""
    path = feature_chunk_path(det_dir, clip_id, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        start=int(start),
        n_frames=int(n_frames),
        emb_dim=int(emb_dim),
        frame=np.asarray(frames, dtype=np.int32),
        det_idx=np.asarray(det_indices, dtype=np.int32),
        emb=np.asarray(embs, dtype=np.float16).reshape(-1, emb_dim),
        torso_lab=np.asarray(torso_labs, dtype=np.float32).reshape(-1, 3),
    )
    tmp.replace(path)
    return path


def extract_torso_lab(image_rgb: np.ndarray, box_xyxy: Sequence[float]) -> np.ndarray:
    """Median CIE-Lab color of the central torso region of ``box_xyxy`` with pitch-green pixels masked out.

    Torso sub-box: middle 40% width (0.30..0.70), upper-middle 37% height (0.18..0.55).
    Returns float32 shape ``(3,)`` in OpenCV Lab scale ([0..255] per channel); falls back to unmasked median
    if >85% of pixels are green, or ``[0, 128, 128]`` if the crop is empty.
    """
    fh, fw = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    tx1 = int(np.clip(round(x1 + 0.30 * bw), 0, fw - 1))
    tx2 = int(np.clip(round(x1 + 0.70 * bw), tx1 + 1, fw))
    ty1 = int(np.clip(round(y1 + 0.18 * bh), 0, fh - 1))
    ty2 = int(np.clip(round(y1 + 0.55 * bh), ty1 + 1, fh))
    crop_rgb = image_rgb[ty1:ty2, tx1:tx2]
    if crop_rgb.size == 0:
        return np.array([0.0, 128.0, 128.0], dtype=np.float32)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    # Pitch turf in OpenCV HSV (H in [0, 179]): green hue ~32..92 with moderate-to-high saturation
    green = (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 92) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 35)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    non_green = ~green.reshape(-1)
    if int(non_green.sum()) >= max(4, int(0.15 * len(non_green))):
        pixels = lab[non_green]
    else:
        pixels = lab
    return np.median(pixels, axis=0).astype(np.float32)


def crop_box_rgb(image_rgb: np.ndarray, box_xyxy: Sequence[float], out_wh: Tuple[int, int] = (128, 256)) -> np.ndarray:
    """Crop ``box_xyxy`` from ``image_rgb`` and resize to ``(W, H)`` uint8 RGB."""
    fh, fw = image_rgb.shape[:2]
    x1 = int(np.clip(np.floor(box_xyxy[0]), 0, fw - 1))
    y1 = int(np.clip(np.floor(box_xyxy[1]), 0, fh - 1))
    x2 = int(np.clip(np.ceil(box_xyxy[2]), x1 + 1, fw))
    y2 = int(np.clip(np.ceil(box_xyxy[3]), y1 + 1, fh))
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_wh[1], out_wh[0], 3), dtype=np.uint8)
    return cv2.resize(crop, out_wh, interpolation=cv2.INTER_LINEAR)


class AppearanceExtractor:
    """L2-normalized ReID embedding extractor: OSNet (torchreid) with zero-install torchvision fallback."""

    def __init__(self, backbone: str = "auto", device: str = "auto"):
        import torch

        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        )
        self.backbone_used = ""
        self.emb_dim = 0
        self._mode = ""
        self._model = None
        self._input_wh = (128, 256)

        if backbone in ("auto", "osnet"):
            try:
                from torchreid.utils import FeatureExtractor  # type: ignore

                self._model = FeatureExtractor(model_name="osnet_x1_0", device=str(self.device))
                self.backbone_used = "torchreid:osnet_x1_0"
                self.emb_dim = 512
                self._mode = "torchreid"
                return
            except Exception as exc:
                if backbone == "osnet":
                    raise ImportError(
                        f"Requested --backbone osnet but torchreid failed ({exc}); install torchreid or use --backbone auto"
                    ) from exc

        if backbone in ("auto", "mobilenet_v3_large", "resnet50"):
            import torchvision.models as tv_models

            if backbone == "resnet50":
                m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
                m.fc = torch.nn.Identity()
                self.backbone_used = "torchvision:resnet50"
                self.emb_dim = 2048
            else:
                m = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT)
                m.classifier = torch.nn.Identity()
                self.backbone_used = "torchvision:mobilenet_v3_large"
                self.emb_dim = 960
            self._model = m.eval().to(self.device)
            self._mode = "torchvision"
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            return

        raise ValueError(f"unknown backbone {backbone!r}; expected auto, osnet, mobilenet_v3_large, or resnet50")

    def extract_batch(self, crops_rgb: Sequence[np.ndarray], batch_size: int = 256) -> np.ndarray:
        """Return L2-normalized float32 embeddings ``(N, D)`` for uint8 RGB crops."""
        import torch

        if not crops_rgb:
            return np.zeros((0, self.emb_dim), dtype=np.float32)

        outs = []
        amp_enabled = self.device.type == "cuda"
        with torch.inference_mode():
            for i in range(0, len(crops_rgb), batch_size):
                chunk = crops_rgb[i:i + batch_size]
                if self._mode == "torchreid":
                    # torchreid FeatureExtractor accepts list of HWC RGB uint8 numpy arrays
                    feats = self._model(list(chunk))
                    if hasattr(feats, "detach"):
                        feats = feats.detach().float().cpu().numpy()
                    else:
                        feats = np.asarray(feats, dtype=np.float32)
                else:
                    arr = np.stack(chunk, axis=0)  # (B, H, W, 3)
                    x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device, non_blocking=True).float().div_(255.0)
                    x = (x - self._mean) / self._std
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                        feats = self._model(x).float().cpu().numpy()
                norms = np.linalg.norm(feats, axis=1, keepdims=True)
                feats = np.divide(feats, np.maximum(norms, 1e-8))
                outs.append(feats.astype(np.float32))
        return np.concatenate(outs, axis=0)


def assemble_features(det_dir: Path, clip_id: str, plan: Sequence[Tuple[int, int]],
                      dets: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Load and align extracted features to per-frame ``dets``.

    Returns ``(per_frame_feats, emb_dim)`` where ``per_frame_feats[f]`` is
    ``{"emb": float32 (N_f, D), "torso_lab": float32 (N_f, 3), "valid": bool (N_f,)}`` matching ``dets[f]``.
    Detections below ``min_score`` during extraction have ``valid=False`` and zero embedding.
    """
    gone = missing_feature_chunks(det_dir, clip_id, plan)
    if gone:
        raise FileNotFoundError(
            f"clip {clip_id}: {len(gone)} feature chunk(s) missing, first start {gone[0][0]}; run `features extract` first"
        )
    per_frame: List[Dict[str, np.ndarray]] = []
    emb_dim = 0
    frame_offset = 0
    for s, n in plan:
        z = np.load(feature_chunk_path(det_dir, clip_id, s))
        emb_dim = int(z["emb_dim"])
        f_arr, idx_arr = z["frame"], z["det_idx"]
        emb_arr, lab_arr = z["emb"].astype(np.float32), z["torso_lab"].astype(np.float32)
        order = np.argsort(f_arr, kind="stable")
        f_arr, idx_arr, emb_arr, lab_arr = f_arr[order], idx_arr[order], emb_arr[order], lab_arr[order]
        edges = np.searchsorted(f_arr, np.arange(n + 1))
        for k in range(n):
            n_det = len(dets[frame_offset + k][1])
            embs = np.zeros((n_det, emb_dim), dtype=np.float32)
            labs = np.full((n_det, 3), np.nan, dtype=np.float32)
            valid = np.zeros(n_det, dtype=bool)
            lo, hi = int(edges[k]), int(edges[k + 1])
            if hi > lo:
                di = idx_arr[lo:hi]
                embs[di] = emb_arr[lo:hi]
                labs[di] = lab_arr[lo:hi]
                valid[di] = True
            per_frame.append({"emb": embs, "torso_lab": labs, "valid": valid})
        frame_offset += n
    return per_frame, emb_dim


def extract_clip_features(extractor: AppearanceExtractor, clip, det_dir: Path, plan: Sequence[Tuple[int, int]],
                          min_score: float = 0.25, batch_size: int = 256, log=print) -> int:
    """Compute missing feature chunks for ``clip`` from cached detections in ``det_dir``."""
    todo = missing_feature_chunks(det_dir, clip.clip_id, plan)
    log(f"clip {clip.clip_id}: {len(plan) - len(todo)}/{len(plan)} feature chunks done, {len(todo)} to run ({extractor.backbone_used})")
    for start, length in todo:
        det_z = np.load(chunk_path(det_dir, clip.clip_id, start))
        d_frame, d_boxes, d_scores = det_z["frame"], det_z["boxes"], det_z["scores"]
        order = np.argsort(d_frame, kind="stable")
        d_frame, d_boxes, d_scores = d_frame[order], d_boxes[order], d_scores[order]
        edges = np.searchsorted(d_frame, np.arange(length + 1))

        cap = cv2.VideoCapture(str(clip.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {clip.video_path}")
        frames_out: List[int] = []
        det_idx_out: List[int] = []
        labs_out: List[np.ndarray] = []
        crops_buf: List[np.ndarray] = []
        embs_chunks: List[np.ndarray] = []
        t0 = time.time()
        try:
            for i in range(start):
                if not cap.grab():
                    raise RuntimeError(f"decoding stopped at frame {i}")
            for k in range(length):
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"could not decode frame {start + k}")
                lo, hi = int(edges[k]), int(edges[k + 1])
                if hi == lo:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                boxes_k, scores_k = d_boxes[lo:hi], d_scores[lo:hi]
                for di in range(len(scores_k)):
                    if float(scores_k[di]) < min_score:
                        continue
                    frames_out.append(k)
                    det_idx_out.append(di)
                    labs_out.append(extract_torso_lab(rgb, boxes_k[di]))
                    crops_buf.append(crop_box_rgb(rgb, boxes_k[di]))
                if len(crops_buf) >= batch_size * 4:
                    embs_chunks.append(extractor.extract_batch(crops_buf, batch_size=batch_size))
                    crops_buf.clear()
            if crops_buf:
                embs_chunks.append(extractor.extract_batch(crops_buf, batch_size=batch_size))
                crops_buf.clear()
        finally:
            cap.release()

        embs_all = np.concatenate(embs_chunks, axis=0) if embs_chunks else np.zeros((0, extractor.emb_dim), dtype=np.float32)
        labs_all = np.stack(labs_out, axis=0) if labs_out else np.zeros((0, 3), dtype=np.float32)
        save_feature_chunk(det_dir, clip.clip_id, start, frames_out, det_idx_out, embs_all, labs_all,
                           length, extractor.emb_dim)
        log(f"clip {clip.clip_id}: feature chunk {start}..{start + length - 1} ({len(frames_out)} crops) "
            f"saved in {time.time() - t0:.1f}s")
    return len(todo)


def cmd_extract(args) -> None:
    from .data import load_manifest
    from .evaluate import environment_record
    from .phase2 import plan_file, select_clips

    ids = select_clips(args.role, args.clip, args.allow_test, args.allow_reserve)
    by_id = {c.clip_id: c for c in load_manifest(args.manifest)}
    det_dir = Path(args.det_root) / args.run
    extractor = AppearanceExtractor(backbone=args.backbone, device=args.device)
    for cid in ids:
        pf = plan_file(det_dir, cid)
        if not pf.is_file():
            raise FileNotFoundError(f"no detection plan for clip {cid} in {det_dir}; run `phase2 detect` first")
        info = json.loads(pf.read_text(encoding="utf-8"))
        plan = [tuple(p) for p in info["plan"]]
        extract_clip_features(extractor, by_id[cid], det_dir, plan, args.min_score, args.batch_size)
    (det_dir / "features_config.json").write_text(
        json.dumps(
            {
                "backbone_requested": args.backbone,
                "backbone_used": extractor.backbone_used,
                "emb_dim": extractor.emb_dim,
                "min_score": args.min_score,
                "batch_size": args.batch_size,
                "clips": ids,
                "environment": environment_record(args.manifest, extractor.device),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"features ({extractor.backbone_used}, dim={extractor.emb_dim}) -> {det_dir}")


def build_parser() -> argparse.ArgumentParser:
    from .data import DEFAULT_MANIFEST
    from .phase2 import DEFAULT_DET_ROOT

    p = argparse.ArgumentParser(description="Extract appearance embeddings and torso Lab colors for cached detections.")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    e.add_argument("--role", choices=("val", "test", "reserve"))
    e.add_argument("--clip", nargs="+")
    e.add_argument("--run", required=True)
    e.add_argument("--det-root", type=Path, default=DEFAULT_DET_ROOT)
    e.add_argument("--allow-test", action="store_true")
    e.add_argument("--allow-reserve", action="store_true")
    e.add_argument("--backbone", choices=("auto", "osnet", "mobilenet_v3_large", "resnet50"), default="auto")
    e.add_argument("--min-score", type=float, default=0.25)
    e.add_argument("--batch-size", type=int, default=256)
    e.add_argument("--device", default="auto")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        {"extract": cmd_extract}[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
