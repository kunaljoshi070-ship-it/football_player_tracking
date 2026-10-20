"""Run a torchvision RetinaNet over native-scale tiles and merge the results into full-frame boxes.

The merge step (``merge_tile_detections``) is plain NumPy; only ``build_retinanet`` and
``detect_tiles`` need torch/torchvision. Class convention: label 1 = person (COCO) = player (our
fine-tuning labels), so the same code evaluates the pretrained baseline and a fine-tuned model.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .tiles import Window


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression, highest score first."""
    from .metrics import iou_matrix

    boxes = np.asarray(boxes).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep: List[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[iou_matrix(boxes[i:i + 1], boxes[rest])[0] <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def merge_tile_detections(per_tile: Sequence[Tuple[np.ndarray, np.ndarray]], windows: Sequence[Window],
                          frame_size: Tuple[int, int], merge_iou: float = 0.5,
                          drop_edge_px: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Combine per-tile detections into full-frame ``(boxes_xyxy float32 (M, 4), scores (M,))``.

    1. Shift tile boxes by the window origin.
    2. Drop boxes touching a tile border that is not a frame border (within ``drop_edge_px``): such a
       player is cut by the tile and appears whole in the neighbouring tile (the tile overlap is wider
       than a player). Set ``drop_edge_px < 0`` to disable.
    3. Clip to the frame and apply NMS at ``merge_iou`` across tiles.
    """
    fw, fh = frame_size
    all_boxes, all_scores = [], []
    for (boxes, scores), (x0, y0, x1, y1) in zip(per_tile, windows):
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(b) == 0:
            continue
        if drop_edge_px >= 0:
            tw, th = x1 - x0, y1 - y0
            cut = np.zeros(len(b), dtype=bool)
            if x0 > 0:
                cut |= b[:, 0] <= drop_edge_px
            if x1 < fw:
                cut |= b[:, 2] >= tw - drop_edge_px
            if y0 > 0:
                cut |= b[:, 1] <= drop_edge_px
            if y1 < fh:
                cut |= b[:, 3] >= th - drop_edge_px
            b, s = b[~cut], s[~cut]
        b = b + np.array([x0, y0, x0, y0])
        all_boxes.append(b)
        all_scores.append(s)
    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh)
    keep = nms_numpy(boxes, scores, merge_iou)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


def adapt_head(model, num_classes: int = 2, prior: float = 0.01):
    """Replace the final classification conv so the model predicts ``num_classes`` (class 0 = unused
    background slot, class 1 = player). The rest of the pretrained network is kept."""
    import math

    import torch

    head = model.head.classification_head
    conv = torch.nn.Conv2d(head.cls_logits.in_channels, head.num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv.weight, std=0.01)
    torch.nn.init.constant_(conv.bias, -math.log((1 - prior) / prior))
    head.cls_logits = conv
    head.num_classes = num_classes
    return model


def build_retinanet(weights: str = "coco", tile: Tuple[int, int] = (1024, 1080), score_thresh: float = 0.05,
                    nms_thresh: float = 0.5, detections_per_img: int = 300):
    """torchvision RetinaNet (ResNet-50 FPN) configured to run at native tile scale.

    ``weights``: ``"coco"`` downloads torchvision's COCO weights (needs network the first time);
    ``"none"`` gives a randomly initialised 91-class model (used for tests); a path ending in ``.pt`` or
    ``.pth`` loads a fine-tuned 2-class checkpoint written by ``train.py``. ``min_size``/``max_size``
    equal the tile's short/long side, so a full tile is not rescaled.
    """
    from pathlib import Path

    from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn

    is_path = Path(weights).suffix in (".pt", ".pth")
    if weights not in ("coco", "none") and not is_path:
        raise ValueError(f"weights must be 'coco', 'none' or a .pt/.pth checkpoint path, got {weights!r}")
    kwargs = dict(weights_backbone=None, min_size=min(tile), max_size=max(tile), score_thresh=score_thresh,
                  nms_thresh=nms_thresh, detections_per_img=detections_per_img)
    if is_path:
        import torch

        if not Path(weights).is_file():
            raise FileNotFoundError(f"checkpoint not found: {weights}")
        model = adapt_head(retinanet_resnet50_fpn(weights=None, **kwargs), 2)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
        return model
    return retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1 if weights == "coco" else None, **kwargs)


def detect_tiles(model, image_rgb: np.ndarray, windows: Sequence[Window], device, batch_size: int = 2,
                 label: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Run ``model`` on each window of an RGB uint8 frame; keep detections of ``label``.
    Returns one ``(boxes_xyxy_in_tile, scores)`` pair per window, in window order."""
    import torch

    model.eval()
    per_tile: List[Tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = [torch.from_numpy(np.ascontiguousarray(image_rgb[y0:y1, x0:x1])).permute(2, 0, 1)
                     .float().div(255.0).to(device) for x0, y0, x1, y1 in windows[i:i + batch_size]]
            for out in model(batch):
                keep = out["labels"] == label
                per_tile.append((out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()))
    return per_tile
