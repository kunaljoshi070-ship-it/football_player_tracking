"""Describe the detector actually used: layers, parameters, frozen/trainable parts and tensor shapes.

Builds the same model as ``train.py`` (torchvision RetinaNet ResNet-50 FPN, COCO weights, final classification
layer replaced for 2 classes), runs one random 1024x1080 tile through it with forward hooks, and prints/saves:
the input pipeline, every stage's output shape, parameter counts (total / trainable / frozen) per component,
the anchor setup, and the loss/matching/post-processing settings read from the live objects. Needs torch and
torchvision; the COCO weights come from the local torch cache after the first download.

    PYTHONPATH=src python -m soccer_tracking.describe_model            # writes results/model_description.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def shape_of(x: Any):
    """Shape(s) of a tensor-like (anything with ``.shape``), or of a dict/list/tuple of them."""
    if hasattr(x, "shape"):
        return [int(d) for d in x.shape]
    if isinstance(x, dict):
        return {str(k): shape_of(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [shape_of(v) for v in x]
    return str(type(x).__name__)


def _count(module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"params": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def module_rows(model, max_depth: int = 3) -> List[dict]:
    """One row per module down to ``max_depth`` (dotted-name depth), with parameter counts."""
    rows = []
    for name, mod in model.named_modules():
        if not name or name.count(".") + 1 > max_depth:
            continue
        row = {"name": name, "type": type(mod).__name__, **_count(mod)}
        if hasattr(mod, "__len__") and not isinstance(mod, (str, bytes)):
            try:
                row["blocks"] = len(mod)
            except TypeError:
                pass
        rows.append(row)
    return rows


def collect_shapes(model, height: int, width: int) -> Dict[str, Any]:
    """Run one random frame through ``model`` (eval mode) and record every stage's output shape."""
    import torch

    records: List[dict] = []
    calls: Dict[str, int] = {}
    handles = []

    def hook_for(name: str, getter=lambda o: o, per_level: bool = False):
        def hook(module, inputs, output):
            entry = {"stage": name, "shape": shape_of(getter(output))}
            if per_level:
                entry["level"] = calls.get(name, 0)
                calls[name] = calls.get(name, 0) + 1
            records.append(entry)
        return hook

    body = model.backbone.body
    plan = [(model.transform, "transform (normalise + pad to a multiple of 32)", lambda o: o[0].tensors, False),
            (body.conv1, "backbone.body.conv1 (7x7, stride 2)", lambda o: o, False),
            (body.maxpool, "backbone.body.maxpool (stride 4 overall)", lambda o: o, False),
            (body.layer1, "backbone.body.layer1 = C2 (not used by the FPN)", lambda o: o, False),
            (body.layer2, "backbone.body.layer2 = C3", lambda o: o, False),
            (body.layer3, "backbone.body.layer3 = C4", lambda o: o, False),
            (body.layer4, "backbone.body.layer4 = C5", lambda o: o, False),
            (model.backbone, "backbone.fpn outputs (P3..P7)", lambda o: o, False),
            (model.head.classification_head.conv, "head.classification_head.conv (4 conv+ReLU, per level)", lambda o: o, True),
            (model.head.classification_head.cls_logits, "head.classification_head.cls_logits (per level)", lambda o: o, True),
            (model.head.regression_head.conv, "head.regression_head.conv (4 conv+ReLU, per level)", lambda o: o, True),
            (model.head.regression_head.bbox_reg, "head.regression_head.bbox_reg (per level)", lambda o: o, True),
            (model.head, "head output (all levels, flattened over anchors)", lambda o: o, False)]
    for module, name, getter, per_level in plan:
        handles.append(module.register_forward_hook(hook_for(name, getter, per_level)))
    model.eval()
    try:
        with torch.no_grad():
            detections = model([torch.rand(3, height, width)])
    finally:
        for h in handles:
            h.remove()
    return {"stages": records, "final_detections": shape_of({"boxes": detections[0]["boxes"], "scores": detections[0]["scores"],
                                                              "labels": detections[0]["labels"]})}


def settings(model) -> dict:
    """Loss, matching, anchor and post-processing settings read from the live model."""
    import torchvision

    sig = inspect.signature(torchvision.ops.sigmoid_focal_loss).parameters
    tr = model.transform
    matcher = model.proposal_matcher
    reg = model.head.regression_head
    return {
        "input_transform": {"min_size": tr.min_size, "max_size": tr.max_size, "size_divisible": getattr(tr, "size_divisible", None),
                            "image_mean": list(tr.image_mean), "image_std": list(tr.image_std)},
        "anchors": {"sizes_per_level": [list(s) for s in model.anchor_generator.sizes],
                    "aspect_ratios_per_level": [list(a) for a in model.anchor_generator.aspect_ratios],
                    "anchors_per_location": list(model.anchor_generator.num_anchors_per_location())},
        "classification_loss": {"name": "sigmoid_focal_loss", "alpha_default": sig["alpha"].default, "gamma_default": sig["gamma"].default,
                                "note": "summed over anchors, divided by the number of foreground anchors"},
        "regression_loss": {"type": getattr(reg, "_loss_type", "unknown"), "note": "on matched foreground anchors, divided by their number"},
        "matcher": {"foreground_iou": matcher.high_threshold, "background_iou": matcher.low_threshold,
                    "allow_low_quality_matches": matcher.allow_low_quality_matches},
        "inference": {"score_thresh": model.score_thresh, "nms_thresh": model.nms_thresh,
                      "detections_per_img": model.detections_per_img, "topk_candidates": model.topk_candidates},
    }


def describe(weights: str, classes: int, height: int, width: int) -> dict:
    import torch
    import torchvision

    from .detect import adapt_head, build_retinanet

    model = build_retinanet(weights, (width, height))
    if classes != 91 and weights in ("coco", "none"):
        adapt_head(model, classes)
    body = model.backbone.body
    top = {}
    for name, child in body.named_children():
        c = _count(child)
        top[name] = {**c, "status": "no parameters" if c["params"] == 0 else ("frozen" if c["trainable"] == 0 else "trainable")}
    shapes = collect_shapes(model, height, width)
    return {
        "model": "torchvision RetinaNet, ResNet-50 FPN backbone",
        "weights": weights, "num_classes": classes, "input_hw": [height, width],
        "totals": _count(model),
        "by_component": {"backbone.body (ResNet-50)": _count(model.backbone.body), "backbone.fpn": _count(model.backbone.fpn),
                         "head.classification_head": _count(model.head.classification_head),
                         "head.regression_head": _count(model.head.regression_head)},
        "backbone_stages": top,
        "norm_layers": sorted({type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__}),
        "modules": module_rows(model),
        "tensor_flow": shapes, "settings": settings(model),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__},
    }


def print_report(d: dict, out=print) -> None:
    t = d["totals"]
    out(f"{d['model']} | weights: {d['weights']} | classes: {d['num_classes']} | input tile {d['input_hw'][1]}x{d['input_hw'][0]} (WxH)")
    out(f"parameters: total {t['params']:,}  trainable {t['trainable']:,}  frozen {t['frozen']:,}")
    for k, v in d["by_component"].items():
        out(f"  {k:<28} total {v['params']:>12,}  trainable {v['trainable']:>12,}")
    out("ResNet-50 stages (trainable / frozen as configured): " + ", ".join(f"{k}: {v['status']}" for k, v in d["backbone_stages"].items()))
    out("normalisation layers: " + ", ".join(d["norm_layers"]))
    out("\ntensor flow for one tile (N, C, H, W):")
    for r in d["tensor_flow"]["stages"]:
        lvl = f" [level {r['level']}]" if "level" in r else ""
        out(f"  {r['stage']}{lvl}: {r['shape']}")
    out(f"  final detections after NMS: {d['tensor_flow']['final_detections']}")
    s = d["settings"]
    out("\nsettings read from the model:")
    for k, v in s.items():
        out(f"  {k}: {v}")
    out("\nmodules (depth <= 3):")
    for r in d["modules"]:
        extra = f", {r['blocks']} blocks" if "blocks" in r else ""
        out(f"  {r['name']:<46} {r['type']:<28} params {r['params']:>11,}  trainable {r['trainable']:>11,}{extra}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Print and save the layers, parameters and tensor shapes of the detector.")
    p.add_argument("--weights", choices=("coco", "none"), default="coco",
                   help="'coco' reproduces the training configuration (frozen BatchNorm, partly frozen backbone); 'none' is offline but not representative")
    p.add_argument("--classes", type=int, default=2, help="output classes (2 = background slot + player, as trained; 91 = original COCO head)")
    p.add_argument("--tile", type=int, nargs=2, default=(1024, 1080), metavar=("W", "H"))
    p.add_argument("--out", type=Path, default=Path("results") / "model_description.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        d = describe(args.weights, args.classes, args.tile[1], args.tile[0])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        print_report(d)
        print(f"\n-> {args.out}")
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
