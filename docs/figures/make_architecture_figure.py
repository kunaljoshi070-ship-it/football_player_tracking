"""Draw the detector diagram (docs/figures/architecture.png) from the shapes printed by
``python -m soccer_tracking.describe_model`` (results/model_description.json). The shapes are written out
below so the figure can be regenerated without torch:  python docs/figures/make_architecture_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FROZEN, TRAIN, NEW, PLAIN = "#c9d6e8", "#f7c98b", "#a9dfb0", "#ffffff"

fig, ax = plt.subplots(figsize=(17, 8.2))
ax.set_xlim(0, 168)
ax.set_ylim(-20, 80)
ax.axis("off")


def box(x, y, w, h, text, color=PLAIN, size=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=color, ec="#333333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal")


def arrow(p, q, style="->", color="#444444", lw=1.3):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=lw))


ax.text(84, 77.5, "RetinaNet, ResNet-50 FPN: one 1024x1080 tile (shapes are N=1, C, H, W; H is padded 1080 to 1088)",
        ha="center", fontsize=12.5, fontweight="bold")

# input
box(1, 36, 21, 22, "Input tile\n3 x 1080 x 1024\n\nnormalise (ImageNet\nmean/std), pad H to a\nmultiple of 32\n\n[3, 1088, 1024]", PLAIN)

# ResNet-50
ax.text(47, 71.5, "ResNet-50 body (FrozenBatchNorm)", ha="center", fontsize=10, fontweight="bold")
rows = [("conv1 7x7/2 + maxpool\n[64, 272, 256]", FROZEN), ("layer1: 3 blocks  = C2 (unused by FPN)\n[256, 272, 256]", FROZEN),
        ("layer2: 4 blocks  = C3, stride 8\n[512, 136, 128]", TRAIN), ("layer3: 6 blocks  = C4, stride 16\n[1024, 68, 64]", TRAIN),
        ("layer4: 3 blocks  = C5, stride 32\n[2048, 34, 32]", TRAIN)]
ys = [60, 48, 36, 24, 12]
for (t, c), y in zip(rows, ys):
    box(29, y, 36, 9, t, c)
arrow((22, 47), (28.5, 64.5))
for a, b in zip(ys[:-1], ys[1:]):
    arrow((47, a - 0.2), (47, b + 9.2))

# FPN
ax.text(94, 71.5, "FPN (256 channels at every level)", ha="center", fontsize=10, fontweight="bold")
ax.text(94, 68.6, "arrows up: top-down path (upsample and add)", ha="center", fontsize=7.3, color="#444444")
fpn = [("P3  [256, 136, 128]", ys[2]), ("P4  [256, 68, 64]", ys[3]), ("P5  [256, 34, 32]", ys[4]),
       ("P6  [256, 17, 16]\n3x3 conv stride 2 on P5", 0), ("P7  [256, 9, 8]\n3x3 conv stride 2 on P6", -12)]
for t, y in fpn:
    box(80, y, 28, 9, t, TRAIN)
for y in ys[2:]:
    arrow((65.4, y + 4.5), (79.6, y + 4.5))
ax.text(72.5, ys[2] + 11.2, "1x1 lateral +\n3x3 output conv", ha="center", fontsize=7.3, color="#444444")
arrow((94, ys[4] + 9.1), (94, ys[3] - 0.1), color="#888888")          # P5 -> P4 (top-down)
arrow((94, ys[3] + 9.1), (94, ys[2] - 0.1), color="#888888")          # P4 -> P3 (top-down)
arrow((94, ys[4] - 0.1), (94, 9.1))                                    # P5 -> P6
arrow((94, -0.1), (94, -2.9))                                         # P6 -> P7
# bracket: every level feeds both heads
mids = [y + 4.5 for _, y in fpn]
ax.plot([110.5, 110.5], [min(mids), max(mids)], color="#444444", lw=1.3)
for m in mids:
    ax.plot([108.2, 110.5], [m, m], color="#444444", lw=1.3)

# heads
ax.text(142, 71.5, "Heads (same weights on every level P3-P7)", ha="center", fontsize=10, fontweight="bold")
box(118, 60, 48, 9, "Classification subnet\n4 x (3x3 conv 256 + ReLU)", TRAIN, 8.6)
box(118, 50.5, 48, 8, "3x3 conv: 9 anchors x 2 classes, per level [18, H, W]\nnew output layer, 41,490 parameters", NEW, 8.2)
box(118, 37.5, 48, 9, "Box-regression subnet\n4 x (3x3 conv 256 + ReLU)", TRAIN, 8.6)
box(118, 28, 48, 8, "3x3 conv: 9 anchors x 4 offsets, per level [36, H, W]\n82,980 parameters", TRAIN, 8.2)
arrow((118, 64.5) if False else (110.5, 40), (117.6, 64.5))
arrow((110.5, 40), (117.6, 42))
arrow((142, 59.9), (142, 58.6))
arrow((142, 37.4), (142, 36.1))

# output
box(118, -8, 48, 31, "Per tile: 208,728 anchors\n(9 per location, sizes 32-812 px,\naspect ratios 0.5 / 1 / 2)\ncls [208728, 2], boxes [208728, 4]\n\nscore >= 0.05, top 1000 per level,\ndecode, NMS 0.5, at most 300 boxes\n\nthen merge the 5 tiles of a frame:\ndrop boxes cut by a tile border,\nNMS 0.5  ->  full-frame player boxes", PLAIN, 8.2)
arrow((142, 27.9), (142, 23.1))

# legend
for i, (c, t) in enumerate([(FROZEN, "frozen during fine-tuning (conv1, layer1, all BatchNorm)"), (TRAIN, "trainable, starts from COCO weights"),
                            (NEW, "new, randomly initialised"), (PLAIN, "no parameters / processing")]):
    ax.add_patch(FancyBboxPatch((1, -3 - 3.8 * i), 3.2, 2.4, boxstyle="round,pad=0.1", fc=c, ec="#333333", lw=0.9))
    ax.text(5.4, -1.8 - 3.8 * i, t, va="center", fontsize=8.2)

out = Path(__file__).with_name("architecture.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(out)
