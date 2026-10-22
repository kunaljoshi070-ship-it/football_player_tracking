"""Tests for describe_model: shape helper (NumPy) and a real small-tile run (needs torch; skipped otherwise).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from soccer_tracking.describe_model import main, shape_of

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class ShapeOfTests(unittest.TestCase):
    def test_nested(self):
        out = shape_of({"a": np.zeros((1, 2, 3)), "b": [np.zeros((4,)), (np.zeros((2, 2)),)], "c": "x"})
        self.assertEqual(out, {"a": [1, 2, 3], "b": [[4], [[2, 2]]], "c": "str"})


@unittest.skipUnless(HAVE_TORCH, "torch/torchvision not installed")
class DescribeTests(unittest.TestCase):
    def test_small_tile_without_downloading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "d.json"
            with redirect_stdout(io.StringIO()) as buf:
                rc = main(["--weights", "none", "--tile", "128", "128", "--out", str(out)])
            self.assertEqual(rc, 0)
            d = json.loads(out.read_text())
            t = d["totals"]
            self.assertEqual(t["params"], t["trainable"] + t["frozen"])
            self.assertEqual(sum(v["params"] for v in d["by_component"].values()), t["params"])
            stages = {r["stage"].split(" ")[0] + str(r.get("level", "")): r["shape"] for r in d["tensor_flow"]["stages"]}
            self.assertEqual(stages["transform"], [1, 3, 128, 128])
            fpn = next(r["shape"] for r in d["tensor_flow"]["stages"] if r["stage"].startswith("backbone.fpn"))
            self.assertEqual(len(fpn), 5)                                       # P3..P7
            self.assertTrue(all(v[1] == 256 for v in fpn.values()))             # 256 channels at every level
            head = next(r["shape"] for r in d["tensor_flow"]["stages"] if r["stage"].startswith("head output"))
            expected_anchors = sum(v[2] * v[3] for v in fpn.values()) * 9
            self.assertEqual(head["cls_logits"], [1, expected_anchors, 2])
            self.assertEqual(head["bbox_regression"], [1, expected_anchors, 4])
            self.assertEqual(d["settings"]["anchors"]["anchors_per_location"], [9] * 5)
            self.assertIn("ResNet-50 stages", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
