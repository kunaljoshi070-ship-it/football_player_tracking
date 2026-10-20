"""CPU smoke tests for training, resuming and evaluating a checkpoint on a tiny synthetic clip.

They need torch/torchvision and take a few seconds (random-init RetinaNet, 128x128 tiles). Run from the
repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from soccer_tracking.train import lr_at


class ScheduleTests(unittest.TestCase):
    def test_lr_schedule(self):
        self.assertAlmostEqual(lr_at(0, 1000, 0.01, 100), 0.01 / 100, places=6)       # warmup start
        self.assertAlmostEqual(lr_at(99, 1000, 0.01, 100), 0.01 * (0.5 * (1 + np.cos(np.pi * 0.099))), places=6)
        self.assertLess(lr_at(999, 1000, 0.01, 100), 0.0002)                          # decayed
        self.assertGreaterEqual(lr_at(5000, 1000, 0.01, 100), 0.01 * 0.01 - 1e-12)    # floor at 1% of peak


@unittest.skipUnless(HAVE_TORCH, "torch/torchvision not installed")
class TrainSmokeTests(unittest.TestCase):
    W, H, N = 256, 128, 12

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        video = self.dir / "v.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (self.W, self.H))
        if not writer.isOpened():
            self.skipTest("OpenCV cannot write a test video here")
        for _ in range(self.N):
            img = np.full((self.H, self.W, 3), 60, np.uint8)
            cv2.rectangle(img, (20, 30), (40, 70), (0, 255, 0), -1)
            cv2.rectangle(img, (150, 40), (170, 80), (0, 255, 0), -1)
            writer.write(img)
        writer.release()
        ann = self.dir / "v.txt"
        ann.write_text("".join(f"{f},1,20,30,20,40,-1,-1,-1,-1\n{f},2,150,40,20,40,-1,-1,-1,-1\n" for f in range(self.N)))
        self.manifest = self.dir / "m.csv"
        cols = ["clip_id", "split", "local_path", "annotation_path", "width", "height", "fps", "frames"]
        with open(self.manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for cid, split in (("t1", "train"), ("v1", "val")):       # same synthetic files under two clip ids
                w.writerow([cid, split, str(video), str(ann), self.W, self.H, 5, self.N])
        self.frames = self.dir / "frames"
        with redirect_stdout(io.StringIO()):
            from soccer_tracking.data import main as data_main
            self.assertEqual(data_main(["extract", "--manifest", str(self.manifest), "--splits", "train", "val",
                                        "--stride", "5", "--out", str(self.frames)]), 0)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return ["--manifest", str(self.manifest), "--frames-root", str(self.frames), "--out-dir", str(self.dir / "runs"),
                "--run-name", "t", "--init", "none", "--stride", "5", "--tile", "128", "128", "--overlap", "16",
                "--batch-size", "4", "--workers", "0", "--device", "cpu", "--val-stride", "5", "--log-every", "1",
                "--warmup-iters", "1", "--max-iters-per-epoch", "2", *extra]

    def test_train_resume_and_evaluate_checkpoint(self):
        from soccer_tracking.evaluate import main as eval_main
        from soccer_tracking.train import main as train_main

        run = self.dir / "runs" / "t"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(train_main(self.args("--epochs", "1")), 0)
        for name in ("metrics.csv", "iterations.csv", "last.pt", "best.pt", "summary.json"):
            self.assertTrue((run / name).is_file(), name)
        self.assertTrue(list(run.glob("config_*.json")))
        self.assertEqual(len(list(csv.DictReader((run / "metrics.csv").open()))), 1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(train_main(self.args("--epochs", "1")), 2)                   # refuses a non-empty run dir
            self.assertEqual(train_main(self.args("--epochs", "2", "--resume", str(run / "last.pt"))), 0)
        rows = list(csv.DictReader((run / "metrics.csv").open()))
        self.assertEqual([r["epoch"] for r in rows], ["1", "2"])                           # continued, not restarted
        out = self.dir / "eval.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(eval_main(["--manifest", str(self.manifest), "--frames-root", str(self.frames), "--split", "val",
                                        "--weights", str(run / "best.pt"), "--stride", "5", "--tile", "128", "128",
                                        "--overlap", "16", "--device", "cpu", "--out", str(out)]), 0)
            self.assertEqual(eval_main(["--manifest", str(self.manifest), "--split", "test"]), 2)  # no test clips / guard
        record = json.loads(out.read_text())
        self.assertIn("ap50", record["results"])
        self.assertEqual(record["results"]["n_gt"], 3 * 2)                                 # 3 frames x 2 players


if __name__ == "__main__":
    unittest.main()
