"""Tests for the error-analysis helpers (NumPy) and a small end-to-end run (needs torch; skipped otherwise).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from soccer_tracking.errors import (analyze_frame, assign, border_distance, crop_panel, crowding, fn_type, fp_type,
                                    main, make_montage, summarize_errors)

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def b(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=float)


class HelperTests(unittest.TestCase):
    def test_assign_one_to_one(self):
        gt = np.array([b(0, 0, 10, 20), b(100, 0, 110, 20)])
        boxes = np.array([b(0, 0, 10, 20), b(0, 0, 10, 20)])
        g_of_p, p_of_g = assign(gt, boxes, np.array([0.7, 0.9]), 0.5)
        self.assertEqual(g_of_p.tolist(), [-1, 0])                       # the higher score takes the player
        self.assertEqual(p_of_g.tolist(), [1, -1])

    def test_fp_types(self):
        gt = np.array([b(0, 0, 20, 40)])
        self.assertEqual(fp_type(b(1, 0, 21, 40), gt), "duplicate")
        self.assertEqual(fp_type(b(10, 0, 30, 40), gt), "poorly_localized")   # IoU 1/3
        self.assertEqual(fp_type(b(100, 0, 120, 40), gt), "no_gt_overlap")
        self.assertEqual(fp_type(b(0, 0, 10, 10), np.zeros((0, 4))), "no_gt_overlap")

    def test_fn_types(self):
        g = b(0, 0, 20, 40)
        self.assertEqual(fn_type(g, np.zeros((0, 4)), np.zeros(0), 0.5), "missed")
        self.assertEqual(fn_type(g, np.array([b(100, 0, 120, 40)]), np.array([0.9]), 0.5), "missed")
        self.assertEqual(fn_type(g, np.array([b(1, 0, 21, 40)]), np.array([0.3]), 0.5), "low_score")
        self.assertEqual(fn_type(g, np.array([b(1, 0, 21, 40)]), np.array([0.9]), 0.5), "merged_with_neighbour")
        self.assertEqual(fn_type(g, np.array([b(10, 0, 30, 40)]), np.array([0.9]), 0.5), "poorly_localized")

    def test_crowding(self):
        c = crowding(np.array([b(0, 0, 10, 20), b(5, 0, 15, 20), b(100, 0, 110, 20)]))
        np.testing.assert_allclose(c, [0.5, 0.5, 0.0])

    def test_border_distance(self):
        windows = [(0, 0, 100, 50), (80, 0, 180, 50)]                # interior borders at x = 80 and x = 100
        np.testing.assert_allclose(border_distance(np.array([90.0, 10.0]), windows, 180), [10.0, 70.0])
        self.assertTrue(np.isinf(border_distance(np.array([5.0]), [(0, 0, 180, 50)], 180)[0]))


def scenario():
    ids = np.array([1, 2, 3, 4])
    gt = np.array([b(10, 10, 30, 50), b(100, 10, 120, 50), b(200, 10, 220, 50), b(300, 10, 320, 50)])
    boxes = np.array([b(10, 10, 30, 50),       # matches player 1
                      b(11, 10, 31, 50),       # second box on player 1 -> duplicate
                      b(200, 10, 220, 50),     # on player 3 but score 0.3 -> low_score miss
                      b(400, 10, 420, 50),     # nobody there -> no_gt_overlap
                      b(310, 10, 330, 50)])    # overlaps player 4 with IoU 1/3 -> poorly_localized
    scores = np.array([0.9, 0.7, 0.3, 0.8, 0.6])
    windows = [(0, 0, 256, 100), (224, 0, 480, 100)]                 # interior borders at x = 224 and 256
    return ids, gt, boxes, scores, windows


class AnalyzeTests(unittest.TestCase):
    def test_frame_types(self):
        ids, gt, boxes, scores, windows = scenario()
        gt_rows, pred_rows = analyze_frame(7, ids, gt, boxes, scores, windows, 480)
        self.assertEqual([r["matched"] for r in gt_rows], [True, False, False, False])
        self.assertEqual([r["type"] for r in gt_rows], ["", "missed", "low_score", "poorly_localized"])
        self.assertEqual([r["near_border"] for r in gt_rows], [False, False, True, False])   # player 3 is 14 px from x = 224
        self.assertEqual(len(pred_rows), 4)                                                  # the 0.3 box is not confident
        self.assertEqual([r["type"] for r in pred_rows], ["", "duplicate", "no_gt_overlap", "poorly_localized"])

    def test_summary(self):
        ids, gt, boxes, scores, windows = scenario()
        s = summarize_errors(*analyze_frame(7, ids, gt, boxes, scores, windows, 480))
        self.assertEqual((s["n_gt"], s["tp"], s["fn"], s["n_pred"], s["fp"]), (4, 1, 3, 4, 3))
        self.assertAlmostEqual(s["recall"], 0.25)
        self.assertAlmostEqual(s["precision"], 0.25)
        self.assertEqual(s["fn_types"], {"merged_with_neighbour": 0, "low_score": 1, "poorly_localized": 1, "missed": 1})
        self.assertEqual(s["fp_types"], {"duplicate": 1, "poorly_localized": 1, "no_gt_overlap": 1})
        row40 = next(r for r in s["recall_by_height"] if r["height_px"] == "40-50")
        self.assertEqual((row40["n_gt"], row40["recall"]), (4, 0.25))
        self.assertEqual(s["recall_by_near_tile_border"]["yes"], {"n_gt": 1, "recall": 0.0})
        self.assertIsNone(s["recall_by_crowded"]["yes"]["recall"])                           # nobody is crowded
        bands = {r["score"]: (r["n_pred"], r["precision"]) for r in s["precision_by_score"]}
        self.assertEqual(bands, {"0.5-0.7": (1, 0.0), "0.7-0.9": (2, 0.0), "0.9-1": (1, 1.0)})


class PanelTests(unittest.TestCase):
    def test_panel_and_montage(self):
        img = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
        panel = crop_panel(img, (5, 5), np.array([b(0, 0, 10, 20)]), np.zeros((0, 4)), b(0, 0, 10, 20), "FP test")
        self.assertEqual(panel.shape, (320, 320, 3))                                          # shifted inside the frame
        sheet = make_montage([panel] * 5, cols=4)
        self.assertEqual(sheet.shape, (2 * 324 + 4, 4 * 324 + 4, 3))
        self.assertIsNone(make_montage([]))

    def test_test_split_guard(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["--split", "test"]), 2)
        self.assertIn("held out", err.getvalue())


@unittest.skipUnless(HAVE_TORCH, "torch/torchvision not installed")
class EndToEndTests(unittest.TestCase):
    def test_runs_on_a_tiny_clip(self):
        W, H, N = 256, 128, 12
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "v.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (W, H))
            if not writer.isOpened():
                self.skipTest("OpenCV cannot write a test video here")
            for _ in range(N):
                writer.write(np.full((H, W, 3), 60, np.uint8))
            writer.release()
            ann = root / "v.txt"
            ann.write_text("".join(f"{f},1,20,30,20,40,-1,-1,-1,-1\n{f},2,150,40,20,40,-1,-1,-1,-1\n" for f in range(N)))
            m = root / "m.csv"
            with open(m, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["clip_id", "split", "local_path", "annotation_path", "width", "height", "fps", "frames"])
                w.writerow(["v1", "val", str(video), str(ann), W, H, 5, N])
            frames = root / "frames"
            from soccer_tracking.data import main as data_main
            with redirect_stdout(io.StringIO()):
                self.assertEqual(data_main(["extract", "--manifest", str(m), "--splits", "val", "--stride", "5", "--out", str(frames)]), 0)
                rc = main(["--manifest", str(m), "--frames-root", str(frames), "--split", "val", "--weights", "none", "--stride", "5",
                           "--tile", "128", "128", "--overlap", "16", "--device", "cpu",
                           "--out-dir", str(root / "out"), "--images-dir", str(root / "img")])
            self.assertEqual(rc, 0)
            rec = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(rec["summary"]["n_gt"], 3 * 2)                                   # 3 frames x 2 players
            self.assertTrue((root / "out" / "errors.csv").is_file())
            self.assertTrue((root / "img" / "fn_examples.jpg").is_file())                     # a random model misses everyone


if __name__ == "__main__":
    unittest.main()
