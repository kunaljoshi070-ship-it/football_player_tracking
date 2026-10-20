"""Tests for metrics, tile-merge logic and the torch glue (torch tests skipped if torch is missing).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from soccer_tracking.detect import merge_tile_detections, nms_numpy
from soccer_tracking.metrics import DetectionEvaluator, average_precision, iou_matrix, match_image

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class MetricTests(unittest.TestCase):
    def test_iou(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
        np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])
        self.assertEqual(iou_matrix(np.zeros((0, 4)), b).shape, (0, 3))

    def test_matching_is_greedy_by_score_and_one_to_one(self):
        gt = np.array([[0, 0, 10, 10]])
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        tp = match_image(gt, boxes, np.array([0.4, 0.9]), 0.5)
        self.assertEqual(tp.tolist(), [False, True])          # higher score wins, duplicate is a false positive

    def test_ap_perfect_and_degenerate(self):
        self.assertAlmostEqual(average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2), 1.0)
        self.assertEqual(average_precision(np.zeros(0), np.zeros(0, bool), 3), 0.0)
        self.assertTrue(np.isnan(average_precision(np.array([0.5]), np.array([False]), 0)))

    def test_ap_hand_computed(self):
        # scores sorted: TP, FP, TP with 2 GT: precision 1, .5, .667; recall .5, .5, 1 -> AP = .5*1 + .5*(2/3)
        ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
        self.assertAlmostEqual(ap, 0.5 + 0.5 * 2 / 3)

    def test_evaluator_summary(self):
        ev = DetectionEvaluator()
        gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        ev.add_image(gt, np.array([[0, 0, 10, 10], [100, 100, 110, 110]]), np.array([0.9, 0.6]))
        s = ev.summary(report_score=0.5)
        self.assertEqual((s["n_gt"], s["tp"], s["fp"], s["fn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["recall"], 0.5)
        self.assertAlmostEqual(s["ap50"], 0.5)
        s_hi = ev.summary(report_score=0.8)                    # only the first prediction counts
        self.assertEqual((s_hi["tp"], s_hi["fp"], s_hi["fn"]), (1, 0, 1))
        self.assertAlmostEqual(s_hi["precision"], 1.0)


class MergeTests(unittest.TestCase):
    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
        self.assertEqual(nms_numpy(boxes, np.array([0.6, 0.9, 0.5]), 0.5).tolist(), [1, 2])
        self.assertEqual(len(nms_numpy(np.zeros((0, 4)), np.zeros(0), 0.5)), 0)

    def test_shift_dedupe_and_edge_drop(self):
        windows = [(0, 0, 100, 100), (50, 0, 150, 100)]       # frame 150x100, tiles overlap on x 50..100
        t0 = (np.array([[60, 10, 80, 50],                      # whole player, frame x 60..80
                        [90, 10, 100, 50]]),                   # touches tile 0's right border (interior) -> dropped
              np.array([0.9, 0.8]))
        t1 = (np.array([[10, 10, 30, 50],                      # same player as above, seen from tile 1 (frame x 60..80)
                        [0, 10, 12, 50],                       # touches tile 1's left border (interior) -> dropped
                        [120 - 50, 20, 100 - 0, 60]]),         # touches tile 1's right border = frame border -> kept
              np.array([0.7, 0.6, 0.5]))
        boxes, scores = merge_tile_detections([t0, t1], windows, (150, 100), merge_iou=0.5, drop_edge_px=2.0)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_allclose(boxes[0], [60, 10, 80, 50])  # higher-scoring duplicate kept
        np.testing.assert_allclose(boxes[1], [120, 20, 150, 60])
        np.testing.assert_allclose(scores, [0.9, 0.5], atol=1e-6)
        kept_all, _ = merge_tile_detections([t0, t1], windows, (150, 100), drop_edge_px=-1)
        self.assertGreater(len(kept_all), len(boxes))           # disabling the edge rule keeps the cut boxes

    def test_empty(self):
        b, s = merge_tile_detections([(np.zeros((0, 4)), np.zeros(0))], [(0, 0, 10, 10)], (10, 10))
        self.assertEqual((b.shape, s.shape), ((0, 4), (0,)))


@unittest.skipUnless(HAVE_TORCH, "torch/torchvision not installed")
class TorchGlueTests(unittest.TestCase):
    def test_detect_tiles_with_fake_model(self):
        import torch
        from soccer_tracking.detect import detect_tiles

        class Fake:
            def eval(self):
                return self

            def __call__(self, batch):
                return [{"boxes": torch.tensor([[1.0, 2.0, 5.0, 9.0], [3.0, 3.0, 8.0, 8.0]]),
                         "labels": torch.tensor([1, 3]), "scores": torch.tensor([0.9, 0.8])} for _ in batch]

        img = np.zeros((20, 30, 3), np.uint8)
        out = detect_tiles(Fake(), img, [(0, 0, 15, 20), (15, 0, 30, 20), (0, 0, 10, 10)], "cpu", batch_size=2)
        self.assertEqual(len(out), 3)                           # one result per window across batches
        for boxes, scores in out:
            self.assertEqual(boxes.shape, (1, 4))               # only label 1 kept
            np.testing.assert_allclose(scores, [0.9], atol=1e-6)

    def test_real_retinanet_runs_on_a_small_tile(self):
        from soccer_tracking.detect import build_retinanet, detect_tiles

        model = build_retinanet("none", tile=(128, 128), score_thresh=0.0)
        out = detect_tiles(model, np.full((128, 128, 3), 90, np.uint8), [(0, 0, 128, 128)], "cpu")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0].shape[1], 4)
        with self.assertRaises(ValueError):
            build_retinanet("bogus")


if __name__ == "__main__":
    unittest.main()
