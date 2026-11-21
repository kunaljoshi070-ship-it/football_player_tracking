"""Tests for the Phase II roles/guards, the arm builder, the TrackEval result parser (on a mock) and the reference scoring wrapper.

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
TrackEval itself is not needed: `python -m soccer_tracking.hota_eval selftest` is the check against the real tool.
"""

import unittest

import numpy as np
import csv, tempfile
from pathlib import Path

from soccer_tracking.hota_eval import HOTA_KEYS, evaluate_sequences, parse_trackeval
from soccer_tracking.phase2 import PHASE2_ROLES, build_arm, check_roles_disjoint, role_of, select_clips


class RoleTests(unittest.TestCase):
    def test_roles_cover_ten_clips_disjointly(self):
        check_roles_disjoint()
        self.assertEqual(sum(len(v) for v in PHASE2_ROLES.values()), 10)
        self.assertEqual(role_of("117092"), "train")
        with self.assertRaises(ValueError):
            role_of("999")

    def test_guards(self):
        self.assertEqual(select_clips("val", None, False, False), ["117093", "128058"])
        self.assertEqual(select_clips(None, ["117093"], False, False), ["117093"])
        with self.assertRaisesRegex(ValueError, "held-out test"):
            select_clips("test", None, False, False)
        self.assertEqual(select_clips("test", None, True, False), ["128057", "132831"])
        with self.assertRaisesRegex(ValueError, "reserve"):
            select_clips("reserve", None, True, False)
        self.assertEqual(select_clips(None, ["118578"], False, True), ["118578"])
        with self.assertRaisesRegex(ValueError, "training clip"):
            select_clips(None, ["118575"], True, True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_clips("val", ["117093"], False, False)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_clips(None, None, False, False)

    def test_suppress_static(self):
        from soccer_tracking.postfilter import suppress_static
        still = np.array([[10., 10., 30., 50.]])
        frames = [(np.vstack([still, [[100. + 3 * i, 10., 120. + 3 * i, 50.]]]), np.array([0.9, 0.9])) for i in range(60)]
        out = suppress_static(frames, window=40, step=2, min_frac=0.9)
        for b, s in out[10:50]:
            self.assertEqual(len(b), 1)
            self.assertGreater(b[0, 0], 50)


def dets(n=10):
    out = []
    for f in range(n):
        x = 100.0 + 2 * f
        b = np.array([[x, 100, x + 20, 140], [500, 100, 520, 140]], dtype=np.float32)
        out.append((b, np.array([0.9, 0.3], dtype=np.float32)))      # second box is below the 0.5 threshold
    return out


class ArmTests(unittest.TestCase):
    def test_detector_arm_unique_ids_and_threshold(self):
        preds, sc = build_arm(dets(), "detector", 0.5)
        ids = np.concatenate([p[0] for p in preds])
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids.tolist())), 10)
        self.assertTrue(all(len(s) == 1 for s in sc))

    def test_iou_arm_keeps_one_id_and_needs_params(self):
        params = {"iou_thr": 0.4, "max_age": 1, "min_hits": 1, "use_velocity": True}
        preds, _ = build_arm(dets(), "iou", 0.5, params)
        self.assertEqual({int(i) for p in preds for i in p[0]}, {1})
        with self.assertRaises(ValueError):
            build_arm(dets(), "iou", 0.5, None)
        with self.assertRaises(ValueError):
            build_arm(dets(), "nope", 0.5)


def mock_entry(hota=0.6, fp=10, fn=20, idsw=3):
    return {"pedestrian": {"HOTA": {"HOTA": np.full(19, hota), "DetA": np.full(19, 0.7), "AssA": np.full(19, 0.5), "LocA": np.full(19, 0.85)},
                           "CLEAR": {"CLR_FP": fp, "CLR_FN": fn, "IDSW": idsw},
                           "Identity": {"IDF1": 0.55}}}


class ParseTests(unittest.TestCase):
    def test_parse_and_mean_over_sequences(self):
        res = {"MotChallenge2DBox": {"mine": {"a": mock_entry(0.6, 10, 20, 3), "b": mock_entry(0.4, 30, 40, 5), "COMBINED_SEQ": mock_entry(0.5, 40, 60, 8)}}}
        out = parse_trackeval(res, "mine", ["a", "b"])
        self.assertAlmostEqual(out["mean_over_seqs"]["HOTA"], 0.5)
        self.assertEqual(out["mean_over_seqs"]["FP"], 20.0)
        self.assertEqual(out["pooled"]["FN"], 60.0)
        self.assertEqual(out["per_seq"]["b"]["IDSW"], 5.0)

    def test_rejects_percent_scale(self):
        res = {"MotChallenge2DBox": {"mine": {"a": mock_entry(60.0), "COMBINED_SEQ": mock_entry(60.0)}}}
        with self.assertRaisesRegex(ValueError, "not fractions"):
            parse_trackeval(res, "mine", ["a"])


class ReferenceWrapperTests(unittest.TestCase):
    def test_perfect_prediction_two_sequences(self):
        gt = [(np.array([1, 2]), np.array([[10 + f, 10, 30 + f, 50], [200, 10, 220, 50]], dtype=float)) for f in range(20)]
        seqs = {"a": {"gt": gt, "pred": gt, "scores": None, "width": 640, "height": 360},
                "b": {"gt": gt, "pred": gt, "scores": None, "width": 640, "height": 360}}
        res = evaluate_sequences(seqs, use_trackeval=False)
        self.assertIsNone(res["trackeval"])
        for k in HOTA_KEYS:
            self.assertAlmostEqual(res["reference"]["mean_over_seqs"][k], 1.0, places=6)
        self.assertEqual(res["reference"]["mean_over_seqs"]["IDSW"], 0.0)


if __name__ == "__main__":
    unittest.main()
