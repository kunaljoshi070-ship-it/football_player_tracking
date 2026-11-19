"""Tests for the HOTA reference implementation, the MOTChallenge exporter and the resumable chunk helpers (NumPy/SciPy only).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from soccer_tracking import fullclip
from soccer_tracking.hota import evaluate, hota_sequence, combine
from soccer_tracking.mot_export import write_gt, write_seqmap, write_tracker


def scene(n=40, players=5):
    """Players moving right at different speeds; boxes 20x40."""
    seq = []
    for f in range(n):
        ids = np.arange(1, players + 1)
        x = 100.0 * ids + (1.0 + ids) * f
        boxes = np.stack([x, np.full(players, 100.0), x + 20, np.full(players, 140.0)], 1)
        seq.append((ids, boxes))
    return seq


class HotaSanity(unittest.TestCase):
    def test_ground_truth_as_prediction_is_perfect(self):
        gt = scene()
        r = evaluate(gt, gt)
        for k in ("HOTA", "DetA", "AssA", "LocA"):
            self.assertAlmostEqual(r[k], 1.0, places=6, msg=k)
        self.assertEqual((r["IDSW"], r["FP_at_0.5"], r["FN_at_0.5"]), (0, 0, 0))

    def test_renaming_ids_changes_nothing(self):
        gt = scene()
        pred = [(ids + 1000, b) for ids, b in gt]
        self.assertAlmostEqual(evaluate(gt, pred)["HOTA"], 1.0, places=6)

    def test_shift_lowers_loca_and_high_alphas_score_zero(self):
        gt = scene()
        pred = [(i, b + np.array([5, 0, 5, 0])) for i, b in gt]        # every box has IoU 0.6 with its truth
        r = evaluate(gt, pred)
        self.assertLess(r["LocA"], 0.9)
        self.assertLess(r["HOTA"], 1.0)
        # alphas 0.05..0.60 (12 of 19) match everything, 0.65..0.95 match nothing and score 0, as in TrackEval
        self.assertAlmostEqual(r["AssA"], 12 / 19, places=6)
        self.assertAlmostEqual(r["DetA"], 12 / 19, places=6)

    def test_id_swap_lowers_assa_and_counts_switches(self):
        gt = scene(n=40, players=2)
        pred = []
        for k, (ids, b) in enumerate(gt):
            swapped = ids[::-1].copy() if k >= 20 else ids.copy()      # the two tracks exchange ids halfway
            pred.append((swapped, b))
        r = evaluate(gt, pred)
        self.assertAlmostEqual(r["DetA"], 1.0, places=6)               # detection is perfect
        self.assertLess(r["AssA"], 0.8)
        self.assertEqual(r["IDSW"], 2)

    def test_missing_and_extra_boxes(self):
        gt = scene()
        miss = [(i[:-1], b[:-1]) for i, b in gt]                       # drop one of five players everywhere
        r = evaluate(gt, miss)
        self.assertAlmostEqual(r["DetA"], 0.8, places=6)
        self.assertEqual(r["FN_at_0.5"], 40)
        extra = [(np.append(i, 99), np.vstack([b, [900, 300, 920, 340]])) for i, b in gt]
        r2 = evaluate(gt, extra)
        self.assertAlmostEqual(r2["DetA"], 5 / 6, places=6)
        self.assertEqual(r2["FP_at_0.5"], 40)

    def test_pooling_sequences_sums_counts(self):
        gt = scene()
        a = hota_sequence(gt, gt)
        both = combine([a, a])
        self.assertEqual(both["n_gt"], 2 * a["n_gt"])
        self.assertAlmostEqual(both["HOTA"], 1.0, places=6)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            hota_sequence(scene(5), scene(4))


class ExportTests(unittest.TestCase):
    def test_layout_and_one_based_frames(self):
        gt = scene(3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root, trk = Path(tmp) / "gt", Path(tmp) / "trk"
            p = write_gt(root, "clipA", gt, 4096, 1080, 25)
            lines = p.read_text().splitlines()
            self.assertEqual(len(lines), 6)
            first = lines[0].split(",")
            self.assertEqual(first[0], "1")                                # 0-based input frame 0 -> frame 1
            self.assertEqual(first[1], "1")
            self.assertEqual(first[6:], ["1", "1", "1"])
            self.assertAlmostEqual(float(first[4]), 20.0)                  # width from xyxy
            self.assertAlmostEqual(float(first[5]), 40.0)
            ini = (root / "clipA" / "seqinfo.ini").read_text()
            self.assertIn("seqLength=3", ini)
            self.assertIn("imWidth=4096", ini)
            t = write_tracker(trk, "mine", "clipA", gt, scores=[np.full(2, 0.9)] * 3)
            self.assertEqual(t, trk / "mine" / "data" / "clipA.txt")
            self.assertTrue(t.read_text().splitlines()[0].endswith("0.900000,-1,-1,-1"))
            sm = write_seqmap(root, ["clipA", "clipB"])
            self.assertEqual(sm.read_text(), "name\nclipA\nclipB\n")


class ChunkTests(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(fullclip.plan_chunks(10, 4), [(0, 4), (4, 4), (8, 2)])
        self.assertEqual(fullclip.plan_chunks(10, 4, start=2, end=7), [(2, 4), (6, 1)])
        for bad in ((10, 0), (10, 4, 10), (10, 4, -1)):
            with self.assertRaises(ValueError):
                fullclip.plan_chunks(*bad)

    def test_resume_and_assemble(self):
        plan = fullclip.plan_chunks(6, 3)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertEqual(fullclip.missing_chunks(d, "c", plan), plan)
            fullclip.save_chunk(d, "c", 0, [0, 0, 2], [[0, 0, 1, 1], [5, 5, 6, 6], [1, 1, 2, 2]], [0.9, 0.8, 0.7], 3)
            self.assertEqual(fullclip.missing_chunks(d, "c", plan), [(3, 3)])
            with self.assertRaises(FileNotFoundError):
                fullclip.assemble(d, "c", plan)
            fullclip.chunk_path(d, "c", 3).write_bytes(b"half written")            # corrupt file is not 'complete'
            self.assertEqual(fullclip.missing_chunks(d, "c", plan), [(3, 3)])
            fullclip.save_chunk(d, "c", 3, [1], [[2, 2, 3, 3]], [0.6], 3)
            self.assertEqual(fullclip.missing_chunks(d, "c", plan), [])
            frames = fullclip.assemble(d, "c", plan)
            self.assertEqual([len(b) for b, _ in frames], [2, 0, 1, 0, 1, 0])
            self.assertAlmostEqual(float(frames[2][1][0]), 0.7, places=5)


if __name__ == "__main__":
    unittest.main()
