"""Tests for tracker v2 (two-threshold IoU tracker) and the selection helpers (NumPy only).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from soccer_tracking.tracker_v2 import (ByteIouTracker, GRID_V2, build_arm_v2, grid_v2, pick_final, rank_candidates)


def box(x, y=100.0, w=20.0, h=40.0):
    return np.array([x, y, x + w, y + h], dtype=float)


def step(t, x, score):
    return t.update(box(x)[None], np.array([score]))


class BehaviourTests(unittest.TestCase):
    def test_invalid_params(self):
        for kw in ({"det_low": 0.6, "det_high": 0.5}, {"det_low": 0.0}, {"match_iou": 0.0}, {"max_age": -1}, {"min_hits": 0}):
            with self.assertRaises(ValueError):
                ByteIouTracker(**kw)

    def test_confirmation_delays_output_and_keeps_id(self):
        t = ByteIouTracker(min_hits=3)
        outs = [step(t, 100 + 2 * k, 0.9)[0].tolist() for k in range(5)]
        self.assertEqual(outs, [[], [], [1], [1], [1]])

    def test_tentative_track_is_dropped_on_a_miss(self):
        t = ByteIouTracker(min_hits=2)
        step(t, 100, 0.9)
        t.update(np.zeros((0, 4)), np.zeros(0))
        ids = step(t, 102, 0.9)[0].tolist()
        self.assertEqual(ids, [])                                  # restarted as a new tentative track
        self.assertEqual(step(t, 104, 0.9)[0].tolist(), [2])

    def test_no_coasting_output_but_same_id_after_short_gap(self):
        t = ByteIouTracker(min_hits=2, max_age=5)
        for k in range(3):
            step(t, 100 + 2 * k, 0.9)
        gap = [t.update(np.zeros((0, 4)), np.zeros(0))[0].tolist() for _ in range(2)]
        self.assertEqual(gap, [[], []])                            # nothing output while unmatched
        self.assertEqual(step(t, 112, 0.9)[0].tolist(), [1])       # re-matched through the velocity prediction

    def test_track_dies_after_max_age(self):
        t = ByteIouTracker(min_hits=2, max_age=2)
        for k in range(3):
            step(t, 100 + 2 * k, 0.9)
        for _ in range(3):
            t.update(np.zeros((0, 4)), np.zeros(0))
        self.assertEqual(len(t.tracks), 0)
        self.assertEqual(step(t, 110, 0.9)[0].tolist(), [])        # new tentative track, new id later
        self.assertEqual(step(t, 112, 0.9)[0].tolist(), [2])

    def test_low_score_box_continues_a_confirmed_track_but_cannot_start_one(self):
        t = ByteIouTracker(det_high=0.6, det_low=0.1, min_hits=2)
        step(t, 100, 0.9); step(t, 102, 0.9)
        ids, _, sc, _ = step(t, 104, 0.3)                          # low score: continues track 1
        self.assertEqual(ids.tolist(), [1])
        self.assertAlmostEqual(float(sc[0]), 0.3, places=6)
        fresh = ByteIouTracker(det_high=0.6, det_low=0.1, min_hits=1)
        self.assertEqual(len(step(fresh, 100, 0.3)[0]), 0)         # low score never starts a track
        self.assertEqual(len(fresh.tracks), 0)

    def test_tentative_track_is_not_continued_by_low_score_box(self):
        t = ByteIouTracker(min_hits=3)
        step(t, 100, 0.9)                                          # tentative (1 hit)
        step(t, 102, 0.3)                                          # low score: must not extend it
        self.assertEqual(len(t.tracks), 0)

    def test_scores_below_det_low_are_ignored(self):
        t = ByteIouTracker(det_low=0.3, min_hits=1)
        self.assertEqual(len(step(t, 100, 0.2)[0]), 0)
        self.assertEqual(len(t.tracks), 0)

    def test_high_score_box_wins_the_track_over_a_low_one(self):
        t = ByteIouTracker(min_hits=2, match_iou=0.3)
        step(t, 100, 0.9); step(t, 102, 0.9)
        boxes = np.stack([box(104), box(106)])
        ids, out, _, _ = t.update(boxes, np.array([0.3, 0.9]))     # low box at 104, high box at 106
        self.assertEqual(ids.tolist(), [1])
        self.assertAlmostEqual(float(out[0][0]), 106.0, places=3)  # the high-score box continued track 1

    def test_far_detection_starts_a_new_track(self):
        t = ByteIouTracker(min_hits=1)
        step(t, 100, 0.9)
        ids = t.update(np.stack([box(102), box(400)]), np.array([0.9, 0.9]))[0]
        self.assertEqual(sorted(ids.tolist()), [1, 2])

    def test_empty_input(self):
        t = ByteIouTracker()
        ids, boxes, sc, co = t.update(np.zeros((0, 4)), np.zeros(0))
        self.assertEqual((ids.shape, boxes.shape, sc.shape, co.shape), ((0,), (0, 4), (0,), (0,)))


class GridAndSelectionTests(unittest.TestCase):
    def test_grid_is_the_preregistered_48(self):
        g = list(grid_v2())
        self.assertEqual(len(g), 48)
        self.assertEqual(len({tuple(sorted(p.items())) for p in g}), 48)
        self.assertEqual(set(g[0]), set(GRID_V2))
        for p in g:
            ByteIouTracker(**p)                                   # every setting is valid

    def test_rank_and_pick_rules(self):
        rows = [{"params": {"max_age": 30}, "hota": 0.50, "idsw": 300},
                {"params": {"max_age": 15}, "hota": 0.503, "idsw": 280},
                {"params": {"max_age": 5}, "hota": 0.52, "idsw": 400}]
        self.assertEqual(rank_candidates(rows)[0]["params"]["max_age"], 5)
        tied = [{"params": {"max_age": 30}, "hota": 0.520, "idsw": 300},
                {"params": {"max_age": 15}, "hota": 0.517, "idsw": 280},
                {"params": {"max_age": 5}, "hota": 0.516, "idsw": 280}]
        self.assertEqual(pick_final(tied)["params"]["max_age"], 5)    # within 0.005: lowest IDSW, then smaller max_age
        far = [{"params": {"max_age": 30}, "hota": 0.52, "idsw": 900}, {"params": {"max_age": 5}, "hota": 0.50, "idsw": 100}]
        self.assertEqual(pick_final(far)["params"]["max_age"], 30)    # 0.02 apart: HOTA decides


class ArmTests(unittest.TestCase):
    def test_build_arm_v2_runs_on_cached_style_detections(self):
        dets = []
        for f in range(10):
            x = 100.0 + 2 * f
            dets.append((np.stack([box(x), box(500)]).astype(np.float32), np.array([0.9, 0.2], dtype=np.float32)))
        preds, sc = build_arm_v2(dets, {"det_high": 0.6, "det_low": 0.1, "match_iou": 0.4, "max_age": 5, "min_hits": 2})
        self.assertEqual(len(preds), 10)
        self.assertEqual(len(preds[0][0]), 0)                         # first frame: tentative
        self.assertTrue(all(len(p[0]) == 1 for p in preds[1:]))       # the 0.2 box never starts a track
        self.assertEqual({int(i) for p in preds for i in p[0]}, {1})


if __name__ == "__main__":
    unittest.main()
