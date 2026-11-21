import unittest

import numpy as np

from soccer_tracking.roster import GRID_C4, build_arm_roster, cap_detections, cap_output, pick_final_c4


def frame(scores):
    n = len(scores)
    b = np.stack([np.arange(n) * 50.0, np.full(n, 100.0), np.arange(n) * 50.0 + 20, np.full(n, 140.0)], 1)
    return b, np.array(scores, dtype=float)


class CapTests(unittest.TestCase):
    def test_grid_has_seven_including_the_comparison_arm(self):
        self.assertEqual(len(GRID_C4), 7)
        self.assertEqual(GRID_C4[0], ("none", None))

    def test_cap_detections_keeps_top_k_in_original_order(self):
        out = cap_detections([frame([0.4, 0.9, 0.2, 0.8])], 2)[0]
        self.assertEqual(out[1].tolist(), [0.9, 0.8])
        self.assertEqual(out[0][:, 0].tolist(), [50.0, 150.0])
        self.assertEqual(len(cap_detections([frame([0.4, 0.9])], 5)[0][1]), 2)
        self.assertEqual(len(cap_detections([frame([0.4, 0.9])], None)[0][1]), 2)

    def test_cap_output_prefers_old_tracks(self):
        b = np.array([[0, 0, 10, 10], [50, 0, 60, 10], [100, 0, 110, 10]], dtype=float)
        preds = [(np.array([1, 2]), b[:2]), (np.array([1, 2, 3]), b)]
        scores = [np.array([0.6, 0.6]), np.array([0.6, 0.6, 0.99])]    # the new track has the best score but is youngest
        p, s = cap_output(preds, scores, 2)
        self.assertEqual(p[1][0].tolist(), [1, 2])
        self.assertEqual(len(p[0][0]), 2)                            # frame 0 untouched (2 <= k)

    def test_cap_output_ties_on_age_go_to_higher_score(self):
        b = np.array([[0, 0, 10, 10], [50, 0, 60, 10], [100, 0, 110, 10]], dtype=float)
        p, s = cap_output([(np.array([1, 2, 3]), b)], [np.array([0.5, 0.9, 0.7])], 2)
        self.assertEqual(p[0][0].tolist(), [2, 3])
        self.assertEqual(s[0].tolist(), [0.9, 0.7])

    def test_none_is_identity(self):
        p, s = cap_output([(np.array([1]), np.zeros((1, 4)))], [np.array([0.9])], None)
        self.assertEqual(p[0][0].tolist(), [1])

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            build_arm_roster([], {}, "bogus", 22)

    def test_pick_final_rule(self):
        rows = [{"mode": "none", "k": None, "hota": 0.530, "idsw": 270},
                {"mode": "out", "k": 22, "hota": 0.533, "idsw": 270},
                {"mode": "det", "k": 24, "hota": 0.532, "idsw": 250}]
        self.assertEqual(pick_final_c4(rows)["k"], 24)                 # all within 0.005: lowest IDSW
        tied = [{"mode": "none", "k": None, "hota": 0.53, "idsw": 250}, {"mode": "out", "k": 22, "hota": 0.53, "idsw": 250}]
        self.assertEqual(pick_final_c4(tied)["mode"], "none")          # tie on IDSW: larger k (none) wins
        far = [{"mode": "none", "k": None, "hota": 0.50, "idsw": 100}, {"mode": "out", "k": 22, "hota": 0.52, "idsw": 900}]
        self.assertEqual(pick_final_c4(far)["k"], 22)                  # 0.02 apart: HOTA decides


class ArmTests(unittest.TestCase):
    def test_roster_arm_runs_and_cap_limits_output(self):
        params = {"det_high": 0.6, "det_low": 0.1, "match_iou": 0.4, "max_age": 5, "min_hits": 2}
        dets = []
        for f in range(8):
            b = np.stack([[100.0 + 100 * i + 2 * f, 100, 120.0 + 100 * i + 2 * f, 140] for i in range(5)]).astype(np.float32)
            dets.append((b, np.array([0.9, 0.9, 0.9, 0.8, 0.7], dtype=np.float32)))
        full, _ = build_arm_roster(dets, params, "none", None)
        capped, _ = build_arm_roster(dets, params, "out", 3)
        self.assertEqual(len(full[-1][0]), 5)
        self.assertTrue(all(len(p[0]) <= 3 for p in capped))
        det_capped, _ = build_arm_roster(dets, params, "det", 3)
        self.assertEqual(len(det_capped[-1][0]), 3)


if __name__ == "__main__":
    unittest.main()