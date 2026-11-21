"""Unit tests for Tracker v3 (Deep-EIoU, GTA-Link, Kit-Color Official Filter) and feature chunking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from soccer_tracking.features import (
    assemble_features,
    extract_torso_lab,
    feature_chunk_complete,
    feature_chunk_path,
    save_feature_chunk,
)
from soccer_tracking.tracker_v3 import (
    DeepEIoUTracker,
    Tracklet,
    expand_boxes,
    filter_officials_and_roster,
    gta_link_connect,
    hungarian_match,
    pick_final_v3,
    run_tracker_v3,
)


class TestTrackerV3(unittest.TestCase):
    def test_expand_boxes_and_hungarian_match(self):
        b = np.array([[10.0, 20.0, 30.0, 60.0]])  # w=20, h=40
        exp = expand_boxes(b, 0.5)
        np.testing.assert_allclose(exp, [[5.0, 10.0, 35.0, 70.0]])

        cost = np.array([[0.2, 0.9], [0.8, 0.3]])
        self.assertEqual(hungarian_match(cost, max_cost=0.5), [(0, 0), (1, 1)])
        self.assertEqual(hungarian_match(cost, max_cost=0.25), [(0, 0)])

    def test_deep_eiou_survives_occlusion_with_turn(self):
        """Player disappears for 10 frames and reverses direction; Deep-EIoU keeps the same ID."""
        tr = DeepEIoUTracker(det_high=0.6, det_low=0.3, max_age=20, min_hits=2, app_weight=0.5)
        emb = np.zeros(8, dtype=np.float32)
        emb[0] = 1.0

        ids_seen = []
        for f in range(5):
            box = np.array([[100.0 + 4 * f, 200.0, 120.0 + 4 * f, 240.0]])
            ids, _, _ = tr.update(box, np.array([0.9]), embs=emb[None, :])
            if len(ids):
                ids_seen.append(int(ids[0]))
        self.assertEqual(len(set(ids_seen)), 1)
        orig_id = ids_seen[0]

        for _ in range(10):
            tr.update(np.zeros((0, 4)), np.zeros(0))

        box_back = np.array([[108.0, 202.0, 128.0, 242.0]])
        ids, _, _ = tr.update(box_back, np.array([0.9]), embs=emb[None, :])
        self.assertEqual(list(ids), [orig_id])

    def test_gta_link_merges_same_player_and_respects_gates(self):
        e1 = np.array([1.0, 0.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0, 0.0])
        red_lab = np.array([130.0, 180.0, 160.0])
        blue_lab = np.array([120.0, 100.0, 80.0])

        t1 = Tracklet(1, list(range(0, 20)), [np.array([100.0, 200.0, 120.0, 240.0])] * 20, [0.9] * 20, [e1] * 20, [red_lab] * 20)
        t2 = Tracklet(2, list(range(40, 60)), [np.array([140.0, 205.0, 160.0, 245.0])] * 20, [0.9] * 20, [e1] * 20, [red_lab] * 20)
        t3 = Tracklet(3, list(range(40, 60)), [np.array([130.0, 200.0, 150.0, 240.0])] * 20, [0.9] * 20, [e1] * 20, [blue_lab] * 20)
        t4 = Tracklet(4, list(range(10, 15)), [np.array([500.0, 200.0, 520.0, 240.0])] * 5, [0.8] * 5, [e2] * 5, [red_lab] * 5)

        merged = gta_link_connect({1: t1, 2: t2, 3: t3, 4: t4}, merge_cos_thresh=0.25, team_color_gate=30.0, min_track_len=10)
        self.assertEqual(set(merged.keys()), {1, 3})
        self.assertEqual(len(merged[1].frames), 40)
        self.assertEqual((merged[1].start_frame, merged[1].end_frame), (0, 59))

    def test_official_filter_drops_referee_and_keeps_teams_and_goalkeepers(self):
        W, H = 4096, 1080
        team_a = np.array([140.0, 185.0, 160.0])
        team_b = np.array([140.0, 95.0, 85.0])
        ref_kit = np.array([210.0, 115.0, 205.0])
        gk_kit = np.array([190.0, 70.0, 190.0])

        tracklets = {}
        tid = 1
        for i in range(10):
            x = 900.0 + i * 220.0
            tracklets[tid] = Tracklet(tid, list(range(60)), [np.array([x, 450.0, x + 24, 500.0])] * 60, [0.95] * 60, [], [team_a] * 60)
            tid += 1
        for i in range(10):
            x = 950.0 + i * 220.0
            tracklets[tid] = Tracklet(tid, list(range(60)), [np.array([x, 520.0, x + 24, 570.0])] * 60, [0.95] * 60, [], [team_b] * 60)
            tid += 1
        gk_left_id = tid
        tracklets[tid] = Tracklet(tid, list(range(60)), [np.array([200.0, 480.0, 224.0, 530.0])] * 60, [0.95] * 60, [], [gk_kit] * 60)
        tid += 1
        gk_right_id = tid
        tracklets[tid] = Tracklet(tid, list(range(60)), [np.array([3850.0, 480.0, 3874.0, 530.0])] * 60, [0.95] * 60, [], [gk_kit] * 60)
        tid += 1
        ref_id = tid
        tracklets[tid] = Tracklet(tid, list(range(60)), [np.array([2000.0, 490.0, 2024.0, 540.0])] * 60, [0.99] * 60, [], [ref_kit] * 60)
        tid += 1
        coach_id = tid
        tracklets[tid] = Tracklet(tid, list(range(60)), [np.array([2000.0, 50.0, 2024.0, 100.0])] * 60, [0.95] * 60, [], [team_a] * 60)

        kept, _ = filter_officials_and_roster(tracklets, W, H, official_lab_thresh=30.0, gk_margin=0.14, ref_min_len=30)
        self.assertIn(gk_left_id, kept)
        self.assertIn(gk_right_id, kept)
        self.assertNotIn(ref_id, kept)
        self.assertNotIn(coach_id, kept)
        self.assertEqual(len(kept), 22)

    def test_torso_lab_masks_green_pitch_and_feature_chunk_roundtrip(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :] = (30, 180, 40)
        img[60:110, 88:112] = (220, 30, 30)
        lab = extract_torso_lab(img, [80, 40, 120, 140])
        self.assertGreater(float(lab[1]), 160.0)

        plan = [(0, 3)]
        dets = [
            (np.array([[10, 10, 30, 50], [40, 10, 60, 50]], dtype=np.float32), np.array([0.9, 0.1], dtype=np.float32)),
            (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)),
            (np.array([[12, 10, 32, 50]], dtype=np.float32), np.array([0.85], dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            save_feature_chunk(
                tmp_path, "117093", 0,
                frames=[0, 2], det_indices=[0, 0],
                embs=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                torso_labs=np.array([[120.0, 170.0, 150.0], [125.0, 165.0, 145.0]], dtype=np.float32),
                n_frames=3, emb_dim=4,
            )
            self.assertTrue(feature_chunk_complete(feature_chunk_path(tmp_path, "117093", 0), 3))
            assembled, dim = assemble_features(tmp_path, "117093", plan, dets)
            self.assertEqual(dim, 4)
            self.assertEqual(list(assembled[0]["valid"]), [True, False])
            self.assertEqual(len(assembled[1]["valid"]), 0)
            self.assertEqual(list(assembled[2]["valid"]), [True])

    def test_run_tracker_v3_and_pick_final_v3(self):
        dets = [(np.array([[100.0 + f, 400.0, 120.0 + f, 440.0]]), np.array([0.9])) for f in range(20)]
        preds, sc = run_tracker_v3(dets, None, 4096, 1080, {"use_official_filter": False, "min_track_len": 5})
        self.assertEqual(len(preds), 20)
        self.assertEqual(len(sc), 20)
        self.assertEqual(len(preds[-1][0]), 1)

        rows = [
            {"hota": 0.550, "fp": 3000.0, "idsw": 200.0, "name": "a"},
            {"hota": 0.548, "fp": 2100.0, "idsw": 220.0, "name": "b"},
            {"hota": 0.540, "fp": 1500.0, "idsw": 100.0, "name": "c"},
        ]
        self.assertEqual(pick_final_v3(rows, tol=0.005)["name"], "b")


if __name__ == "__main__":
    unittest.main()
