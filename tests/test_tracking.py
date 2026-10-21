"""Tests for the IoU tracker, the tracking measures and the track_eval commands (no torch needed).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from soccer_tracking.track_eval import main
from soccer_tracking.track_metrics import score_sequence, summarize
from soccer_tracking.tracker import IouTracker, greedy_match


def box(x, y=100.0, w=20.0, h=40.0):
    return np.array([x, y, x + w, y + h], dtype=float)


class TrackerTests(unittest.TestCase):
    def test_greedy_match(self):
        iou = np.array([[0.9, 0.2], [0.8, 0.1]])
        self.assertEqual(greedy_match(iou, 0.3), [(0, 0)])           # row 1 loses column 0 and is below threshold elsewhere
        self.assertEqual(greedy_match(np.zeros((0, 3)), 0.3), [])

    def test_invalid_params(self):
        for kw in ({"iou_thr": 0.0}, {"max_age": -1}, {"min_hits": 0}):
            with self.assertRaises(ValueError):
                IouTracker(**kw)

    def test_identity_is_kept_while_moving(self):
        t = IouTracker()
        ids = [t.update(box(100 + 2 * k)[None], np.array([0.9]))[0].tolist() for k in range(8)]
        self.assertEqual(ids, [[1]] * 8)

    def test_coasting_bridges_a_gap_and_expiry_starts_a_new_id(self):
        def run(max_age):
            t = IouTracker(max_age=max_age)
            out = []
            for k in range(7):
                if k == 4:
                    out.append(t.update(np.zeros((0, 4)), np.zeros(0)))     # missed detection
                else:
                    out.append(t.update(box(100 + 2 * k)[None], np.array([0.9])))
            return out
        out = run(max_age=2)
        self.assertEqual(out[4][0].tolist(), [1])
        self.assertTrue(out[4][3][0])                                       # output is flagged as coasted
        self.assertTrue(106.0 <= out[4][1][0][0] <= 108.0)                  # extrapolated forward, not frozen
        self.assertEqual(out[5][0].tolist(), [1])                           # same id after the gap
        self.assertFalse(out[5][3][0])
        out0 = run(max_age=0)
        self.assertEqual(len(out0[4][0]), 0)                                # nothing is coasted
        self.assertEqual(out0[5][0].tolist(), [2])                          # track died, so a new id starts

    def test_coast_expires_after_max_age(self):
        t = IouTracker(max_age=2)
        t.update(box(100)[None], np.array([0.9]))
        lens = [len(t.update(np.zeros((0, 4)), np.zeros(0))[0]) for _ in range(4)]
        self.assertEqual(lens, [1, 1, 0, 0])

    def test_far_detection_gets_a_new_id(self):
        t = IouTracker()
        t.update(box(100)[None], np.array([0.9]))
        ids = t.update(np.stack([box(102), box(400)]), np.array([0.9, 0.9]))[0]
        self.assertEqual(sorted(ids.tolist()), [1, 2])

    def test_min_hits_delays_output(self):
        t = IouTracker(min_hits=2)
        self.assertEqual(len(t.update(box(100)[None], np.array([0.9]))[0]), 0)
        self.assertEqual(t.update(box(102)[None], np.array([0.9]))[0].tolist(), [1])


def gt_still(n):
    return [(np.array([7]), np.array([[0.0, 0.0, 10.0, 20.0]])) for _ in range(n)]


def pred(ids_list, present):
    out = []
    for k in range(len(present)):
        if present[k]:
            ids = None if ids_list is None else np.array([ids_list[k]])
            out.append((ids, np.array([[0.0, 0.0, 10.0, 20.0]])))
        else:
            out.append((None if ids_list is None else np.zeros(0, dtype=int), np.zeros((0, 4))))
    return out


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        s = summarize([score_sequence(gt_still(10), pred([1] * 10, [True] * 10))])
        self.assertEqual((s["tp"], s["fp"], s["fn"], s["gaps"], s["idsw"]), (10, 0, 0, 0, 0))
        self.assertAlmostEqual(s["ids_per_track"], 1.0)
        self.assertAlmostEqual(s["f1"], 1.0)
        self.assertAlmostEqual(s["mota"], 1.0)

    def test_interior_gap_counts_once_but_edge_misses_do_not(self):
        present = [True] * 4 + [False] * 2 + [True] * 4
        s = summarize([score_sequence(gt_still(10), pred(None, present))])      # detector-only (no ids)
        self.assertEqual((s["fn"], s["gaps"]), (2, 1))
        self.assertAlmostEqual(s["frag_per_track"], 1.0)
        self.assertIsNone(s["idsw"])
        self.assertIsNone(s["mota"])
        edges = [False] + [True] * 8 + [False]
        s2 = summarize([score_sequence(gt_still(10), pred(None, edges))])
        self.assertEqual((s2["fn"], s2["gaps"]), (2, 0))

    def test_two_gaps(self):
        present = [True, False, True, False, True]
        s = summarize([score_sequence(gt_still(5), pred(None, present))])
        self.assertEqual(s["gaps"], 2)

    def test_id_switch_and_ids_per_track(self):
        s = summarize([score_sequence(gt_still(10), pred([1] * 5 + [2] * 5, [True] * 10))])
        self.assertEqual(s["idsw"], 1)
        self.assertAlmostEqual(s["ids_per_track"], 2.0)
        self.assertAlmostEqual(s["mota"], 1 - 1 / 10)

    def test_same_id_after_a_gap_is_not_a_switch(self):
        present = [True, True, False, True, True]
        s = summarize([score_sequence(gt_still(5), pred([1, 1, 1, 1, 1], present))])
        self.assertEqual((s["idsw"], s["gaps"]), (0, 1))

    def test_false_positives_and_summing_segments(self):
        gt = gt_still(4)
        extra = [(np.array([1, 2]), np.array([[0.0, 0.0, 10.0, 20.0], [200.0, 200.0, 210.0, 220.0]])) for _ in range(4)]
        a = score_sequence(gt, extra)
        self.assertEqual((a["tp"], a["fp"]), (4, 4))
        s = summarize([a, a])
        self.assertEqual((s["tp"], s["fp"], s["n_gt"], s["n_tracks"]), (8, 8, 8, 2))
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["f1"], 2 * 0.5 * 1.0 / 1.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            score_sequence(gt_still(3), pred(None, [True, True]))

    def test_tracker_coasting_reduces_fragmentation(self):
        n = 12
        gt = [(np.array([7]), box(100 + 2 * k)[None]) for k in range(n)]
        dets = [box(100 + 2 * k)[None] if k not in (5, 6) else np.zeros((0, 4)) for k in range(n)]
        det_only = summarize([score_sequence(gt, [(None, d) for d in dets])])
        t = IouTracker(max_age=3)
        preds = []
        for d in dets:
            ids, boxes, _, _ = t.update(d, np.full(len(d), 0.9))
            preds.append((ids, boxes))
        trk = summarize([score_sequence(gt, preds)])
        self.assertEqual((det_only["gaps"], det_only["fn"]), (1, 2))
        self.assertEqual((trk["gaps"], trk["fn"], trk["idsw"]), (0, 0, 0))


def write_fixture(root, n=60):
    """Manifest + annotations + saved detections for one validation clip (2 players, 3 missed detections)."""
    ann = root / "v.txt"
    rows, det_frame, det_boxes, det_scores = [], [], [], []
    for f in range(n):
        for pid, x0, vx in ((1, 100.0, 2.0), (2, 300.0, 1.0)):
            x = x0 + vx * f
            rows.append(f"{f},{pid},{x},100,20,40,-1,-1,-1,-1\n")
            if not (pid == 1 and f in (10, 11, 30)):
                det_frame.append(f); det_boxes.append([x, 100, x + 20, 140]); det_scores.append(0.9)
        det_frame.append(f); det_boxes.append([500, 200, 520, 240]); det_scores.append(0.1)      # low score: filtered out
    ann.write_text("".join(rows))
    m = root / "m.csv"
    with open(m, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "split", "local_path", "annotation_path", "width", "height", "fps", "frames"])
        w.writerow(["v", "val", str(root / "v.mp4"), str(ann), 640, 360, 25, n])
    ddir = root / "dets"
    ddir.mkdir()
    np.savez_compressed(ddir / "v_0.npz", start=0, n_frames=n, frame=np.array(det_frame, dtype=np.int32),
                        boxes=np.array(det_boxes, dtype=np.float32), scores=np.array(det_scores, dtype=np.float32))
    return m, ddir


class CommandTests(unittest.TestCase):
    def test_tune_then_evaluate_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, ddir = write_fixture(root)
            common = ["--manifest", str(m), "--det-dir", str(ddir), "--starts", "0", "--seg-len", "60"]
            tuning = root / "tuning.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["tune", "--split", "val", *common, "--out", str(tuning)]), 0)
                rec = json.loads(tuning.read_text())
                self.assertEqual(rec["detector_only"]["gaps"], 2)                       # gap at frames 10-11 and at 30
                self.assertAlmostEqual(rec["detector_only"]["frag_per_track"], 1.0)
                self.assertTrue(rec["constraint_met"])
                self.assertEqual(len(rec["grid_results"]), 4 * 4 * 2 * 2)
                chosen = next(r for r in rec["grid_results"] if r["params"] == rec["chosen"])
                self.assertEqual(chosen["summary"]["gaps"], 0)
                out = root / "eval.json"
                self.assertEqual(main(["evaluate", "--split", "val", *common, "--params", str(tuning), "--out", str(out)]), 0)
                crit = json.loads(out.read_text())["criteria"]
                self.assertTrue(crit["fragmentation_ok"] and crit["f1_ok"])
                self.assertAlmostEqual(crit["frag_reduction_relative"], 1.0)
                for rec_json in (rec, json.loads(out.read_text())):                      # tune and evaluate both store it
                    rows = rec_json["per_segment"]
                    self.assertEqual(len(rows), 1)                                         # one segment in this fixture
                    self.assertEqual(rows[0]["detector_only"]["gaps"], rec["detector_only"]["gaps"])
                    self.assertEqual((rows[0]["clip_id"], rows[0]["start"]), ("v", 0))
                self.assertEqual(json.loads(out.read_text())["per_segment"][0]["tracker"]["gaps"], 0)
                # guards: no tuning on test, no test run without the flag, no detect on test without the flag
                self.assertEqual(main(["tune", "--split", "test", *common]), 2)
                self.assertEqual(main(["evaluate", "--split", "test", *common, "--params", str(tuning)]), 2)
                self.assertEqual(main(["detect", "--split", "test", "--weights", "x.pt", *common]), 2)
                self.assertEqual(main(["evaluate", "--split", "val", *common, "--params", str(root / "none.json")]), 2)
            self.assertIn("held out", err.getvalue())

    def test_segment_beyond_clip_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, ddir = write_fixture(root)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = main(["tune", "--manifest", str(m), "--det-dir", str(ddir), "--starts", "30", "--seg-len", "60",
                           "--out", str(root / "t.json")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
