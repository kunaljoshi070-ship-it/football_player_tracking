"""Unit tests for Phase II Product CLI Core (input validation, confidence flags, and cached-mode outputs)."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from soccer_tracking.fullclip import save_chunk
from soccer_tracking.phase2 import plan_file
from soccer_tracking.product import (
    InputValidationError,
    compute_track_annotations,
    run_cached_analysis,
    validate_video_input,
)


def _make_synthetic_mp4(path: Path, width: int = 640, height: int = 360, n_frames: int = 20, fps: float = 25.0) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for f in range(n_frames):
        frame = np.full((height, width, 3), (35, 120, 45), dtype=np.uint8)
        cv2.rectangle(frame, (100 + f * 3, 150), (124 + f * 3, 200), (220, 40, 40), -1)
        writer.write(frame)
    writer.release()
    return path


class TestProductCore(unittest.TestCase):
    def test_input_validation_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(InputValidationError):
                validate_video_input(root / "missing.mp4")

            bad_ext = root / "clip.txt"
            bad_ext.write_text("not a video", encoding="utf-8")
            with self.assertRaises(InputValidationError):
                validate_video_input(bad_ext)

            vid = _make_synthetic_mp4(root / "ok.mp4", n_frames=20)
            meta = validate_video_input(vid, start_frame=2, end_frame=15, max_segment_frames=20)
            self.assertEqual((meta.width, meta.height, meta.total_frames), (640, 360, 20))

            with self.assertRaises(InputValidationError):
                validate_video_input(vid, start_frame=15, end_frame=10)
            with self.assertRaises(InputValidationError):
                validate_video_input(vid, start_frame=0, end_frame=20, max_segment_frames=10)

    def test_confidence_tiers_and_flags(self):
        preds = [
            (np.array([1, 2]), np.array([[100, 100, 130, 160], [110, 100, 140, 160]], dtype=np.float32))
            for _ in range(35)
        ]
        scores = [np.array([0.92, 0.50], dtype=np.float32) for _ in range(35)]
        rows, stats = compute_track_annotations(preds, scores, det_high=0.78, crowd_iou=0.15)
        self.assertEqual(stats["unique_tracks"], 2)
        self.assertEqual(stats["track_confidence_counts"]["high"], 1)
        self.assertEqual(stats["track_confidence_counts"]["low"], 1)
        self.assertGreater(stats["crowded_box_events"], 0)
        self.assertEqual(stats["low_score_continuation_events"], 35)
        self.assertIn("crowded", rows[0][0]["flags"])
        self.assertIn("low_score", rows[0][1]["flags"])

    def test_end_to_end_cached_mode_writes_all_four_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vid = _make_synthetic_mp4(root / "117093.mp4", n_frames=20)
            det_dir = root / "dets"
            frames_arr, boxes_arr, scores_arr = [], [], []
            for f in range(20):
                frames_arr.append(f)
                boxes_arr.append([100 + f * 3, 150, 124 + f * 3, 200])
                scores_arr.append(0.95)
            save_chunk(det_dir, "117093", 0, frames_arr, boxes_arr, scores_arr, n_frames=20)
            plan_file(det_dir, "117093").write_text(
                json.dumps({"clip_id": "117093", "start": 0, "end": 20, "chunk": 20, "plan": [[0, 20]]}),
                encoding="utf-8",
            )
            out_dir = root / "out"
            summary = run_cached_analysis(vid, det_dir, "117093", out_dir, start_frame=0, end_frame=20,
                                          tracker_overrides={"min_track_len": 5})
            self.assertTrue((out_dir / "annotated.mp4").is_file())
            self.assertTrue((out_dir / "trajectories.csv").is_file())
            self.assertTrue((out_dir / "heatmap.png").is_file())
            self.assertTrue((out_dir / "summary.json").is_file())
            self.assertEqual(summary["stats"]["unique_tracks"], 1)
            with (out_dir / "trajectories.csv").open("r", encoding="utf-8") as fh:
                csv_rows = list(csv.DictReader(fh))
            self.assertGreaterEqual(len(csv_rows), 15)


if __name__ == "__main__":
    unittest.main()
