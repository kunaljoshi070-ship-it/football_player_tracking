"""Tests use only synthetic data generated in a temp dir (no dataset files).

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from soccer_tracking.mot import MotFormatError, load_mot, parse_mot_line
from soccer_tracking.visualize import annotation_frame_for, draw_boxes, main

GOOD = "0,1,10.5,20.5,30,40,-1,-1,-1,-1\r\n0,2,100,50,20,30,-1,-1,-1,-1\r\n1,1,11,21,30,40,-1,-1,-1,-1\r\n"


class MotLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, name="a.txt"):
        path = self.dir / name
        path.write_bytes(text.encode())  # bytes: keep CRLF exactly
        return path

    def test_valid_file_crlf(self):
        data = load_mot(self.write(GOOD))
        self.assertEqual(sorted(data), [0, 1])
        self.assertEqual(len(data[0]), 2)
        self.assertEqual(data[0][0].track_id, 1)
        self.assertAlmostEqual(data[0][0].x, 10.5)
        self.assertEqual(data[0][0].conf, -1.0)

    def test_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            load_mot(self.dir / "nope.txt")

    def test_empty_file(self):
        with self.assertRaisesRegex(MotFormatError, "no annotation rows"):
            load_mot(self.write("\r\n\r\n"))

    def test_malformed_rows_report_line_number(self):
        for bad in ("0,1,10,20,30\n", "a,1,10,20,30,40\n", "0,1,x,20,30,40\n", "0.5,1,10,20,30,40\n"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(MotFormatError, r"a\.txt:2:"):
                    load_mot(self.write("0,1,1,1,5,5\n" + bad))

    def test_invalid_boxes(self):
        for bad in ("0,1,10,20,0,40", "0,1,10,20,30,-5", "0,1,nan,20,30,40",
                    "0,1,10,inf,30,40", "-1,1,10,20,30,40"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_mot_line(bad)

    def test_duplicate_frame_id(self):
        with self.assertRaisesRegex(MotFormatError, "duplicate"):
            load_mot(self.write("0,1,1,1,5,5\n0,1,2,2,5,5\n"))


class FrameMappingTests(unittest.TestCase):
    def test_explicit_mapping(self):
        self.assertEqual(annotation_frame_for(0, 0), 0)
        self.assertEqual(annotation_frame_for(0, 1), 1)
        self.assertEqual(annotation_frame_for(99, 1), 100)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            annotation_frame_for(0, 2)
        with self.assertRaises(ValueError):
            annotation_frame_for(-1, 0)


class DrawTests(unittest.TestCase):
    def test_draw_and_clip(self):
        img = np.zeros((100, 200, 3), np.uint8)
        dets = [parse_mot_line("0,1,10,10,20,30"),       # inside
                parse_mot_line("0,2,190,90,50,50"),      # partly outside -> clipped
                parse_mot_line("0,3,500,500,10,10")]     # fully outside -> skipped
        out, drawn, clipped = draw_boxes(img, dets)
        self.assertEqual((drawn, clipped), (2, 2))
        self.assertEqual(int(img.sum()), 0)              # input untouched
        self.assertGreater(int(out.sum()), 0)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "v.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (320, 180))
        if not writer.isOpened():
            self.skipTest("OpenCV cannot write a test video here")
        for i in range(5):
            frame = np.full((180, 320, 3), 40, np.uint8)
            cv2.putText(frame, str(i), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            writer.write(frame)
        writer.release()
        self.ann = self.dir / "v.txt"
        self.ann.write_text("0,1,20,20,30,60\n2,1,40,20,30,60\n2,2,100,50,20,40\n4,1,60,20,30,60\n")
        self.out = self.dir / "out" / "f.jpg"

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, frame, base=0, extra=()):
        return main(["--video", str(self.video), "--annotations", str(self.ann), "--frame",
                     str(frame), "--annotation-base", str(base), "--output", str(self.out), *extra])

    def test_zero_based_default(self):
        self.assertEqual(self.run_cli(2), 0)
        self.assertEqual(cv2.imread(str(self.out)).shape, (180, 320, 3))

    def test_sequential_option(self):
        self.assertEqual(self.run_cli(2, extra=["--sequential"]), 0)

    def test_base_one_shifts_lookup_explicitly(self):
        self.assertEqual(self.run_cli(1, base=1), 0)      # video frame 1 -> annotation frame 2
        self.assertEqual(self.run_cli(2, base=1), 2)      # annotation frame 3 does not exist

    def test_errors_return_2(self):
        self.assertEqual(self.run_cli(1), 2)              # no annotations at frame 1
        self.assertEqual(self.run_cli(99), 2)             # beyond end of video / annotations
        self.assertEqual(main(["--video", str(self.dir / "x.mp4"), "--annotations", str(self.ann),
                               "--frame", "0"]), 2)
        self.assertEqual(main(["--video", str(self.video), "--annotations", str(self.dir / "x.txt"),
                               "--frame", "0"]), 2)


if __name__ == "__main__":
    unittest.main()
