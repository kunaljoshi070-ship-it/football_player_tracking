"""Synthetic-data tests for tiling, manifest checks, frame extraction and the tile dataset.

Run from the repo root:  PYTHONPATH=src python -m unittest discover -s tests -v
The torch test is skipped if torch is not installed.
"""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from soccer_tracking.data import (Clip, ManifestError, SoccerTilesDataset, audit, check_manifest,
                                  extract_frames, load_manifest, main, sample_frames)
from soccer_tracking.tiles import boxes_to_tile, tile_grid, tile_starts

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TileTests(unittest.TestCase):
    def test_starts_cover_axis(self):
        self.assertEqual(tile_starts(4096, 1024, 128), [0, 896, 1792, 2688, 3072])
        self.assertEqual(tile_starts(1080, 1080, 128), [0])
        self.assertEqual(tile_starts(640, 256, 32), [0, 224, 384])
        for length, tile, ov in ((4096, 1024, 128), (3840, 1000, 100), (1906, 1080, 128)):
            starts = tile_starts(length, tile, ov)
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + tile, length)
            self.assertTrue(all(b - a <= tile - ov for a, b in zip(starts, starts[1:])))  # no gaps

    def test_invalid_args(self):
        for args in ((100, 0, 0), (100, 10, 10), (100, 10, -1)):
            with self.assertRaises(ValueError):
                tile_starts(*args)

    def test_grid_shapes(self):
        g = tile_grid(4096, 1080, 1024, 1080, 128)
        self.assertEqual(len(g), 5)
        self.assertTrue(all(w[3] - w[1] == 1080 and w[2] - w[0] == 1024 for w in g))
        self.assertEqual(len(tile_grid(3840, 1906, 1024, 1080, 128)), 5 * 2)
        self.assertEqual(tile_grid(100, 50, 1024, 1080, 0), [(0, 0, 100, 50)])  # tile larger than frame

    def test_boxes_clip_shift_and_filter(self):
        win = (100, 0, 200, 100)
        boxes = np.array([[110, 10, 20, 30],    # inside
                          [90, 10, 20, 30],     # half inside -> kept at 0.5, clipped to x=100
                          [85, 10, 20, 30],     # 25% inside -> dropped
                          [300, 10, 20, 30],    # outside
                          [150, 10, 2, 30]])    # too narrow
        out, kept = boxes_to_tile(boxes, win, min_visible=0.5, min_size=4.0)
        self.assertEqual(kept.tolist(), [0, 1])
        np.testing.assert_allclose(out, [[10, 10, 30, 40], [0, 10, 10, 40]])
        self.assertEqual(out.dtype, np.float32)

    def test_visibility_is_relative_to_frame(self):
        # box half outside the frame at the right edge: fully visible inside the frame -> kept at 0.99
        out, kept = boxes_to_tile(np.array([[190, 10, 20, 30]]), (100, 0, 200, 100), 0.99, 4.0, frame_size=(200, 100))
        self.assertEqual(kept.tolist(), [0])
        out, kept = boxes_to_tile(np.array([[190, 10, 20, 30]]), (100, 0, 200, 100), 0.99, 4.0)
        self.assertEqual(len(kept), 0)

    def test_empty(self):
        out, kept = boxes_to_tile(np.zeros((0, 4)), (0, 0, 10, 10))
        self.assertEqual(out.shape, (0, 4))


def write_manifest(path, rows):
    cols = ["clip_id", "split", "local_path", "annotation_path", "width", "height", "fps", "frames", "sha256", "annotation_sha256"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, clip_id, split, **kw):
        base = dict(clip_id=clip_id, split=split, local_path=f"{clip_id}.mp4", annotation_path=f"{clip_id}.txt",
                    width=640, height=360, fps=25, frames=12)
        base.update(kw)
        return base

    def test_load_and_filter(self):
        m = self.dir / "m.csv"
        write_manifest(m, [self.row("a", "train"), self.row("b", "val"), self.row("c", "unused")])
        self.assertEqual([c.clip_id for c in load_manifest(m, ["train", "val"])], ["a", "b"])
        with self.assertRaises(ManifestError):
            load_manifest(m, ["test"])

    def test_duplicate_clip_across_splits_is_flagged(self):
        m = self.dir / "m.csv"
        write_manifest(m, [self.row("a", "train"), self.row("a", "test")])
        with self.assertRaisesRegex(ManifestError, "exactly one split"):
            load_manifest(m)

    def test_bad_split_missing_file_and_bad_numbers(self):
        m = self.dir / "m.csv"
        write_manifest(m, [self.row("a", "training")])
        with self.assertRaisesRegex(ManifestError, "not in"):
            load_manifest(m)
        write_manifest(m, [self.row("a", "train", width="x")])
        with self.assertRaisesRegex(ManifestError, "bad row"):
            load_manifest(m)
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.dir / "none.csv")
        self.assertTrue(check_manifest([Clip("a", "train", Path("a"), Path("a"), 0, 1, 25, 1)]))

    def test_sample_frames(self):
        self.assertEqual(list(sample_frames(100, 25)), [0, 25, 50, 75])
        self.assertEqual(list(sample_frames(100, 25, start=10, end=60)), [10, 35])
        with self.assertRaises(ValueError):
            sample_frames(100, 0)
        with self.assertRaises(ValueError):
            sample_frames(100, 5, start=100)


class PipelineTests(unittest.TestCase):
    N, W, H = 12, 640, 360

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "a.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (self.W, self.H))
        if not writer.isOpened():
            self.skipTest("OpenCV cannot write a test video here")
        for i in range(self.N):
            img = np.full((self.H, self.W, 3), 30, np.uint8)
            cv2.rectangle(img, (50, 100), (80, 160), (0, 255, 0), -1)
            cv2.putText(img, str(i), (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            writer.write(img)
        writer.release()
        self.ann = self.dir / "a.txt"
        self.ann.write_text("".join(f"{f},1,50,100,30,60,-1,-1,-1,-1\n{f},2,300,200,40,80,-1,-1,-1,-1\n"
                                    for f in range(self.N)))
        n = int(cv2.VideoCapture(str(self.video)).get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertEqual(n, self.N)
        self.clip = Clip("a", "train", self.video, self.ann, self.W, self.H, 5.0, self.N)
        self.root = self.dir / "frames"

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_stride_log_and_skip_existing(self):
        self.assertEqual(extract_frames(self.clip, self.root, stride=5), 3)       # frames 0, 5, 10
        self.assertEqual(sorted(p.name for p in (self.root / "a").glob("*.jpg")),
                         ["000000.jpg", "000005.jpg", "000010.jpg"])
        self.assertTrue((self.root / "a" / "extract_log.json").is_file())
        self.assertEqual(extract_frames(self.clip, self.root, stride=5), 0)       # nothing new
        bad = Clip("a", "train", self.video, self.ann, self.W + 1, self.H, 5.0, self.N)
        with self.assertRaisesRegex(ValueError, "manifest says"):
            extract_frames(bad, self.dir / "other", stride=5)
        wrong_count = Clip("a", "train", self.video, self.ann, self.W, self.H, 5.0, self.N + 3)
        with self.assertRaisesRegex(ValueError, "reports"):
            extract_frames(wrong_count, self.dir / "other2", stride=5)

    def make_ds(self, **kw):
        extract_frames(self.clip, self.root, stride=5)
        args = dict(stride=5, tile=(256, 360), overlap=32, min_size=4.0)
        args.update(kw)
        return SoccerTilesDataset([self.clip], self.root, **args)

    def test_dataset_content_and_boxes(self):
        ds = self.make_ds()
        self.assertEqual(len(ds), 3 * 3)                                          # 3 frames x 3 tiles
        img, boxes, meta = ds.get_numpy(0)                                        # frame 0, window (0,0,256,360)
        self.assertEqual(img.shape, (360, 256, 3))
        self.assertEqual(meta["window"], (0, 0, 256, 360))
        np.testing.assert_allclose(boxes, [[50, 100, 80, 160]])                   # only player 1 in this tile
        self.assertEqual(meta["track_ids"].tolist(), [1])
        saved = cv2.cvtColor(cv2.imread(str(self.root / "a" / "000000.jpg")), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(img, saved[:, 0:256])                       # tile == crop of the stored frame
        _, boxes2, meta2 = ds.get_numpy(1)                                        # window (224, 0, 480, 360)
        np.testing.assert_allclose(boxes2, [[300 - 224, 200, 340 - 224, 280]])
        self.assertEqual(meta2["track_ids"].tolist(), [2])
        _, boxes3, _ = ds.get_numpy(2)                                            # window (384, 0, 640, 360): empty tile
        self.assertEqual(boxes3.shape, (0, 4))

    def test_flip_is_consistent_and_deterministic(self):
        plain = self.make_ds()
        aug = SoccerTilesDataset([self.clip], self.root, stride=5, tile=(256, 360), overlap=32, augment=True, seed=7)
        flips = [aug.get_numpy(i)[2]["flipped"] for i in range(len(aug))]
        self.assertTrue(any(flips) and not all(flips) or len(flips) < 4)
        again = SoccerTilesDataset([self.clip], self.root, stride=5, tile=(256, 360), overlap=32, augment=True, seed=7)
        self.assertEqual(flips, [again.get_numpy(i)[2]["flipped"] for i in range(len(again))])
        aug.set_epoch(1)
        self.assertNotEqual(flips, [aug.get_numpy(i)[2]["flipped"] for i in range(len(aug))] if len(flips) > 8 else [not f for f in flips])
        for i in range(len(aug)):
            img_f, b_f, m = aug.get_numpy(i)
            img_p, b_p, _ = plain.get_numpy(i)
            if m["flipped"]:
                np.testing.assert_array_equal(img_f, img_p[:, ::-1])
                w = img_p.shape[1]
                np.testing.assert_allclose(b_f, np.stack([w - b_p[:, 2], b_p[:, 1], w - b_p[:, 0], b_p[:, 3]], 1) if len(b_p) else b_p)

    def test_missing_frames_and_bad_annotations(self):
        with self.assertRaisesRegex(FileNotFoundError, "extract"):
            SoccerTilesDataset([self.clip], self.dir / "empty", stride=5)
        short = Clip("a", "train", self.video, self.ann, self.W, self.H, 5.0, self.N + 1)
        extract_frames(self.clip, self.root, stride=5)
        with self.assertRaisesRegex(ManifestError, "do not match"):
            SoccerTilesDataset([short], self.root, stride=5)

    def test_audit_and_cli(self):
        buf = io.StringIO()
        audit([self.clip], 5, (256, 360), 32, 0.5, 4.0, out=lambda s: buf.write(s + "\n"))
        text = buf.getvalue()
        self.assertIn("[train]", text)
        self.assertIn("100.00%", text)  # every synthetic box is fully inside some tile
        m = self.dir / "m.csv"
        write_manifest(m, [dict(clip_id="a", split="train", local_path=str(self.video), annotation_path=str(self.ann),
                                width=self.W, height=self.H, fps=5, frames=self.N)])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["extract", "--manifest", str(m), "--splits", "train", "--stride", "5",
                                   "--out", str(self.dir / "cli_frames")]), 0)
            self.assertEqual(main(["audit", "--manifest", str(m), "--splits", "train", "--stride", "5",
                                   "--tile", "256", "360", "--overlap", "32"]), 0)
            self.assertEqual(main(["audit", "--manifest", str(m), "--splits", "test"]), 2)   # no test clips
            self.assertEqual(main(["audit", "--manifest", str(self.dir / "nope.csv")]), 2)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_torch_item_and_loader(self):
        import torch
        from torch.utils.data import DataLoader
        from soccer_tracking.data import collate
        ds = self.make_ds()
        image, target = ds[0]
        self.assertEqual(tuple(image.shape), (3, 360, 256))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(0.0 <= float(image.min()) and float(image.max()) <= 1.0)
        self.assertEqual(tuple(target["boxes"].shape), (1, 4))
        self.assertEqual(target["labels"].dtype, torch.int64)
        images, targets = next(iter(DataLoader(ds, batch_size=3, collate_fn=collate)))
        self.assertEqual(len(images), 3)
        self.assertEqual(tuple(targets[2]["boxes"].shape), (0, 4))


if __name__ == "__main__":
    unittest.main()
