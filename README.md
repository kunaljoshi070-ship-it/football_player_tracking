# CSAI3027P Deep Learning PBL — Project 36

**Project:** Sports Player Tracking and Tactical Event Analyzer
**Language:** Python (professor-approved)
**Current phase:** Phase I only

## Phase I scope

Use a small, fixed subset of soccer video to build one player-detection baseline and, if matching tracking labels are available, visualize short player trajectories. Compare a simple tracker with independent frame detections. Keep the result descriptive; do not claim player identity or full tactical understanding.

The selected dataset is **SoccerTrack v2, release v1.2**, using only its annotated MOT subset (ten 4-minute clips, about 3.5 GB; the full release is 145 GB and is not needed). Phase I uses four whole matches: two for training, one for validation and one held-out test, frozen in [`data/manifest.csv`](data/manifest.csv) and explained in [`docs/dataset_card.md`](docs/dataset_card.md), which you should read before accessing data. Raw video, frames, model weights, and secrets stay out of Git.

## Project files

- [`AGENTS.md`](AGENTS.md): shared rules for any person or AI tool working on this repository.
- [`docs/phase1_proposal.md`](docs/phase1_proposal.md): one-page proposal and experiment hypothesis.
- [`docs/dataset_card.md`](docs/dataset_card.md): source, access, license, and sample plan.
- [`data/manifest.csv`](data/manifest.csv): record only the selected dataset version, splits, clip IDs, and permitted metadata.
- [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md): individual and prompt-wise contribution record required by the assignment.

## Phase split

The handout scores Phase I and Phase II at 50 marks each. Phase I focuses on problem framing, data audit/splits, a working baseline and initial metric, architecture/training explanation, reproducibility, and an experiment plan. Keep Phase I to one detector and a small tracking comparison. The compendium's additional segmentation, action/event, second-architecture, sequence-model, and VAE work is for Phase II unless the instructor directs otherwise.

The compendium names C/C++/Java generically, while the professor has permitted Python for this project. Keep the professor's approval available for the viva and report.

## Minimal workflow

1. Confirm access to the SoccerTrack v2 files and verify annotations match the selected videos.
2. Freeze a small sample and split by match in `data/manifest.csv` before evaluating.
3. Build the Python/PyTorch detector baseline and record its initial metric and training settings.
4. Compare detections with a simple tracker on the same frames; report results and errors.
5. Record each external source and every permitted generative-AI prompt in `CONTRIBUTIONS.md`.

Use regular ChatGPT Chat for quick planning and prompt drafting. Use this repository as the source of truth; manually pass Claude/Gemini only safe project context and review their output yourself. The assignment requires faculty permission for generative-AI use.

## Data loading and visualization utility

`src/soccer_tracking/mot.py` reads and validates MOT rows (`frame,id,x,y,width,height,...`; `(x, y)` is the box's top-left corner in pixels). `src/soccer_tracking/visualize.py` saves one video frame with its annotated boxes and track IDs. No new dependencies: it uses `opencv-python` and `numpy` from `requirements.txt`.

Place each clip and its label file (git-ignored) in `data/raw/` as `<id>.mp4` and `<id>.txt`; the clip ids and their Phase I roles are in `data/manifest.csv`. Then run from the repository root (example for training clip `118575`):

```powershell
$env:PYTHONPATH = "src"
python -m soccer_tracking.visualize --video data/raw/118575.mp4 --annotations data/raw/118575.txt --frame 0
# writes data/processed/118575_frame0.jpg (git-ignored); use --output to choose another path
```

**Frame numbering.** `--frame` is a **0-based video frame index** (OpenCV convention). The annotation frame looked up is `--frame + --annotation-base`. `--annotation-base` is `0` by default because all ten downloaded label files were measured to start at frame 0 (for example `117092.txt`: frames 0–5999, 22 boxes per frame); pass `1` only for a 1-based MOT file. Nothing is shifted unless you ask, and the mapping used is printed on every run. By default OpenCV *seeks* to the frame, which can be imprecise for some codecs; add `--sequential` for slow but frame-exact decoding when checking alignment.

Errors (missing file, malformed row with its line number, non-positive/non-finite box, duplicate `(frame, id)`, frame without annotations, frame beyond the video) exit with code 2 and a message. Boxes outside the image are clipped and reported.

Tests use only synthetic data generated at test time:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Detector data pipeline

`src/soccer_tracking/data.py` (with `tiles.py`) turns the manifest into training tiles. Players are small (in 117092 the median box is about 20 x 41 px), and resizing a full 4096x1080 frame to a detector's default input would shrink them to about 7 x 14 px, so frames are cut into **native-scale tiles** instead.

```powershell
$env:PYTHONPATH = "src"
python -m soccer_tracking.data audit      # manifest checks + box-size and tiling statistics (add --verify-hashes to re-hash files)
python -m soccer_tracking.data extract    # decode sampled frames to data/processed/frames/<clip_id>/ (git-ignored)
```

```python
from soccer_tracking.data import load_manifest, SoccerTilesDataset, collate
train = SoccerTilesDataset(load_manifest(splits=["train"]), stride=25, augment=True, seed=0)
image, target = train[0]   # float32 CHW RGB in [0, 1]; target: boxes (xyxy, tile pixels), labels (all 1 = player), image_id
train.meta(0)              # clip id, frame, tile window, flipped, track ids: maps tile output back to full-frame coordinates
```

Settings (all are arguments, defaults shown): frames sampled every 25th (1 per second at 25 fps) and stored as JPEG quality 95; tiles 1024x1080 with 128 px overlap (5 tiles per 4096x1080 frame); a box is kept in a tile if at least 50% of it is visible and it is at least 4 px in both directions; horizontal flip with p = 0.5 only when `augment=True`, decided by `default_rng([seed, epoch, index])` (call `set_epoch`). Splits are whole matches taken from `data/manifest.csv`; the loader refuses a clip listed twice, an unknown split, a frame-size mismatch, or annotations that do not span the video's frames.

Known limits: a player cut by a tile edge is not labelled in the tile where less than half of them is visible (they appear whole in the neighbouring tile), so tile-level detections must be merged (for example with NMS) before frame-level evaluation or tracking. Tiles are cut at native scale, so Phase I results are for this camera setup and 4096x1080 clips.

## Pretrained baseline and evaluation

`src/soccer_tracking/evaluate.py` runs a detector over the tiles, merges the tile detections into full-frame boxes (`detect.py`: shift to frame coordinates, drop boxes cut by an interior tile border, NMS across tiles) and scores them against the labels (`metrics.py`: AP at IoU 0.5 and 0.75, plus precision, recall and F1 at a score threshold). The Phase I **baseline** is torchvision's COCO-pretrained RetinaNet (ResNet-50 FPN) used as-is, keeping only the `person` class as `player`; no training is involved, so it runs on a CPU.

```powershell
$env:PYTHONPATH = "src"
python -m soccer_tracking.evaluate --split val --weights coco --stride 250
# writes results/baseline_coco_val.json (settings, versions, git revision, metrics); the first run downloads the COCO weights
```

`--weights` also accepts a fine-tuned `.pt` checkpoint from `train.py`. `--stride` must be a multiple of the extraction stride (25); the default 250 evaluates 24 frames per clip so a CPU run stays short (use 125 or 25 for a larger, less noisy sample). The test split is refused unless `--allow-test` is passed: use it once, at the end, never to choose settings. Metrics are computed on full frames, with ground-truth boxes clipped to the frame. AP uses all-point interpolation, so it is close to but not identical to COCO's 101-point AP.

## Training on a hosted GPU (Colab)

`src/soccer_tracking/train.py` fine-tunes the COCO-pretrained RetinaNet on the train tiles (one class, `player`): SGD with momentum, linear warmup then cosine decay, gradient clipping, mixed precision on CUDA, validation (full-frame AP) after every epoch, early stopping on validation AP50, and a best/last checkpoint. It writes `runs/<run-name>/` with `config_<utc>.json` (all settings, versions, git revision, GPU), `iterations.csv` and `metrics.csv` (learning curves), `curves.png`, `summary.json` and the checkpoints (`*.pt` is git-ignored). A laptop CPU is far too slow (inference alone took about 4 s per tile); use a GPU session, for example Colab (Runtime, Change runtime type, GPU). Colab already has torch, torchvision, OpenCV, NumPy and matplotlib.

Download the clips inside Colab from Hugging Face with your own account; do not upload them from your laptop, and never paste your token into a cell or commit it. Store it as a Colab secret named `HF_TOKEN` (the token must belong to the account that accepted the dataset conditions). Storage in a Colab session is temporary.

```python
# cell 1: code and data
!git clone <your-repo-url> repo
%cd repo
!pip -q install huggingface_hub
import os, shutil
from google.colab import userdata
from huggingface_hub import hf_hub_download, login
login(token=userdata.get("HF_TOKEN"))
os.makedirs("data/raw", exist_ok=True)
for i in ["118575", "118576", "117093", "128057"]:          # the clips listed in data/manifest.csv
    for ext, folder in (("mp4", "mot/clips"), ("txt", "mot")):
        path = hf_hub_download("atomscott/soccertrack-v2", f"{folder}/{i}.{ext}", repo_type="dataset")
        shutil.copy(path, f"data/raw/{i}.{ext}")
```

```python
# cell 2: verify, extract, test, train, evaluate (the test split stays untouched)
%cd /content/repo
!PYTHONPATH=src python -m soccer_tracking.data audit --verify-hashes
!PYTHONPATH=src python -m soccer_tracking.data extract
!PYTHONPATH=src python -m unittest discover -s tests
!PYTHONPATH=src python -m soccer_tracking.train --run-name finetune --epochs 8
!PYTHONPATH=src python -m soccer_tracking.evaluate --split val --weights runs/finetune/best.pt --stride 125
```

If you see `No module named 'soccer_tracking'`, run `!pwd` and `!ls src/soccer_tracking`: you must be in the cloned repo (`/content/repo`) and the listing must include `train.py`. If the folder is missing the runtime was reset (the clone and downloads are wiped, so rerun cell 1); if `train.py` is missing the latest code was not pushed (`git push` from your laptop, then `!git pull`). Run a quick speed and memory check before the full run: `!PYTHONPATH=src python -m soccer_tracking.train --run-name smoke --epochs 1 --max-iters-per-epoch 20`.

`--verify-hashes` confirms the downloaded files are byte-identical to the ones recorded in `data/manifest.csv`. If the GPU runs out of memory add `--batch-size 2`; to continue an interrupted run use `--resume runs/finetune/last.pt`. Copy `runs/finetune/` (the small CSV, JSON and PNG files; the checkpoint if you want it) and `results/` back to your computer before the session ends, then commit the small files only.

## Results

Frozen protocol: [`docs/phase1_proposal.md`](docs/phase1_proposal.md) (amendment of 2026-10-21). The validation clip (117093) was used to choose the epoch and the tracker settings; the test match (128057) was scored once for each model and once for the tracking comparison, and the results are reported as obtained.

**Detector** (every 125th frame, 48 frames per clip; same tiling, merge and metric code; precision and recall at score 0.5 and IoU 0.5):

| Model | Clip | AP50 | AP75 | Precision | Recall |
|---|---|---|---|---|---|
| Pretrained COCO RetinaNet, no training | val 117093 | 0.657 | 0.462 | 0.740 | 0.705 |
| Fine-tuned (matches 118575, 118576; epoch 3 of 6) | val 117093 | 0.959 | 0.860 | 0.874 | 0.958 |
| Pretrained COCO RetinaNet, no training | **test 128057** | 0.788 | 0.560 | 0.849 | 0.791 |
| Fine-tuned | **test 128057** | 0.950 | 0.830 | 0.893 | 0.951 |

Criterion 1 (test: at least +0.10 AP50 and +0.15 AP75 over the pretrained model): +0.162 and +0.270, **met**. The gain is smaller on test than on validation because the pretrained model does better on the test match, whose players are larger (median box height 50.8 px against 44.0 px in training); the fine-tuned model scores about the same on both clips. Fine-tuning ran on a Colab T4 (batch 4, SGD peak learning rate 0.005, warmup then cosine decay, mixed precision, seed 0, about 42 minutes); curves in `runs/finetune/curves.png`.

**Tracking** (three segments of 250 consecutive frames per clip; both arms use the same detections at score 0.5; tracker settings chosen on validation: `iou_thr` 0.4, `max_age` 1, `min_hits` 1, motion prediction on):

| Arm | Clip | Interruptions per player track | F1 | MODA | ID switches | ids per player |
|---|---|---|---|---|---|---|
| Detector-only | val | 0.985 | 0.926 | 0.841 | n/a | n/a |
| IoU tracker | val | 0.652 | 0.914 | 0.812 | 98 | 2.44 |
| Detector-only | **test** | 1.545 | 0.912 | 0.811 | n/a | n/a |
| IoU tracker | **test** | 0.788 | 0.897 | 0.774 | 140 | 2.98 |

Criterion 2 (test: interruptions per track at least 25% lower, F1 lower by at most 0.02): **49.0% lower** (all 3 segments improved) and **F1 0.015 lower**, **met**. Read this together with the cost: the tracker bridges one-frame misses, which removes interruptions, but it also adds false-positive boxes (MODA falls by 0.037 on test) and its ids are not reliable (about 3 different ids per player in 10 seconds, 140 switches). The tracks are image-plane continuity aids, not player identities. Limits: one validation and one test match, three segments each, small counts (102 versus 52 interruptions on test), a deliberately simple tracker.

**Runtime:** pretrained model on the laptop CPU about 20 s per frame (5 tiles); fine-tuned model on a Colab T4 about 1.1 s per frame including video decoding.

## Tracking comparison (detector-only vs IoU tracker)

`src/soccer_tracking/tracker.py` is a deliberately simple IoU tracker (greedy association, optional linear motion prediction, short-gap coasting, no appearance model). `track_metrics.py` defines the measures (F1, MODA, fragmentation per track, ID switches, ids per track, MOTA) and `track_eval.py` runs the protocol frozen in the "Amendment" of [`docs/phase1_proposal.md`](docs/phase1_proposal.md): three 250-frame segments per clip (starting at frames 500, 2500, 4500), both arms on the same saved detections at score >= 0.5. `detect` needs a GPU (run it in Colab, after the training steps above, with the clips in `data/raw/`); `tune` and `evaluate` need only NumPy.

```python
# Colab, in /content/repo (git pull first); best.pt is the checkpoint on your Drive
!PYTHONPATH=src python -m soccer_tracking.track_eval detect   --split val --weights /content/drive/MyDrive/soccer_runs/finetune/best.pt
!PYTHONPATH=src python -m soccer_tracking.track_eval tune     --split val       # grid search on validation only -> results/tracker_tuning.json
!PYTHONPATH=src python -m soccer_tracking.track_eval evaluate --split val       # detector-only vs tracker on validation
```

Commit `results/tracker_tuning.json` (and push) before touching the test match: the tracker settings must be frozen first. Then, once and only once:

```python
!PYTHONPATH=src python -m soccer_tracking.track_eval detect   --split test --weights /content/drive/MyDrive/soccer_runs/finetune/best.pt --allow-test
!PYTHONPATH=src python -m soccer_tracking.track_eval evaluate --split test --allow-test
```

For criterion 1 (detector against the pretrained baseline on the test match) extract the test frames first, then score both models once each:

```python
!PYTHONPATH=src python -m soccer_tracking.data extract --splits test
!PYTHONPATH=src python -m soccer_tracking.evaluate --split test --weights coco --stride 125 --allow-test --out results/test_baseline_coco.json
!PYTHONPATH=src python -m soccer_tracking.evaluate --split test --weights /content/drive/MyDrive/soccer_runs/finetune/best.pt --stride 125 --allow-test --out results/test_finetuned.json
```

`tune` refuses the test split and `evaluate`/`detect` refuse it without `--allow-test`. Detections are saved to `data/processed/detections/finetune/` (git-ignored); the JSON files in `results/` are small and are committed. `tune` and `evaluate` also print and save a per-segment breakdown (gaps, F1 and ID switches for each 250-frame segment, plus how many segments the tracker improves), so you can see whether the three segments agree. `evaluate` prints PASS or FAIL for each frozen criterion. Fragmentation counts interruptions (a run of unmatched frames between two matched frames) per player track, so it is defined for detector-only output too; ids are image-plane track ids, not player identities.

## Error analysis

`src/soccer_tracking/errors.py` explains the detector's mistakes at the operating point (score >= 0.5, IoU >= 0.5) on the same sampled frames as `evaluate.py` (default: the validation clip, every 125th frame). Missed players are typed `merged_with_neighbour`, `low_score`, `poorly_localized` or `missed`; false positives are typed `duplicate`, `poorly_localized` or `no_gt_overlap` (overlaps no labelled player: an unlabelled person such as a referee or substitute, or background). It also reports recall by box height, by crowding (share of a player covered by a neighbour's box) and by distance to a tile border, and precision by score band.

```python
# Colab, in /content/repo, with the validation clip in data/raw/
!PYTHONPATH=src python -m soccer_tracking.data extract --splits val
!PYTHONPATH=src python -m soccer_tracking.errors --split val --weights /content/drive/MyDrive/soccer_runs/finetune/best.pt
!PYTHONPATH=src python -m soccer_tracking.errors --split val --weights coco --out-dir results/errors_val_baseline --images-dir data/processed/errors_val_baseline   # optional: the pretrained model for contrast
```

Counts and typed rows are written to `results/errors_<split>/` (`summary.json`, `errors.csv`; no pixels, safe to commit). Image panels of sampled errors (`fp_examples.jpg`, `fn_examples.jpg`: yellow = labelled players, cyan = confident detections, red = the error) go to `data/processed/errors_<split>/`, which is git-ignored because they contain frames of people: keep them local and do not upload them to AI services; describe what you see in words. **Reading the panels.** Each montage groups up to 8 randomly chosen examples (seed 0) per error type, so it is a sample for illustration, not a count: report counts and percentages from `summary.json`, and describe the panels as "k of 8 examples show ...". For each type, note what is inside the red box (a referee, substitute or staff member; an on-pitch player who has no yellow label box, which would be a labelling gap; a non-person such as a line, shadow or board; two overlapping players; a box that is too large, too small or shifted; a distant, blurred or half-hidden player) and whether the same pattern repeats.

The test split needs `--allow-test` and is for a descriptive look only: never change the model, the tiling or the tracker because of what test errors show.
