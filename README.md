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
%env PYTHONPATH=src
!python -m soccer_tracking.data audit --verify-hashes
!python -m soccer_tracking.data extract
!python -m unittest discover -s tests
!python -m soccer_tracking.train --run-name finetune --epochs 8
!python -m soccer_tracking.evaluate --split val --weights runs/finetune/best.pt --stride 125
```

`--verify-hashes` confirms the downloaded files are byte-identical to the ones recorded in `data/manifest.csv`. If the GPU runs out of memory add `--batch-size 2`; to continue an interrupted run use `--resume runs/finetune/last.pt`. Copy `runs/finetune/` (the small CSV, JSON and PNG files; the checkpoint if you want it) and `results/` back to your computer before the session ends, then commit the small files only.
