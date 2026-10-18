# CSAI3027P Deep Learning PBL — Project 36

**Project:** Sports Player Tracking and Tactical Event Analyzer
**Language:** Python (professor-approved)
**Current phase:** Phase I only

## Phase I scope

Use a small, fixed subset of soccer video to build one player-detection baseline and, if matching tracking labels are available, visualize short player trajectories. Compare a simple tracker with independent frame detections. Keep the result descriptive; do not claim player identity or full tactical understanding.

The selected dataset is **SoccerTrack v2, release v1.2**, using only a few clips from its annotated MOT subset. See [`docs/dataset_card.md`](docs/dataset_card.md) before accessing data. The full release is large; download only files needed for the frozen sample. Raw video, frames, model weights, and secrets stay out of Git.

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

Place the local files (git-ignored) at `data/raw/117092.mp4` and `data/raw/117092.txt`, then run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m soccer_tracking.visualize --video data/raw/117092.mp4 --annotations data/raw/117092.txt --frame 0
# writes data/processed/117092_frame0.jpg (git-ignored); use --output to choose another path
```

**Frame numbering.** `--frame` is a **0-based video frame index** (OpenCV convention). The annotation frame looked up is `--frame + --annotation-base`. `--annotation-base` is `0` by default because `117092.txt` was observed to start at frame 0 (frames 0–5999, 22 boxes per frame); pass `1` for a 1-based MOT file. Nothing is shifted unless you ask, and the mapping used is printed on every run. By default OpenCV *seeks* to the frame, which can be imprecise for some codecs; add `--sequential` for slow but frame-exact decoding when checking alignment.

Errors (missing file, malformed row with its line number, non-positive/non-finite box, duplicate `(frame, id)`, frame without annotations, frame beyond the video) exit with code 2 and a message. Boxes outside the image are clipped and reported.

Tests use only synthetic data generated at test time:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
