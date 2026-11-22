# CSAI3027P Deep Learning PBL — Project 36

**Project:** Sports Player Tracking and Tactical Event Analyzer
**Language:** Python (professor-approved)
**Current phase:** Phase II complete (Phase I is finished and frozen; see `docs/phase2_plan.md` and `docs/phase2_final_config.md`)

## Phase I scope

Use a small, fixed subset of soccer video to build one player-detection baseline and, if matching tracking labels are available, visualize short player trajectories. Compare a simple tracker with independent frame detections. Keep the result descriptive; do not claim player identity or full tactical understanding.

The selected dataset is **SoccerTrack v2, release v1.2**, using only its annotated MOT subset (ten 4-minute clips, about 3.5 GB; the full release is 145 GB and is not needed). Phase I uses four whole matches: two for training, one for validation and one held-out test, frozen in [`data/manifest.csv`](data/manifest.csv) and explained in [`docs/dataset_card.md`](docs/dataset_card.md), which you should read before accessing data. Raw video, frames, model weights, and secrets stay out of Git.

## Project files

- [`AGENTS.md`](AGENTS.md): shared rules for any person or AI tool working on this repository.
- [`docs/phase1_proposal.md`](docs/phase1_proposal.md): one-page proposal and experiment hypothesis.
- [`docs/dataset_card.md`](docs/dataset_card.md): source, access, license, and sample plan.
- [`docs/architecture.md`](docs/architecture.md): the detector's layers, tensor shapes, frozen and trainable parts, loss, training choices and what is original.
- [`docs/experiment_log.md`](docs/experiment_log.md): every reported run (Phase I and Phase II) with its setup, result and record file.
- [`docs/phase2_plan.md`](docs/phase2_plan.md): Phase II game plan, targets, protocol, workstreams, gates, risks, handoff protocol and status board.
- [`docs/phase2_final_config.md`](docs/phase2_final_config.md): dated final configuration frozen before the single held-out Phase II `--allow-test` run.
- [`docs/model_card.md`](docs/model_card.md): Responsible AI model card (intended/non-intended use, metrics, limitations, risks).
- [`docs/error_analysis.md`](docs/error_analysis.md) and [`docs/error_analysis_v2.md`](docs/error_analysis_v2.md): Phase I typed detector errors and Phase II full-clip / resolution-shift error analysis.
- [`docs/gtatrack_comparison.md`](docs/gtatrack_comparison.md): how the published GTATrack paper works and how our pipeline compares.
- [`data/manifest.csv`](data/manifest.csv): dataset version, splits, clip IDs, and verified SHA-256 hashes.
- [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md): individual, source-log, and prompt-wise AI contribution record.

## Phase II Final Results (Full 4-Minute Clips, TrackEval)

Protocol frozen prior to test in [`docs/phase2_final_config.md`](docs/phase2_final_config.md). Validation clips (`117093`, `128058`) were used for epoch selection and tracker tuning; held-out test clips (`128057`, `132831`) were scored **once** via `--allow-test`. Reserve clips (`118578`, `132877`) remain untouched for the live unseen viva test.

### Validation Ablation (Mean over 117093 & 128058)

| Stage | HOTA | DetA | AssA | LocA | FP | FN | IDSW | IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase I (`p1model + IoU`) | 0.3188 | 0.7063 | 0.1446 | 0.8790 | 19,768.0 | 6,392.0 | 1,483.0 | — |
| Phase II (`P2-more + Tracker v2`) | 0.3681 | 0.7613 | 0.1792 | 0.8888 | 9,716.5 | 7,120.5 | 983.0 | 0.2900 |
| `+ v3 EIoU + Online Lab Gate` (`v3_eiou_only`) | 0.3583 | 0.7798 | 0.1654 | 0.8894 | 5,257.0 | 7,488.5 | 945.5 | 0.2940 |
| `+ Pretrained OSNet + GTA-Link` | 0.4049 | 0.7779 | 0.2113 | 0.8893 | 5,204.5 | 7,522.0 | 874.0 | 0.3878 |
| `+ Soccer-Finetuned OSNet + GTA-Link` (`dh=0.75`) | 0.4943 | 0.7791 | 0.3143 | 0.8896 | 5,219.5 | 7,519.0 | 812.0 | 0.4981 |
| **Final Frozen Config** (`dh=0.78, dl=0.45, cos=0.20, gap=600`) | **0.5011** | **0.7812** | **0.3222** | **0.8898** | **4,615.5** | 7,711.0 | **787.0** | **0.5198** |
| *GTATrack ByteTrack baseline (paper Table 3)* | *0.42* | *0.74* | *0.26* | *0.84* | *961* | *10,115* | *630* | — |
| *GTATrack full (paper Table 3, leaderboard)* | *0.60* | *0.76* | *0.47* | *0.84* | *982* | *5,454.5* | *331.5* | — |

### Held-Out Test Results (`results/p2_final_test.json`, `--allow-test`, Single Run)

| Clip | Resolution | HOTA | DetA | AssA | LocA | FP | FN | IDSW | IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`128057`** *(matched res; disclosed Phase I clip)* | `4096×1080` | **0.4778** | **0.7743** | **0.2952** | **0.8842** | **3,336** | 8,912 | **689** | **0.4801** |
| **`132831`** *(resolution-shift check)* | `3840×1504` | **0.3876** | 0.5375 | **0.2802** | **0.8691** | 67,955 | **4,654** | **685** | 0.3618 |
| **Test Mean** | — | **0.4327** | 0.6559 | 0.2877 | 0.8767 | 35,645.5 | 6,783.0 | 687.0 | 0.4210 |

- On matched-resolution held-out test (`128057`), the frozen pipeline generalizes closely (`0.4778` vs. val `0.5011`), beats GTATrack on `DetA` (`0.774` vs. `0.76`) and `LocA` (`0.884` vs. `0.84`), beats GTATrack's ByteTrack baseline (`0.42`), and runs at `0.36–0.43 s/frame` on a T4 (FP16, `batch_size=5`).
- On the resolution-shift clip (`132831`, `3840×1504`), ReID + tracking generalize steadily (`AssA 0.280`, `IDSW 685`, `LocA 0.869`), while the 40% taller frame exposes benches and stands outside the pitch, inflating FP to `67,955` (analyzed in [`docs/error_analysis_v2.md`](docs/error_analysis_v2.md)).

## Offline Tactical Analyzer MVP (CLI & Gradio App)

`src/soccer_tracking/product.py` provides input validation, track-confidence tiers, crowding/low-score flags, and exports four artifacts (`annotated.mp4`, `trajectories.csv`, `heatmap.png`, `summary.json`). `src/soccer_tracking/app.py` wraps it in a thin Gradio UI with **Cached Demo Mode** (instant on CPU) and **Live Video Segment Mode** (GPU recommended).

### Running Locally on a Laptop (Windows PowerShell, CPU Cached Demo Mode)

```powershell
.\.venv\Scripts\Activate.ps1
pip install gradio
$env:PYTHONPATH = "src"
python -m soccer_tracking.app
# Open http://127.0.0.1:7860 in your browser (use Cached Demo Mode on CPU)
```

Or via CLI directly:

```powershell
$env:PYTHONPATH = "src"
python -m soccer_tracking.product cached `
  --video data/raw/117093.mp4 `
  --det-dir data/processed/detections_p2/p2more_full `
  --clip-id 117093 --start 4000 --end 4200 `
  --out-dir data/processed/demo_out
```

### Running Unit Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Environment & Phase I Reference

Python 3.13 (`requirements.txt` pins core packages; `gradio` is an optional UI dependency, and `TrackEval` / `deep-person-reid` are MIT-licensed external clones added to `PYTHONPATH` for TrackEval scoring and OSNet ReID). Phase I baseline, training, error analysis, and architecture details remain preserved in `docs/phase1_proposal.md`, `docs/phase1_summary.md`, `docs/architecture.md`, `docs/error_analysis.md`, and `results/`.
