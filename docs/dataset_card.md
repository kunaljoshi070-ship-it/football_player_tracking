# Dataset card — SoccerTrack v2

**Recommendation:** Primary candidate for this soccer-only project, subject to instructor acceptance, per-file access review, and a feasible partial download.
**Release checked:** 2026-10-18; Hugging Face card reports v1.2.
**Total listed size:** 145 GB; do not download the full release for Phase I.

## Why it fits

The dataset describes ten university-level soccer matches filmed with fixed panoramic cameras, with per-frame game-state annotations, persistent track IDs, player roles/teams, ball-action spotting labels, and a curated MOT bounding-box subset. This supports a feasible Phase I player-detection/tracking baseline and may support later sequence/action analysis. Its fixed full-pitch view also enables descriptive movement maps more directly than moving broadcast cameras, once coordinate conventions are validated.

## Components to request/use

- **Phase I minimum:** a small, selectively downloaded set of video clips and corresponding MOT bounding-box/identity annotations for player detection/tracking.
- **Optional later Phase II:** game-state (GSR) annotations for pitch-coordinate trajectories and ball-action (BAS) labels for a sequence/event model.
- Do not treat the interpolated ball track as dense ground truth: the dataset card says it is reconstructed from event anchors and straight constant-speed interpolation. It is unsuitable as an unbiased framewise ball-detection evaluation target without a separate audit.
- The public data card describes a match-level train/validation/test split. Match IDs and assignments are in the upstream card; mirror them in the project manifest and keep matches separate to prevent frame leakage.

## Access, license and privacy checked 2026-10-18

- Official dataset card: [Hugging Face SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2). It says access requires agreeing to share contact information, labels the data CC BY 4.0, and reports 145 GB total.
- Project maintainers' [GitHub repository](https://github.com/AtomScott/SoccerTrack-v2) separates MIT-licensed code from CC BY 4.0 dataset video/annotations. The data license requires attribution; retain citation and attribution with all derived reports and outputs.
- Confirm the exact conditions displayed on Hugging Face and in `LICENSE-DATA` at download time. Do not infer consent to public redistribution of personal information or unrelated data from the dataset license. The upstream page says player names are omitted and describes participants as amateur players with informed consent.
- A private course repository is sensible but does not replace the dataset terms. Keep videos/frames local and ignored. Never upload video, person crops, or credentials to Claude/Gemini/ChatGPT unless the instructor and applicable dataset terms explicitly permit it.

## Download plan

1. Open the Hugging Face dataset card and review/accept the current access condition.
2. Inspect the release's file tree and `LICENSE-DATA`; record the version tag (currently v1.2) and upstream split.
3. Download only selected match files and corresponding annotation files using the official project scripts or selective Hugging Face download. Test selection on one small clip before obtaining more. The upstream GitHub repo documents a `scripts/download.sh` helper and a match filter; confirm its current interface before running it.
4. Extract/convert only to an ignored local `data/raw/` or `data/processed/` path. Do not commit the dataset or derived frames.
5. Verify frame numbering, fps, annotation mapping, bbox convention, match-level split and checksum. Record only non-sensitive identifiers, URLs, access date, version, split, checksums and local ignored paths in `data/manifest.csv`.

## MOT subset used in Phase I (measured 2026-10)

The MOT subset lives under `mot/` on Hugging Face: ten `<id>.txt` label files (about 60 MB in total) and ten paired 4-minute clips in `mot/clips/` (about 3.5 GB in total). Because the subset is small, all ten were downloaded to ignored `data/raw/`; only the clips listed below are used in Phase I and the rest stay untouched. Hashes, measured size, fps and frame counts for every clip are in `data/manifest.csv`.

Measured on the downloaded files: every clip is 25 fps, its video frame count equals its annotation frame count, annotations are 0-based (frames 0..N-1), and every frame has exactly 22 boxes with 22 IDs per match (outfield players and goalkeepers only: no ball, no referees). Clip sizes vary: seven are 4096x1080, 132877 is 4096x1084, 132831 is 3840x1504 and 117092 is 3840x1906. The `mot/README.md` states 4096x1080 for the clips, which holds for only some of them, so use the measured values. Box alignment was checked visually only on 117092 (frames 0 and 5999); check each chosen clip before training.

**Split.** The upstream card's benchmark split is: test 128057 + 132831, validation 117093 + 132877, train 117092, 118575, 118576, 118577, 118578 and 128058. (The Drive folder grouping described in `mot/README.md` differs; this project follows the Hugging Face card.) Phase I keeps the official roles, uses whole matches, and uses only 4096x1080 clips so that resolution is not confounded with the split:

| Role | Match | Notes |
|---|---|---|
| train | 118575, 118576 | official train |
| validation | 117093 | official validation |
| held-out test | 128057 | official test; use once at the end, never for tuning |
| unused | 117092, 118577, 118578, 128058, 132831, 132877 | reserve; 117092 also used for tool verification |

Frozen in `data/manifest.csv` before any evaluation. Only frame subsampling inside a match may be used for training speed; never split frames of one match across roles.

## Risks and limitations

- The full download is large; confirm selective-download behavior and disk space before retrieving data.
- The MOT subset covers all ten matches but contains players only (22 per frame); there are no ball or referee boxes, so Phase I is player-only.
- Clip resolution varies (3840x1906 to 4096x1080); results from the chosen 4096x1080 clips do not automatically transfer to the others.
- Fixed panoramic cameras differ from broadcast views. Claims should be limited to this camera setup.
- Action labels are events, not per-frame action classes; sequence labeling needs careful time-window construction and is later-phase work.
- The dataset does not supply all of the compendium's requested field segmentation masks or every desired model task. Do not silently imply that it does. If Phase II requires field masks, add a separate specifically licensed field-segmentation source or annotate a small approved sample yourself.

## Citation

Scott et al., *SoccerTrack v2: A Full-Pitch Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting*, arXiv:2508.01802 (2025). Use the latest BibTeX and attribution wording from the dataset card at submission time.

## Secondary candidate

[SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) is a strong alternative for broadcast-view player tracking. Its official raw tracking video is gated and needs an approved Hugging Face access request/NDA; annotations and video access must be verified separately. The FAQ states the videos are copyrighted and research-focused. Prefer it if your proposal specifically needs broadcast soccer, but do not assume it is immediately downloadable or redistributable.
