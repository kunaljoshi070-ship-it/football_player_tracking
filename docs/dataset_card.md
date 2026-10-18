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

## Phase I sample proposal

Select 3–5 short clips from the available annotated MOT subset after confirming it is downloadable and has both video and usable ground truth. Use whole matches for train/validation/test separation; do not randomly split nearby frames. A practical initial target is 1–2 training matches, 1 validation match, and 1 held-out evaluation match if the MOT annotation subset supports that split. Otherwise disclose the constraint and obtain instructor approval for a small, clearly labeled evaluation sample. Freeze clip/frame IDs before final metrics.

## Risks and limitations

- The full download is large; confirm selective-download behavior and disk space before retrieving data.
- The MOT bounding-box subset may not cover every match/split or contain ball boxes; verify the actual files. Use player-only Phase I scope if so.
- Fixed panoramic cameras differ from broadcast views. Claims should be limited to this camera setup.
- Action labels are events, not per-frame action classes; sequence labeling needs careful time-window construction and is later-phase work.
- The dataset does not supply all of the compendium's requested field segmentation masks or every desired model task. Do not silently imply that it does. If Phase II requires field masks, add a separate specifically licensed field-segmentation source or annotate a small approved sample yourself.

## Citation

Scott et al., *SoccerTrack v2: A Full-Pitch Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting*, arXiv:2508.01802 (2025). Use the latest BibTeX and attribution wording from the dataset card at submission time.

## Secondary candidate

[SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) is a strong alternative for broadcast-view player tracking. Its official raw tracking video is gated and needs an approved Hugging Face access request/NDA; annotations and video access must be verified separately. The FAQ states the videos are copyrighted and research-focused. Prefer it if your proposal specifically needs broadcast soccer, but do not assume it is immediately downloadable or redistributable.
