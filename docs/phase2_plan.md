# Phase II game plan: beat GTATrack on SoccerTrack

Status: **plan written, nothing in Phase II has been run yet.** The "Status board" (section 9) is updated at the end of every working session; read it first if you are picking this project up (human or AI).

## 1. Goal and what "beat" means

**Comparator:** GTATrack (Jian et al., ACM MMSports '25, DOI 10.1145/3728423.3759416), winner of the SoccerTrack Challenge 2025 on the same dataset. Its reported results (leaderboard): HOTA 0.60, DetA 0.76, AssA 0.47, LocA 0.84, ID switches 331.5, false negatives 5,454.5, false positives 982. Details and cautions: `docs/gtatrack_comparison.md`.

**Our Phase I starting point:** detection comparable on recall (about 0.96), precision weaker (about 12.6% false positives against under 1%), identities weaker (about 3 ids per player in 10 s), and no HOTA number yet.

**Pre-registered targets** (proposed; adjust before the first full evaluation, then freeze). All measured with the standard TrackEval tool on full 4-minute clips, mean per clip:

| Tier | Meaning | Target |
|---|---|---|
| T0 | Harness is valid and we know where we start | Phase I model and tracker scored with HOTA, DetA, AssA, LocA, FP, FN, IDSW on the validation clips; ground truth scored as predictions gives HOTA 1.0 |
| T1 | Competitive detection | DetA at least 0.74 and false positives no worse than the other top Challenge teams (about 3,000 per clip) |
| T2 | Top-5 level | HOTA at least 0.59 with AssA at least 0.45 |
| T3 | Beat | HOTA above 0.60 on the matched protocol; **or**, if HOTA is not exceeded, within 0.02 HOTA and better than GTATrack on at least two of: false positives, LocA, recall of players under 30 px, run time per frame |

"Beat" claims must use the same clips and the same metric. GTATrack's leaderboard numbers come from a hidden test set whose clips are not stated, so the strongest claim is only possible if we can run their public code on our clips (gate G2).

## 2. Rubric map (Phase II, 50 marks)

| Criterion | Marks | Workstream and evidence |
|---|---|---|
| Innovation implementation and measured impact | 12 | WS-B, WS-C, WS-D: the combined method and its measured effect against GTATrack and Phase I |
| Ablation or comparative experiment | 5 | WS-D: component-by-component ablation table; whole-frame versus tiles |
| Error analysis and interpretation | 4 | WS-F: typed errors v2, by size, crowding, officials; robustness; bias |
| Product integration and final functionality | 10 | WS-E: usable prototype with input validation, prediction, confidence and interpretation |
| Improvement after Phase I | 2 | Record panel feedback in `docs/phase1_feedback.md` and map each point to an action |
| Learning evidence across experiments | 5 | Experiment log, learning curves for every training run, reasoned model selection |
| Live unseen test, change or debugging | 4 | Reserve clips never used in development; cached-detection demo so thresholds change instantly |
| Responsible AI and limitations | 2 | Model card (WS-F) |
| Final viva and communication | 6 | Architecture trace, ablation defence, demo; viva sheet |

Final submission checklist items: source and requirements, dataset docs and manifest, baseline and final checkpoints, experiment log, curves, comparison and ablation, error analysis, product screenshots and demo, source log, model card, separate timestamped Phase_I and Phase_II folders, signed originality declaration.

## 3. Protocol

**Splits for Phase II** (GTATrack-aligned so comparisons are fair; this differs from the Phase I split, which stays frozen and untouched):

| Role | Clips | Note |
|---|---|---|
| Train | 118575, 118576, 118577, 117092 | the four clips GTATrack trained on; 117092 is 3840x1906 |
| Validation | 117093, 128058 | GTATrack's held-out pair (the paper calls it both "test" and "validation"); all development decisions use only these |
| Final test | 128057, 132831 | the dataset's official test pair; scored once per frozen configuration; 128057 was already scored once in Phase I (disclose); 132831 is 3840x1504 (resolution-shift check) |
| Reserve | 118578, 132877 | never used in development: used for the live unseen test in the viva |

**Rules:** keep Phase I's discipline: choose settings on validation only; write the final configuration into a dated amendment before the first test run; test is run once; every run gets a JSON record and an experiment-log row; Phase I result files are never overwritten (new runs get new names).

**Two-tier evaluation (compute):** at the current speed (about 1.1 s per frame on a T4) a full clip takes about 1.8 hours. Run ablations on fixed windows (for example three 1,000-frame windows per validation clip) and score only finalists on full clips with TrackEval. Inference speed work (WS-A4) comes first.

**Baselines in every table:** B0 = Phase I detector and tracker; B1 = GTATrack as reported; B2 = GTATrack's public code run on our clips (if possible).

## 4. Workstreams

Effort: S = one session, M = two to three, L = four or more. "Evidence" is what goes into the experiment log.

### WS-A: Evaluation harness and full-clip inference (do first)
- A1. Extend `track_eval.py detect` to whole clips with chunked, resumable saving (Colab disconnects). Files: `src/soccer_tracking/track_eval.py`, tests.
- A2. New `mot_export.py` (write tracker output and ground truth in MOTChallenge text format) and `hota_eval.py` (run TrackEval, parse HOTA, DetA, AssA, LocA, FP, FN, IDSW, IDF1 into `results/hota_<run>.json`). TrackEval is MIT licensed (check); add to the source log.
- A3. Sanity tests: ground truth as prediction gives HOTA 1.0; shifted boxes lower LocA; synthetic id swap lowers AssA.
- A4. Inference speed: fp16 inference, larger tile batches, channels-last, crop each clip to the pitch band (players occupy a vertical band), skip empty tiles; record latency and GPU memory for the product and model card.
- Effort: M. Exit gate G1.

### WS-B: Detector (precision, crowds, small players)
- B1. **More data:** train on the four clips with every 5th or 10th frame (about 12,000 to 24,000 tiles), longer schedule, resume from Drive checkpoints; select by validation AP75 or the mean of AP50 and AP75 (AP50 saturated in Phase I). Estimated 30 to 60 minutes per epoch on a T4 (estimate).
- B2. **Officials and unlabelled people (our largest false-positive source):** (a) rely on more data; (b) post-filter by kit-colour clustering (two teams plus others); (c) optional pseudo-label an "other person" class from confident detections that overlap no label, clustered by colour. Evidence: false positives, precision by score band.
- B3. **Crowd-aware merging:** disable the in-model NMS (raise its threshold), then compare NMS, Soft-NMS and adjusted IoU in the merge step; inference-only, so cheap. Evidence: recall for crowded players (Phase I: 0.750).
- B4. **Anchors and loss:** smaller anchors (16 to 24 px), aspect ratios up to about 3, IoU-based box loss; short training runs.
- B5. **Flip test-time augmentation with box fusion.**
- B6. **Headline experiment, whole frame versus tiles:** same architecture and data, trained once on whole frames resized to 1280 px on the long side (GTATrack's input scale) and once on native-scale tiles. Evidence: recall by player height, LocA, AP75. Needs a whole-frame dataset class.
- B7. **Second architecture (rubric and compendium):** easiest is a torchvision detector (FCOS or Faster R-CNN v2) in the same pipeline; a YOLO model gives the most direct comparison with GTATrack but its code is AGPL-3.0 licensed (check) and needs tiles exported to disk. Decision D3.
- B8. Optional: pseudo-labelling as in GTATrack, with the kit-colour filter so officials are not added as players.
- Effort: L. Exit gate G3.

### WS-C: Tracker and association
- C1. **Tracker v2:** ByteTrack-style two-threshold association (high-score boxes start and continue tracks, low-score boxes may only continue them), confirmation before output, Expansion IoU in matching, no output for unconfirmed coasting tracks.
- C2. **Appearance:** (i) cheap colour-histogram features of the torso region, which also separate teams and officials; (ii) OSNet embeddings (torchreid, MIT, check) with a same-team gate. Evidence: AssA and ID switches.
- C3. **Global tracklet linking:** our own Connector (hierarchical clustering on average appearance distance with time and space constraints) and an optional Splitter, evaluated with HOTA.
- C4. **Roster prior:** every frame has exactly 22 labelled players, so allow at most 22 confirmed tracks (plus slack) and aim the global stage at 22 trajectories; drop tracks assigned to the "other" colour cluster (officials). Not described in GTATrack; risk: fails on data with substitutions.
- C5. Tune on validation with a fixed grid and a pre-stated rule (as in Phase I).
- Effort: L. Exit gate G4.

### WS-D: Innovation statement and ablation
Proposed innovation: **native-scale tiled detection with crowd-aware merging, plus a roster-constrained, appearance-aware global association with official filtering.** Ablation rows (each adds one component): B0 Phase I; + more data; + crowd-aware merging; + official filter; + tracker v2; + appearance; + global linking; + roster prior = final. Columns: HOTA, DetA, AssA, LocA, FP, FN, IDSW, recall under 30 px, seconds per frame. Add a paired bootstrap over windows for uncertainty. Effort: M after WS-B and WS-C.

### WS-E: Product (10 marks)
- User and context: an analyst reviewing a fixed-camera match clip offline.
- Interface: command line first, then a Gradio or Streamlit app run in Colab (a laptop CPU needs about 20 s per frame).
- Input validation: container, resolution, fps, frame count, duration limit, segment bounds, clear errors.
- Output: annotated video, trajectory CSV (frame, track id, box, score, track confidence), heat map, summary JSON.
- Confidence and interpretation: per-detection score, track confidence (length, mean score), flags for crowded, low-score and possible-official tracks; fixed on-screen note that ids are image-plane tracks, not players.
- Demo mode on cached detections so threshold and tracker settings can be changed live in the viva.
- Tests for validation; weights kept out of Git with download instructions.
- Effort: M. Exit gate G6.

### WS-F: Responsible AI, error analysis v2, documentation
- Model card: intended use (offline analysis of fixed-camera footage), non-intended use (identifying individuals, performance or disciplinary judgements, surveillance, live decisions, other camera types), data and consent notes, limitations, risks.
- Error analysis v2 on the final model: by size, crowding, officials, and by clip; robustness checks (resolution-shifted clip 132831, blur and brightness changes, frame decimation, mirror).
- `docs/phase1_feedback.md` from the Phase I panel, with actions taken.
- Phase_I and Phase_II snapshots (git tag plus folder), signed originality declaration (yours), viva sheet.
- Effort: M.

### WS-G (optional hedge): fundamentals pack from the project compendium
The compendium lists a NumPy MLP with backpropagation and SGD, momentum and Nesterov, gradient checking, SVD diagnostics, Hebbian and competitive learning demos, and a VAE or LSTM extension; the hackathon rubric does not score them. Recommended: only a small time-boxed pack (MLP with optimiser comparison and gradient check; one VAE latent plot) if the compendium is binding for your submission. Decision D2.

## 5. Order of work and gates

| Gate | When | Exit criterion |
|---|---|---|
| G0 | session 1 | Decisions D1-D4 made; targets frozen in a dated amendment; Phase I committed (tag, snapshot optional); this plan committed |
| G1 | sessions 2-4 | WS-A done: TrackEval pipeline on a full validation clip; sanity tests pass; B0 numbers recorded (T0) |
| G2 | session 4 | Try GTATrack's code and weights on a validation clip; outcome recorded (B2 available or not) |
| G3 | sessions 5-9 | Detector improvements measured on validation; best detector chosen by the pre-stated rule |
| G4 | sessions 7-11 | Tracker improvements measured; best tracker chosen; T1/T2 status known |
| G5 | sessions 11-12 | Full ablation table; error analysis v2; final configuration frozen; test run once (T3 decision) |
| G6 | sessions 12-15 | Product, model card, docs, snapshots, viva preparation |

Roughly 15 sessions of 2 to 3 hours plus Colab time; the real number depends on your deadline (decision D1).

## 6. Risks and mitigations

- **GPU time.** Free Colab sessions are limited; full-clip inference and longer training are heavy. Mitigation: two-tier evaluation, speed work first, resumable runs, consider Colab Pro or Kaggle (D4).
- **Identity is hard** (near-identical kits). Mitigation: colour and OSNet features plus global linking; HOTA can still be reached mainly through DetA.
- **Hidden-test mismatch.** Mitigation: report validation-pair and test-pair numbers separately, never claim more than the protocol supports.
- **Label gaps** (unlabelled people) cap measurable precision. Mitigation: official filter; state it in the report.
- **Contamination.** Mitigation: one-time test rule, reserve clips untouched, amendments before runs.
- **Hand-offs between AI sessions.** Mitigation: the status board and handoff block below; every session adds a `CONTRIBUTIONS.md` entry.

## 7. Open decisions (defaults in brackets)

- D1. Phase II deadline and weekly hours. [unknown: plan assumes about 15 sessions]
- D2. Include the compendium fundamentals pack (WS-G)? [small pack only if required]
- D3. Second architecture: torchvision FCOS or Faster R-CNN v2 (easy) or YOLO (direct comparison, AGPL licence). [torchvision first]
- D4. Compute: free Colab, Colab Pro, or Kaggle. [free Colab, revisit at G1]

## 8. Handoff protocol (for every session, human or AI)

End each session with this block, in chat and (for status changes) in the board below, and add a `CONTRIBUTIONS.md` entry for any AI use:

1. **Work done** (what ran, what did not, with numbers and file names).
2. **Files created or edited** (paths).
3. **Next steps** (ordered, with exact commands).
4. **Open decisions and blockers.**
5. **Notes for the next model** (conventions, pitfalls, what not to touch).

Conventions: Phase I results and the Phase I split are frozen; new runs get new names; the test split is only run through the guarded commands with `--allow-test`; never commit video, frames, crops, weights or secrets; every result file records versions, git revision and command; never invent results.

## 9. Status board

| Item | Status |
|---|---|
| Phase II plan written | done (this file) |
| D1-D4 decisions | open |
| Phase I snapshot (tag and folder) | deferred |
| Panel feedback recorded | not started (needs the student) |
| WS-A harness (TrackEval, full-clip inference) | not started |
| G2 GTATrack code check | not started |
| WS-B detector | not started |
| WS-C tracker | not started |
| WS-D ablation | not started |
| WS-E product | not started |
| WS-F model card, error analysis v2, snapshots | not started |
| WS-G fundamentals pack | undecided |
