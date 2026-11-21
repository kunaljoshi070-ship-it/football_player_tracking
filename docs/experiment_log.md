# Experiment log (Phase I)

Every run whose result is reported. Dates are UTC from the result files where available; for blank fields open the named file (`environment.git_revision` and `git_dirty` are recorded inside each JSON). The protocol was frozen in `docs/phase1_proposal.md` (amendment of 2026-10-21) before the tracking and test results existed. Test-split runs are marked **TEST**; each was run once.

| ID | Date | What | Setup | Key result | Record |
|---|---|---|---|---|---|
| E1 | 2026-10-20 07:44 | Pretrained COCO RetinaNet, validation (first run, 24 frames) | laptop CPU, stride 250, revision `94c7eac` (uncommitted changes) | AP50 0.664, AP75 0.464, P 0.744, R 0.699; 20.4 s per frame | numbers in `CONTRIBUTIONS.md`; the file was overwritten by E2 |
| E2 | 2026-10-20 10:22 | Same, rerun from a clean commit (the reported baseline) | laptop CPU, stride 125 (48 frames), revision `f82ea50`, clean | AP50 0.657, AP75 0.462, P 0.740, R 0.705; 20.7 s per frame | `results/baseline_coco_val.json` |
| E3 | 2026-10-21 (before E4) | Smoke run of the training script | Colab T4, 20 iterations | pipeline works; validation AP50 0.399 with recall 0.000 is not a result | not kept |
| E4 | 2026-10-21 05:32 | Fine-tuning, matches 118575 and 118576 | Colab T4, torch 2.11.0, SGD, peak lr 0.005, batch 4, mixed precision, seed 0, revision `25098dc` (uncommitted files in the repo: smoke-run output) | 6 epochs (early stopping), best epoch 3: val AP50 0.959, AP75 0.860, P 0.874, R 0.958; about 42 min | `runs/finetune/` (`config_*.json`, `metrics.csv`, `iterations.csv`, `curves.png`, `summary.json`); `best.pt` on Drive only |
| E5 | 2026-10-21 | Tracker tuning and comparison on validation (3 segments of 250 frames) | 64 settings, rule fixed in advance; chosen `iou_thr` 0.4, `max_age` 1, `min_hits` 1, motion prediction on | detector-only 0.985 interruptions per track, F1 0.926; tracker 0.652, F1 0.914; 98 ID switches | `results/tracker_tuning.json`, `results/tracking_val.json` |
| E6 | 2026-10-21 | Error analysis on validation, fine-tuned and pretrained | stride 125, score 0.5, IoU 0.5 | fine-tuned: 44 missed (37 crowded), 146 false positives (111 overlap no labelled player); pretrained: 312 missed (258 low score), 261 false positives | `results/errors_val/`, `results/errors_val_baseline/` (image panels kept local); write-up `docs/error_analysis.md` |
| E7 | 2026-10-21 | Model description (layers, parameters, tensor shapes) | laptop, COCO weights, 2 classes | 32,168,694 parameters, 31,946,294 trainable | `results/model_description.json` |
| E8 | 2026-10-21 | **TEST** criterion 1: pretrained against fine-tuned on match 128057 | Colab T4, stride 125 (48 frames) | pretrained AP50 0.788 / AP75 0.560; fine-tuned 0.950 / 0.830 (+0.162, +0.270: met) | `results/test_baseline_coco.json`, `results/test_finetuned.json` |
| E9 | 2026-10-21 | **TEST** criterion 2: detector-only against IoU tracker on match 128057 | frozen settings from E5, 3 segments of 250 frames | interruptions per track 1.545 to 0.788 (-49.0%, met); F1 0.912 to 0.897 (-0.015, met); 140 ID switches, 2.98 ids per player | `results/tracking_test.json` |

Notes: E1 and E2 were run on a laptop CPU, E4 and E8-E9 on a Colab T4, so the baseline rows on validation and on test were not produced on the same hardware (the metrics do not depend on it beyond tiny numerical differences, but the timings do). All fine-tuning, tuning and selection decisions used the validation clip only; the test match was scored once per model and once for tracking. GPU training is not bit-exact reproducible, and only one training run (seed 0) exists. Frames were extracted separately on the laptop (OpenCV 4.13.0, used by E1 and E2) and on Colab (OpenCV 4.14.0, used by E4-E9); JPEG output may differ very slightly between builds, so the validation baseline and the fine-tuned model were not scored on byte-identical frames (the effect was not measured and is expected to be negligible).

## Phase II training resume note (2026-11-20)

The P2-more six-epoch run was resumed from the epoch-4 `last.pt` checkpoint after an interrupted partial epoch-5 attempt. The iteration log was deduplicated by iteration, keeping the latest row; 47 duplicate iteration IDs were removed. Two timestamped config files record the original and resumed invocations. Best epoch 4 was selected by mean AP50/AP75 (selected value 0.9248; AP50 0.9656; AP75 0.8841).

## Phase II evaluation log (2026-11-21)

All metrics below are from the `trackeval` sections of the named JSON files (`trackeval_error: null`). Full-clip validation scores are development results: P2-more's epoch had been selected using these validation clips. Each clip contributes equally to the reported per-clip mean.

| Run | Setup | Mean HOTA / DetA / AssA / LocA | Mean FP / FN / IDSW | Records |
|---|---|---|---|---|
| T0 `p1model_full`, IoU | Phase I weights and frozen Phase I IoU tracker; validation frames 0–5975 (117093) and 0–5980 (128058) | 0.3188 / 0.7063 / 0.1446 / 0.8790 | 19,768 / 6,392 / 1,483 | `results/p2_full_t0_iou.json` |
| T0 `p1model_full`, IoU-v2 | Same weights; tracker-v2 settings `det_high=0.6`, `det_low=0.3`, `match_iou=0.3`, `max_age=5`, `min_hits=3` | 0.3439 / 0.7175 / 0.1659 / 0.8796 | 15,972 / 7,544 / 1,126.5 | `results/p2_full_t0_iou_v2.json` |
| P2-more `p2more_full`, IoU | P2-more weights and frozen Phase I IoU tracker; same full validation ranges | 0.3436 / 0.7557 / 0.1575 / 0.8884 | 12,097 / 6,143 / 1,356.5 | `results/p2_full_p2more_iou.json` |
| P2-more `p2more_full`, IoU-v2 | P2-more weights and the same tracker-v2 settings as above | 0.3681 / 0.7613 / 0.1792 / 0.8888 | 9,716.5 / 7,120.5 / 983 | `results/p2_full_p2more_iou_v2.json` |

With IoU-v2 held fixed, P2-more versus T0 improved mean HOTA by 0.0242 and reduced mean FP from 15,972 to 9,716.5; mean FN also fell from 7,544 to 7,120.5. This is a full-validation comparison, not a held-out test result or a matched comparison with GTATrack. Detector-only full-clip JSON records were not present in the reviewed local changes.

**Threshold isolation, clip 117093 frames 1000–1999:** the two runs used P2-more weights and changed one threshold from the 0.5/0.5 setting at a time. NMS 0.7 / merge IoU 0.5 gave IoU-tracker HOTA 0.3988, FP 2,831, FN 3,935 and IDSW 507, versus the 0.5/0.5 P2-more window result 0.3959, FP 2,976, FN 3,757 and IDSW 522. Merge IoU 0.7 / NMS 0.5 gave HOTA 0.3953, FP 3,060, FN 3,729 and IDSW 532. Thus the isolated NMS change made a small HOTA gain with more misses; the isolated merge-IoU change did not improve HOTA. Detector-arm and IoU-arm records: `results/p2more_nms07_detector.json`, `results/p2more_nms07_iou.json`, `results/p2more_merge07_detector.json`, `results/p2more_merge07_iou.json`. These are one-window development results.

**C4 roster-cap tuning:** the frozen protocol is `docs/phase2_amendment_c4_roster.md`. On the two tuning windows (frames 1000–1999), TrackEval selected pre-tracking detection cap `mode=det, K=22`: mean HOTA 0.53924, DetA 0.70568, AssA 0.41855, FP 1,243, FN 2,652, IDSW 254.5. The uncapped `none` candidate scored HOTA 0.53162, DetA 0.69656, AssA 0.41120, FP 1,868, FN 2,338.5, IDSW 270. The selected setting is provisional until the predeclared confirmation window (frames 4000–4999, run `p2more_w3`) is scored; it has not yet passed the confirmation gate. Records: `results/c4_sweep.json`, `results/c4_tuning.json`.
