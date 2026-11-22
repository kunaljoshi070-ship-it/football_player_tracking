# Experiment log (Phase I & Phase II)

Every run whose result is reported. Dates are UTC from the result files where available; for blank fields open the named file (`environment.git_revision` and `git_dirty` are recorded inside each JSON). The Phase I protocol was frozen in `docs/phase1_proposal.md` (amendment of 2026-10-21) before the tracking and test results existed. Test-split runs are marked **TEST**; each was run once.

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

## Phase II evaluation log (2026-11-21 to 2026-11-22)

All metrics below are from the `trackeval` sections of the named JSON files (`trackeval_error: null`). Full-clip validation scores are development results: P2-more's epoch had been selected using these validation clips (`117093` frames 0–5975 and `128058` frames 0–5980). Each clip contributes equally to the reported per-clip mean.

### Full-Clip Validation Ablation (117093 & 128058)

| Run | Setup | Mean HOTA / DetA / AssA / LocA | Mean FP / FN / IDSW / IDF1 | Records |
|---|---|---|---|---|
| T0 `p1model_full`, IoU | Phase I weights and frozen Phase I IoU tracker | 0.3188 / 0.7063 / 0.1446 / 0.8790 | 19,768 / 6,392 / 1,483 / — | `results/p2_full_t0_iou.json` |
| T0 `p1model_full`, IoU-v2 | Phase I weights; tracker-v2 (`det_high=0.6, det_low=0.3, match_iou=0.3, max_age=5, min_hits=3`) | 0.3439 / 0.7175 / 0.1659 / 0.8796 | 15,972 / 7,544 / 1,126.5 / — | `results/p2_full_t0_iou_v2.json` |
| P2-more `p2more_full`, IoU | P2-more weights and frozen Phase I IoU tracker | 0.3436 / 0.7557 / 0.1575 / 0.8884 | 12,097 / 6,143 / 1,356.5 / — | `results/p2_full_p2more_iou.json` |
| P2-more `p2more_full`, IoU-v2 | P2-more weights and tracker-v2 | 0.3681 / 0.7613 / 0.1792 / 0.8888 | 9,716.5 / 7,120.5 / 983 / 0.2900 | `results/p2_full_p2more_iou_v2.json` |
| P2-more + v3 EIoU only (`v3_eiou_only`) | Motion-agnostic EIoU (`e=0.0, 0.35, 0.70`) + Hungarian + online Lab gate (`dh=0.75, dl=0.45`) | 0.3583 / 0.7798 / 0.1654 / 0.8894 | 5,257.0 / 7,488.5 / 945.5 / 0.2940 | `results/tracker_v3_soccer_reid_ablation.json` |
| P2-more + v3 + MobileNetV3 + GTA-Link | Off-the-shelf `torchvision:mobilenet_v3_large` + GTA-Link (`dh=0.75, cos=0.30, gap=300`) | 0.4144 / 0.7771 / 0.2215 / 0.8893 | 5,225.0 / 7,535.0 / 869.5 / 0.3901 | `results/tracker_v3_1_mobilenet_ablation.json` |
| P2-more + v3 + Pretrained OSNet + GTA-Link | Pretrained `torchreid:osnet_x1_0` + GTA-Link (`dh=0.75, cos=0.30, gap=300`) | 0.4049 / 0.7779 / 0.2113 / 0.8893 | 5,204.5 / 7,522.0 / 874.0 / 0.3878 | `results/tracker_v3_ablation.json` |
| P2-more + v3 + **Soccer-Finetuned OSNet** + GTA-Link | `osnet_x1_0` fine-tuned 15 epochs on 4 train clips (88 IDs, Triplet + CE); `dh=0.75, cos=0.30, gap=300` | 0.4943 / 0.7791 / 0.3143 / 0.8896 | 5,219.5 / 7,519.0 / 812.0 / 0.4981 | `results/tracker_v3_soccer_reid_ablation.json` |
| **Final Frozen Config (Val)** | Same detector & fine-tuned OSNet; `dh=0.78, dl=0.45, cos=0.20, gap=600, official_filter=False` | **0.5011 / 0.7812 / 0.3222 / 0.8898** | **4,615.5 / 7,711.0 / 787.0 / 0.5198** | `results/tracker_v3_soccer_reid_tuning.json` |

### Closed Negative Results (Validation)

- **B3 Threshold isolation (`117093`, frames 1000–1999):** NMS 0.7 / merge IoU 0.5 gave IoU-tracker HOTA 0.3988 vs. 0.3959 (+0.0029, higher FN); merge IoU 0.7 / NMS 0.5 gave 0.3953. Changing both together degraded HOTA (0.3959 to 0.3739). Dropped. Records: `results/p2more_nms07_*.json`, `results/p2more_merge07_*.json`.
- **C4 Roster-cap confirmation (`p2more_w3`, frames 4000–4999):** Pre-tracking cap `mode=det, K=22` failed confirmation (`results/c4_confirm.json`): FP fell from 1,021 to 282 (-739), raising DetA (0.8374 → 0.8544), but AssA dropped from 0.6280 to 0.5664 (-0.0616) and HOTA fell from 0.7240 to 0.6951 (-0.0290) because high-scoring officials evicted real players mid-track. Dropped.
- **Track-level Lab Official Filter (`v3_full`):** Dropped ~11,800 real player frames (`FN 7,519 → 19,305`) for only -140 FP (`HOTA 0.4943 → 0.4311`) because fisheye shadows shift real player Lab values more than officials differ from team centroids. Dropped.

### **TEST** — Phase II Final Held-Out Evaluation (2026-11-22)

Protocol frozen prior to execution in `docs/phase2_final_config.md` (`git_revision: f20b286070613a4cf66a36ef9082218501581854`, `git_dirty: false`). Run once via `--allow-test`. Disclosure: clip `128057` was previously scored once in Phase I on the frozen Phase I pipeline; clip `132831` (`3840×1504`) is a resolution-shift check. Reserve clips (`118578`, `132877`) remain untouched.

| Clip | Resolution | HOTA | DetA | AssA | LocA | FP | FN | IDSW | IDF1 | Record |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`128057`** (matched res) | `4096×1080` | **0.4778** | **0.7743** | **0.2952** | **0.8842** | **3,336** | 8,912 | **689** | **0.4801** | `results/p2_final_test.json` |
| **`132831`** (resolution shift) | `3840×1504` | **0.3876** | 0.5375 | **0.2802** | **0.8691** | 67,955 | **4,654** | **685** | 0.3618 | `results/p2_final_test.json` |
| **Test Mean** | — | **0.4327** | 0.6559 | 0.2877 | 0.8767 | 35,645.5 | 6,783.0 | 687.0 | 0.4210 | `results/p2_final_test.json` |

Runtime on Colab Tesla T4 with FP16 + `batch_size=5`: `0.36–0.43 s/frame` on `128057` (`4096×1080`, 5 tiles) and `0.58–0.82 s/frame` on `132831` (`3840×1504`, 8 tiles), down from `~1.1 s/frame` in Phase I.
