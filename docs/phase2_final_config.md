# Phase II Final Configuration Amendment (2026-11-22)

Frozen before any Phase II held-out test run (`--allow-test`). All model selection, feature extraction, and hyperparameter decisions below were made exclusively on the Phase II validation pair (`117093`, `128058`). Reserve clips (`118578`, `132877`) remain completely untouched for the live unseen viva test.

## 1. Frozen Final Pipeline

- **Detector:** `P2-more` RetinaNet (ResNet-50 FPN), checkpoint `runs/p2_more_data/best.pt` (epoch 4, selected on validation mean AP50/AP75 = `0.9248`; AP50 `0.9656`, AP75 `0.8841`).
  - Inference settings: native-scale tiling `1024×1080`, overlap `128` px, `score_floor=0.05`, `nms_thresh=0.5`, `merge_iou=0.5`, `drop_edge_px=2.0`, CUDA FP16 (`torch.inference_mode()` + `torch.amp.autocast`), `batch_size=5`.
- **Re-Identification & Appearance Features (`src/soccer_tracking/features.py`):**
  - `osnet_x1_0` fine-tuned for 15 epochs with Batch-Hard Triplet Loss (`margin=0.3`) + Label-Smoothed Cross-Entropy (`0.1`) exclusively on the 4 Phase II training clips (`118575`, `118576`, `118577`, `117092`; 88 unique player identities, stride 25), checkpoint `reid_osnet/best_osnet_soccer.pt`.
  - Extracted for all detections with `score >= 0.25`, paired with green-masked torso CIE-Lab median color vectors.
- **Tracker (`src/soccer_tracking/tracker_v3.py`, `run_tracker_v3`):**
  - Online stage (`DeepEIoUTracker`): `det_high=0.78`, `det_low=0.45`, `expand_ratios=(0.0, 0.35, 0.70)`, `prox_thresh=0.80`, `app_weight=0.50`, `app_thresh=0.45`, `online_lab_gate=42.0`, `max_age=20`, `min_hits=3`, `emb_alpha=0.90`.
  - Offline global stitching (`gta_link_connect`): `use_gta=True`, `max_gap_frames=600`, `max_speed_px=12.0`, `base_dist_px=60.0`, `merge_cos_thresh=0.20`, `team_color_gate=45.0`, `min_track_len=12`.
  - Official filter & roster cap: `use_official_filter=False`, `roster_k=None` (both rejected by validation ablations below).

## 2. Full-Clip Validation Ablation Summary (Development Evidence)

Mean over full validation clips `117093` (5,976 frames) and `128058` (5,981 frames), scored with official TrackEval:

| Stage | HOTA | DetA | AssA | LocA | FP | FN | IDSW | IDF1 | Record |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Phase I (`p1model + IoU`) | 0.3188 | 0.7063 | 0.1446 | 0.8790 | 19,768.0 | 6,392.0 | 1,483.0 | — | `results/p2_full_t0_iou.json` |
| Phase II (`P2-more + Tracker v2`) | 0.3681 | 0.7613 | 0.1792 | 0.8888 | 9,716.5 | 7,120.5 | 983.0 | 0.2900 | `results/p2_full_p2more_iou_v2.json` |
| `+ v3 EIoU + Lab Gate` (`v3_eiou_only`) | 0.3583 | 0.7798 | 0.1654 | 0.8894 | 5,257.0 | 7,488.5 | 945.5 | 0.2940 | `tracker_v3_soccer_reid_ablation.json` |
| `+ Pretrained OSNet + GTA-Link` | 0.4049 | 0.7779 | 0.2113 | 0.8893 | 5,204.5 | 7,522.0 | 874.0 | 0.3878 | `tracker_v3_ablation.json` |
| `+ Soccer-Finetuned OSNet + GTA-Link` (`dh=0.75`) | 0.4943 | 0.7791 | 0.3143 | 0.8896 | 5,219.5 | 7,519.0 | 812.0 | 0.4981 | `tracker_v3_soccer_reid_ablation.json` |
| **Final Frozen Config** (`dh=0.78, cos=0.20, gap=600`) | **0.5011** | **0.7812** | **0.3222** | **0.8898** | **4,615.5** | **7,711.0** | **787.0** | **0.5198** | `tracker_v3_soccer_reid_tuning.json` |

## 3. Documented Negative Results (Dropped)

- **B3 (NMS 0.7 / Merge IoU 0.7):** Isolated changes moved HOTA by `< 0.004`; changing both together degraded HOTA (`0.3959 -> 0.3739`).
- **Static-box post-filter (`postfilter.py`):** Removed stationary goalkeepers, raising FN and lowering HOTA.
- **C4 frame-level roster cap (`det K=22`, `results/c4_confirm.json`):** Failed confirmation on `p2more_w3` (`delta_hota = -0.0290`, `delta_assa = -0.0616`). High-scoring match officials (`>= 0.95`) displaced real players mid-track.
- **Track-level Lab official filter (`v3_full`):** Dropped `~11,800` real player frames (`FN 7,519 -> 19,305`) for only `-140` FP because fisheye shadows shift real player Lab values more than referees differ from the two team centroids.

## 4. Held-Out Test Protocol & Disclosures

- **Test pair:** `128057` (`4096×1080`, 5,976 frames) and `132831` (`3840×1504`, 5,976 frames — resolution-shift check).
- **Disclosure:** `128057` was previously scored once in Phase I (`results/test_finetuned.json`, `results/tracking_test.json`) on the frozen Phase I model and tracker. Neither test clip was used for any Phase II training, feature tuning, or hyperparameter choice.
- **Single run:** Execute detection, OSNet feature extraction, and `score_final_test.py` **once** with `--allow-test`, recording the output to `results/p2_final_test.json`.
