# Phase II amendment: Tracker v3 (Deep-EIoU + GTA-Link + Kit-Color Official Filter) (2026-11-21)

Written before any Tracker v3 result exists. Phase I (`IouTracker`), Tracker v2 (`ByteIouTracker`), and C4 (`roster.py`) files and result records are untouched.

**Context from C4 confirmation (`results/c4_confirm.json`).** On the unseen confirmation window (`p2more_w3`, frames 4000–4999), pre-tracking frame-level score capping (`det K=22`) reduced FP from 1,021 to 282 (-72%) and raised DetA from 0.8374 to 0.8544, but dropped AssA from 0.6280 to 0.5664 (-0.0616) and HOTA from 0.7240 to 0.6951 (-0.0290), failing its confirmation gate. Because match officials score 0.95+, a raw score cap retains officials while dropping real players mid-trajectory.

**Detections and features.** Cached full-clip P2-more detections (`p2more_full`, score floor 0.05, NMS 0.5, merge IoU 0.5) on validation clips `117093` (frames 0–5975) and `128058` (frames 0–5980), plus chunked appearance embeddings (L2-normalized OSNet `osnet_x1_0` or documented torchvision fallback) and green-masked torso CIE-Lab color vectors extracted by `src/soccer_tracking/features.py` for detections with score >= 0.25. No test (`128057`, `132831`) or reserve (`118578`, `132877`) clip is touched.

**Method (`src/soccer_tracking/tracker_v3.py`, three composable stages):**

1. **Online Deep-EIoU (`DeepEIoUTracker`):** Motion-agnostic Iterative Scale-Up Expansion IoU (`expand_ratios=(0.5, 0.7, 0.9)`) with exact Hungarian assignment (`scipy.optimize.linear_sum_assignment`), proximity-gated appearance cosine cost, ByteTrack two-stage high/low continuation, and longer track retention (`max_age=45` frames = 1.8 s) without runaway linear velocity extrapolation.
2. **Offline Global Tracklet Association (`gta_link_connect`):** Agglomerative merging of non-overlapping tracklets using mean L2-normalized appearance cosine distance, a gap-scaled spatial speed gate, a soccer-specific **same-team torso-color gate**, and short-tracklet noise pruning (`min_track_len`).
3. **Track-Level Kit-Color Official Filter & Soft Roster Prior (`filter_officials_and_roster`):** Deterministic 2-means clustering of long midfield tracklets in torso Lab space to identify the two outfield team kits; goalmouth x-zone exemption for goalkeepers; removal of color-outlier official tracks and off-pitch sideline tracks; optional soft per-frame roster cap ranked by `(is_team_or_gk, track_length, score)`.

**Pre-registered ablation rows (full validation clips `117093` and `128058`, TrackEval):**

- `v3_eiou_only`: Stage 1 with Expansion IoU + Hungarian only (`app_weight=0.0`, no Stage 2/3).
- `v3_deep_eiou`: Stage 1 with OSNet appearance (`app_weight=0.5`, no Stage 2/3).
- `v3_gta`: Stage 1 + Stage 2 (`gta_link_connect` + short-tracklet pruning, no Stage 3).
- `v3_full`: Stage 1 + Stage 2 + Stage 3 (official filter + soft roster prior).

**Compact tuning grid (`tune_tracker_v3.py sweep`, 16 settings on full validation clips):**

- `det_high` in `{0.65, 0.75}`
- `merge_cos_thresh` in `{0.25, 0.35}`
- `official_lab_thresh` in `{28.0, 38.0}`
- `roster_k` in `{None, 24}`

Fixed defaults: `det_low=0.30`, `expand_ratios=(0.5, 0.7, 0.9)`, `prox_thresh=0.85`, `app_weight=0.50`, `app_thresh=0.45`, `max_age=45`, `min_hits=3`, `max_gap_frames=300`, `max_speed_px=12.0`, `base_dist_px=60.0`, `team_color_gate=35.0`, `min_track_len=12`, `gk_margin=0.14`.

**Selection rule.** (1) Rank all 16 settings on full-clip NumPy-reference mean HOTA. (2) Rescore the top 3 with TrackEval. (3) Pick the highest TrackEval mean HOTA; among settings within 0.005 of it, pick the lowest FP, then the lowest IDSW.

**Known limits.** Tuning uses the full validation clips (`117093`, `128058`), which also selected the P2-more training epoch; validation metrics remain development numbers. Unbiased generalization is measured once on the held-out test split (`128057`, `132831`) after committing the final configuration amendment.
