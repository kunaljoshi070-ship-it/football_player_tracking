# Error Analysis v2 — Phase II Final System

This document extends Phase I's single-clip detector error study (`docs/error_analysis.md`) to full 4-minute clips across the Phase II validation pair (`117093`, `128058`) and held-out test pair (`128057`, `132831`), using TrackEval records (`results/p2_full_*.json`, `results/tracker_v3_soccer_reid_*.json`, `results/c4_confirm.json`, `results/p2_final_test.json`).

## 1. Error Reduction Across Iterations (Full-Clip Validation Mean)

| Stage | FP ↓ | FN ↓ | IDSW ↓ | DetA ↑ | AssA ↑ | LocA ↑ | HOTA ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase I (`p1model + IoU`) | 19,768.0 | 6,392.0 | 1,483.0 | 0.7063 | 0.1446 | 0.8790 | 0.3188 |
| Phase II (`P2-more + Tracker v2`) | 9,716.5 | 7,120.5 | 983.0 | 0.7613 | 0.1792 | 0.8888 | 0.3681 |
| Phase II (`v3 + Pretrained OSNet + GTA`) | 5,204.5 | 7,522.0 | 874.0 | 0.7779 | 0.2113 | 0.8893 | 0.4049 |
| **Phase II Final (`+ Soccer-Finetuned OSNet`)** | **4,615.5** | 7,711.0 | **787.0** | **0.7812** | **0.3222** | **0.8898** | **0.5011** |

- **False positives fell by 76.7%** (`19,768 → 4,615.5`) through 4-clip detector training, eliminating coasting output, raising initiation confidence (`det_high=0.78`), and pruning short noise tracklets (`min_track_len=12`).
- **Identity switches fell by 46.9%** (`1,483 → 787`) and **AssA more than doubled** (`0.1446 → 0.3222`, `IDF1 0.290 → 0.520`) after replacing linear velocity extrapolation with motion-agnostic Deep-EIoU, adding an online torso-Lab team gate, and domain-adapting `osnet_x1_0` on the 4 training matches.
- **Localization precision (`LocA = 0.8898` val, `0.8842` matched test)** consistently exceeds GTATrack (`0.84`) because native-scale `1024×1080` tiling preserves full pixel resolution on distant 20–40 px players.

## 2. Why Automated Official Filtering Failed (C4 & Lab Filter)

Two independent approaches to suppress unlabelled match officials failed on validation:

1. **Frame-level score cap (`C4 det K=22`, `results/c4_confirm.json`):** Slashed FP by 72% (`1,021 → 282`), but lowered AssA by `-0.0616` and HOTA by `-0.0290`. Match officials score `>= 0.95`, so capping at 22 boxes per frame kept referees while evicting real players whose score briefly dipped during occlusions.
2. **Track-level Lab color outlier filter (`v3_full`, `results/tracker_v3_soccer_reid_ablation.json`):** Even after calibrating the Lab threshold to `46.0` with goalmouth goalkeeper exemptions, it dropped `+11,786` real player frames (`FN 7,519 → 19,305`) for only `-140` FP. On fisheye stadium views, shadow transitions and distant blur shift real players' torso Lab values more than officials differ from team centroids.

## 3. Resolution-Shift Failure Mode on Held-Out Test (`132831`)

Comparing the two held-out test clips (`results/p2_final_test.json`) isolates a clear deployment boundary:

| Clip | Resolution | HOTA | DetA | AssA | LocA | FP | FN | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `128057` | `4096×1080` | 0.4778 | 0.7743 | 0.2952 | 0.8842 | 3,336 | 8,912 | 689 |
| `132831` | `3840×1504` | 0.3876 | 0.5375 | 0.2802 | 0.8691 | **67,955** | 4,654 | 685 |

- **Tracker & ReID generalized cleanly across the resolution shift:** `AssA` stayed nearly flat (`0.295 → 0.280`), `IDSW` was virtually identical (`689` vs. `685`), `LocA` remained `0.869` (above GTATrack's `0.84`), and recall on `132831` was higher (`FN = 4,654` vs. `8,912`).
- **Root cause of the `67,955` FP spike (~11.4 extra boxes/frame):** At `3840×1504` (40% taller vertically than `4096×1080`), the extra 424 vertical pixels expose **sidelines, team benches, staff, and spectator stands** outside the playing pitch. Because the detector is a person-trained single-class model with no pitch-boundary mask, it confidently detects off-pitch people as players.

## 4. Remaining Gap & Concrete Next Steps

1. **Pitch-polygon / turf-mask pre-filter:** Masking detections outside the contiguous green pitch hull (estimated per static camera once on frame 0) directly addresses the `132831` spectator/bench FP explosion without harming players.
2. **Tracklet Splitter + Crowd NMS:** Remaining `IDSW` (`~687` on test) occurs primarily when same-team players cross in dense goalmouth clusters; adding a GTA-Link *Splitter* (splitting tracklets with bimodal ReID clusters) and training a dedicated `non_player` class for officials are the cleanest next levers.
