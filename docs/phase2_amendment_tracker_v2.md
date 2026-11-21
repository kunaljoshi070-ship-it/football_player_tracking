# Phase II amendment: tracker v2 protocol (2026-11-21)

Written before any tracker v2 result exists. Tracker v2 is `ByteIouTracker` in `src/soccer_tracking/tracker_v2.py`; the Phase I `IouTracker` and its settings (`results/tracker_tuning.json`) are unchanged and are the comparison arm.

**Detections.** Cached P2-more detections (`best.pt` of `runs/p2_more_data`, score floor 0.05, NMS 0.5, merge IoU 0.5). Validation clips 117093 and 128058 only. No test or reserve clip is touched.

**Tuning window.** Frames 1000-1999 of both validation clips (run `p2more`). Note: the P2-more epoch was also chosen on these clips, so absolute numbers are development numbers.

**Grid (48, fixed):** `det_high` {0.5, 0.6}; `det_low` {0.1, 0.3}; `match_iou` {0.3, 0.4}; `max_age` {5, 15, 30}; `min_hits` {2, 3}. Fixed: velocity prediction on, `vel_alpha` 0.5.

**Selection rule.** (1) Rank all 48 on mean-over-clips HOTA from the NumPy reference. (2) Rescore the top 3 with TrackEval. (3) Choose the best TrackEval mean HOTA; if several are within 0.005 of it, take the lowest IDSW, then the smaller `max_age`.

**Confirmation.** Frames 3000-3999 of both validation clips (run `p2more_w2`, same checkpoint and detection settings), never used for tuning. Success: chosen v2 beats the inherited Phase I settings by at least +0.01 mean HOTA (TrackEval) with AssA not lower. Reported whether or not it passes.

**Not part of this step.** Roster cap (C4) and appearance features (C2) are separate ablation rows added one at a time later.

**Known limits.** Two clips, one 1,000-frame window each per stage; matched low-score boxes are output (ByteTrack convention), which can add false positives, so FP is reported alongside HOTA; ids are image-plane track ids, not player identities.
