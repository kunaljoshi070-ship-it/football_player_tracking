# Phase II amendment: C4 roster cap (2026-11-21)

Written before any C4 result exists. C4 is a cap on how many boxes or tracks may exist per frame, because every frame
has exactly 22 labelled players. Tracker v2 settings are frozen from `results/tracker_v2_tuning.json` (chosen) and are not retuned.

**Detections.** Cached P2-more detections (score floor 0.05, NMS 0.5, merge IoU 0.5). Validation clips 117093 and 128058 only.

**Grid (7, fixed).** `none` (plain tracker v2, the comparison arm); `det` K in {22, 24, 26}: keep the K highest-scoring
detections per frame before tracking; `out` K in {22, 24, 26}: after tracking, output at most K tracks per frame, ranked by
track age (frames since first output, counted before any dropping), then by score.

**Tuning window.** Frames 1000-1999 of both validation clips (run `p2more`). Development numbers: the P2-more epoch and the
v2 settings were also chosen on these clips.

**Selection rule.** (1) Rank all 7 on NumPy-reference mean HOTA. (2) Rescore the top 3 plus `none` with TrackEval.
(3) Best TrackEval mean HOTA; if several are within 0.005 of it, take the lowest IDSW, then the larger K (`none` counts as
largest). If `none` is chosen, C4 is rejected and no confirmation is needed.

**Confirmation.** Frames 4000-4999 of both validation clips (run `p2more_w3`, same checkpoint and detection settings),
a window not used by any earlier stage. Success: chosen C4 beats `none` by at least +0.005 mean HOTA (TrackEval), AssA not
lower, and FP not higher. Reported whether or not it passes.

**Known limits.** Two clips, one window per stage. The cap cannot remove officials who score highly; it only removes the
lowest-ranked extras. It assumes 22 players are on screen (substitutions or red cards would break it). In `det` mode a cap
can drop low-score boxes that continued a track. The tie-break and the +0.005 threshold are my proposal; change them before
committing if you disagree.