# Phase I experiment plan

## Question and hypothesis

On a fixed sample of soccer clips, does short-window temporal association improve trajectory continuity compared with independent per-frame detector outputs? Hypothesis H1 and its tolerance are stated in the proposal; pre-register the final clip sample and thresholds before evaluation.

## Conditions

| Condition | Description | Purpose |
|---|---|---|
| A: detector baseline | Same pretrained detector applied independently to sampled frames | Detection reference |
| B: detector + simple association | Same detections linked using a fixed IoU/position rule and short lost-track buffer | Test temporal continuity |

If a suitable training subset and compute are available and required by the rubric, add one bounded fine-tuning experiment as a separate comparison. Do not make fine-tuning a prerequisite for initial feasibility.

## Protocol

1. Freeze dataset edition, clip IDs, split, frame sampling rate, and annotation filter in `data/manifest.csv`.
2. Use identical input frames and detector outputs for both conditions; document detector and tracker parameters in `configs/phase1.yaml`.
3. Tune thresholds only on training/validation clips. Keep evaluation clips untouched until settings are frozen.
4. Record software versions, hardware, random seeds, command, date, and output revision for every run.
5. Compute frame-level detection precision/recall and AP at a declared IoU threshold if supported. For tracking, use HOTA/DetA/AssA with a compatible evaluator if feasible; otherwise define and report ID switches, fragmentations, and track coverage.
6. Summarize per clip and overall, include runtime per frame, and inspect representative failures without cherry-picking.
7. Save machine-readable metrics and small derived plots in `results/`. Do not commit raw video or weights.

## Reproducibility record

Each run should record: run ID; git commit; dataset edition/split/clip IDs; frame indices; detector model and weight origin/license; configuration; seeds; environment; hardware; exact command; metrics; and known deviations. Put details in a CSV/JSON alongside result summaries.

## Interpretation

The experiment tests a practical temporal baseline, not a new general tracking algorithm. Image-plane trajectories cannot directly establish field position, speed, formation, or tactical intent without camera calibration and additional assumptions. If gated video access or annotation alignment fails, stop and revise the sample/data plan before coding the video pipeline.
