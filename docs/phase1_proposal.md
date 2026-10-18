# Phase I proposal

**Course:** CSAI3027P Deep Learning PBL  
**Project:** 36 — Soccer player tracking and basic tactical analysis  
**Status:** Draft pending course-reference review and dataset access confirmation.

## Problem

Manual analysis of soccer footage is time-consuming. This Phase I project will investigate a narrow computer-vision pipeline that detects people/players in a few fixed clips and, where ground-truth tracking data are available, associates detections across nearby frames to produce interpretable image-plane trajectories.

## Aim and research question

Can a lightweight temporal association step improve the continuity of player trajectories over a frame-by-frame detection baseline on a fixed, annotated sample of soccer footage, without reducing detection quality beyond a documented tolerance?

## Proposed method

Use a pretrained person detector as the reference baseline, then evaluate one simple tracker (e.g. IoU/position-based association with a short lost-track buffer). Fine-tuning a detector is conditional on having a suitable labeled training subset, compute, and rubric support. Keep train/validation/test clips separated by match or source sequence where possible to reduce leakage. Visualize bounding boxes and short image-plane trajectories. Any tactical summary is limited to transparent descriptive quantities such as observed movement paths or coarse spatial occupancy; no player identity, pitch coordinates, or tactical causation claims without the data and calibration to support them.

## Scope and feasibility limits

- Soccer only; one fixed camera/view type where possible.
- A small, predeclared subset of clips and frame sampling rate; final counts are set after verifying data access and annotation alignment.
- Player/person detection is the minimum deliverable. Tracking visualization is conditional on video rights and compatible annotations.
- No ball tracking, jersey-number recognition, cross-camera re-identification, automatic formations, event recognition, or live deployment in Phase I.
- Do not redistribute source video or model weights.

## Proposed innovation hypothesis

**H1:** On the same fixed clips and detector outputs, short-window IoU/position-based temporal association will reduce identity switches or track fragmentation relative to treating detections as independent frame outputs, while keeping detection precision/recall within a predeclared tolerance (suggested: no more than 2 percentage points on the same sampled frames). This is a modest, testable systems hypothesis; it is not a claim of a novel tracking algorithm.

## Evaluation

Detection: precision, recall, and AP/mAP at a stated IoU threshold if labels and evaluator support it. Tracking: HOTA with DetA/AssA when the official-format annotations and compatible evaluator can be used; otherwise report clearly defined ID switches, fragmentation, and track coverage with the limitations. Also report runtime per frame on specified hardware as a descriptive measure. Compare framewise detections to the temporal association output on identical frames. Report clip-level results and aggregate summary; do not tune on held-out evaluation clips.

## Learning and reproducibility

Explain the selected detector's neural-network components and learning/inference pipeline at a level mapped to the rubric. Record model/version, weights source and license, input resolution, thresholds, tracker parameters, sample IDs, split, seeds, environment, hardware, exact commands, and evaluation code. Include at least one controlled comparison and discuss failure cases such as occlusion, camera motion, similar uniforms, and crowded scenes.

## Phase I deliverables

1. Approved proposal and exact rubric mapping.
2. Dataset manifest and access/rights decision.
3. Reproducible baseline run on the fixed clip sample.
4. Comparative experiment for H1 and error analysis.
5. Trajectory visualization where allowed and supported.
6. Short report/demo with measured results, limitations, and contribution log.

Phase II is intentionally not specified as implementation work here; only the course-approved phase boundary should determine its later scope.
