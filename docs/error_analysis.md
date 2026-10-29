# Error analysis of the fine-tuned detector (validation clip 117093)

Operating point: score >= 0.5, IoU >= 0.5, the 48 validation frames used everywhere else (every 125th frame), `best.pt` (epoch 3). Counts come from `results/errors_val/summary.json` and `errors.csv` (the pretrained model: `results/errors_val_baseline/`). The image panels were reviewed by the student; they contain frames of people and are kept local, not committed. Tool and error types: README, "Error analysis".

## 1. Counts

| | Fine-tuned | Pretrained |
|---|---|---|
| Players (ground truth) / found / missed | 1,056 / 1,012 / 44 (recall 0.958) | 1,056 / 744 / 312 (recall 0.705) |
| Confident detections / false | 1,158 / 146 (precision 0.874) | 1,005 / 261 (precision 0.740) |

| Missed players by reason | Fine-tuned | Pretrained |
|---|---|---|
| a confident box covers them but was matched to a neighbour | 17 | 15 |
| a box with IoU >= 0.5 exists but scores below 0.5 | 17 | 258 |
| only an overlapping, poorly placed box | 9 | 36 |
| nothing overlaps | 1 | 3 |

| False positives by reason | Fine-tuned | Pretrained |
|---|---|---|
| duplicate (IoU >= 0.5 with a player who has a better box) | 7 | 0 |
| poorly placed (IoU 0.1-0.5 with a player) | 28 | 23 |
| overlaps no labelled player | 111 | 238 |

Fine-tuned recall by box height: under 30 px 0.757 (n = 37), 30-40 px 0.935 (246), 40-50 px 0.952 (374), 50-70 px 0.996 (266), 70 px and above 1.000 (133). Crowded players (at least 20% of the box covered by a neighbour's box): recall 0.750 (n = 148) against 0.992 otherwise (n = 908); 37 of the 44 misses are crowded players. Within 16 px of a tile border: 0.914 (n = 116) against 0.964. Precision by score band: 0.25 for 0.5-0.7 (72 detections), 0.53 for 0.7-0.9 (104), 0.96 for 0.9 and above (982).

## 2. What the panels show

The student's observations on the sampled panels (up to 8 per type, chosen at random with seed 0; "k of n" is within the sample, not a rate). Yellow boxes are labelled players, so "no yellow box" means no label.

| Error type | In the sample / in total | Observed |
|---|---|---|
| False positive, duplicate | 7 of 7 / 7 | all 7 boxes cover 2 players |
| False positive, poorly placed | 8 / 28 | 2 are referees with no label; 6 cover 2 players (2 of these also overlap a referee or an unlabelled outfield player) |
| False positive, overlaps no labelled player | 8 / 111 | all 8 are match officials or outfield players with no label |
| Missed, merged with a neighbour | 8 / 17 | all 8: two or more players overlap into one detection |
| Missed, low score | 8 / 17 | 1 unusual pose; 1 mostly hidden; 2 half hidden (1 of them blurred); 1 distant; 3 look fine and are probably missed because of distance or a neighbour |
| Missed, poorly placed box | 8 / 9 | 7 are overlapping players at the same position (in 1 the box is too small); 1 is a player in an unusual position |
| Missed, nothing found | 1 of 1 / 1 | a distant, blurred player |

## 3. Reading

1. **Overlapping players are the main failure, on both sides.** The duplicate false positives (7 of 7) and most poorly placed ones (6 of 8) are single boxes spanning two players, and the misses of type "merged" (8 of 8) and "poorly placed" (7 of 8) are players that overlap another. This matches the counts: 37 of 44 misses are crowded players. The usual pattern is one box for two players, which counts as one false positive and one or two misses.
2. **Most false positives are people the labels do not cover.** In the sample, every "overlaps no labelled player" case (8 of 8, out of 111) is a match official or a person who looks like an outfield player but has no label. Reported precision therefore understates how well the model finds people on the pitch; it also means the model cannot tell players from officials, because officials are not a class in the labels. Whether the unlabelled outfield players are substitutes or players the annotation omitted was not determined.
3. **Most remaining misses have a visible cause.** Of the 8 sampled low-score misses, 3 are hidden or half hidden, 1 has an unusual pose, 1 is distant; 3 show no clear cause. The single "found nothing" case is distant and blurred. The recall by height (0.757 under 30 px) agrees with distance being a factor.
4. **Fine-tuning fixed the confidence, not the crowding.** The pretrained model found most players but scored them below 0.5 (258 of its 312 misses), and fired confidently on 238 unlabelled people; fine-tuning removed most of the first problem and about half of the second, while crowded players remain hard (recall 0.318 pretrained, 0.750 fine-tuned).
5. **Link to tracking (hypothesis, not tested).** Frames where players overlap are also where a one-box-for-two detection can swap identities, which is a plausible source of the ID switches in the tracking study.

## 4. Consequences for the report and ideas (suggestions, not done)

- Report precision as "precision against the labels" and state that officials and some unlabelled people are counted as false positives; do not describe the 111 as detector errors without this caveat.
- To check whether labels are missing on the pitch, count how often the same unlabelled person recurs across frames (`errors.csv`, type `overlaps no labelled player`) and look at a few full frames.
- For crowds: test a higher NMS threshold or soft-NMS inside the detector and at the tile merge on validation (a one-box-for-two result can come from either); a higher-resolution feature level is a larger change.
- For officials: a pitch mask or a role label would be needed to filter them; neither is in the dataset.
- Lowering the score threshold would recover up to 17 low-score misses at the cost of the lower-precision detection bands.

## 5. Limits

One clip, 48 frames; panels are samples of at most 8 per type; one reviewer; officials and substitutes were not distinguished from unlabelled players; no ground truth for what the unlabelled people are.
