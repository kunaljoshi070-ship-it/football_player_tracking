# Model Card: Fisheye Soccer Player Detector & Offline Tracker (Phase II)

## 1. Model Details

- **System:** Two-stage offline multi-object tracking pipeline for fixed-camera fisheye soccer footage:
  1. **Detector (`P2-more`):** `torchvision` RetinaNet (ResNet-50 FPN, 32.17M parameters) fine-tuned from COCO weights on native-scale `1024×1080` tiles (overlap `128` px) across 4 SoccerTrack v2 training matches (`118575`, `118576`, `118577`, `117092`).
  2. **Appearance & Kit Representation:** `osnet_x1_0` fine-tuned with Batch-Hard Triplet Loss + Label-Smoothed Cross-Entropy on the 4 training matches (88 player identities), paired with green-masked torso CIE-Lab color vectors.
  3. **Association (`Tracker v3`):** Motion-agnostic Iterative Scale-Up Expansion IoU (`e = 0.0, 0.35, 0.70`) with Hungarian matching and an online torso-Lab team gate, followed by offline Global Tracklet Association (`GTA-Link` connector, `max_gap=600`, `merge_cos_thresh=0.20`).
- **License & Code Provenance:** Project code written in Python (PyTorch, torchvision, OpenCV, NumPy, SciPy). Evaluation uses MIT-licensed TrackEval; ReID backbone uses MIT-licensed `deep-person-reid` (`osnet_x1_0`).

## 2. Intended Use & Non-Intended Use

- **Intended use:** Offline post-match tactical review and spatial analytics (trajectories, pitch heat maps, crowding/coverage review) on fixed-camera full-pitch soccer recordings by analysts and coaches.
- **Out-of-scope / non-intended use:**
  - Identifying or biometrically profiling named individuals (outputs are anonymous image-plane track IDs, not verified player identities).
  - Automated officiating, disciplinary decisions, or individual player performance grading / contract evaluation.
  - Real-time broadcast or moving/panning/zooming camera feeds (the pipeline assumes a static full-pitch camera).
  - General public surveillance outside sports footage.

## 3. Data, Splits & Consent Notes

- **Dataset:** SoccerTrack v2 (`v1.2`, gated Hugging Face release under CC BY 4.0; terms accepted at download). Raw videos, frames, and person crops are kept strictly local/Drive and are never committed to Git.
- **GTATrack-aligned Phase II splits:** Train (`118575`, `118576`, `118577`, `117092`), Validation (`117093`, `128058`), Held-out Test (`128057`, `132831` — includes `3840×1504` resolution shift), Reserve (`118578`, `132877` — untouched for live unseen evaluation).

## 4. Quantitative Performance (Full-Clip Validation Mean, TrackEval)

- **Full-clip validation (`117093`, `128058`):** `HOTA = 0.5011`, `DetA = 0.7812`, `AssA = 0.3222`, `LocA = 0.8898`, `IDF1 = 0.5198`, `FP = 4,615.5`, `FN = 7,711.0`, `IDSW = 787.0`.
- **Against baselines:** `+0.1823 HOTA` over Phase I (`0.3188`), `+0.1330 HOTA` over Phase II Tracker v2 (`0.3681`), `+0.081 HOTA` over GTATrack's reported ByteTrack baseline (`0.42`), and higher `DetA` (`0.781` vs. `0.76`) and `LocA` (`0.890` vs. `0.84`) than GTATrack's reported full system—while remaining behind GTATrack on `HOTA` (`0.501` vs. `0.60`), `AssA` (`0.322` vs. `0.47`), and `FP` (`4,615.5` vs. `982`).
- **Inference speed:** ~`0.35–0.45 s/frame` on NVIDIA Tesla T4 (FP16, `batch_size=5`); ~`20 s/frame` on laptop CPU (live runs require a GPU; CPU demo uses cached detections).

## 5. Known Limitations, Bias & Risks

1. **Image-plane track IDs are not player identities:** Over 4 minutes, `AssA = 0.3222` and `IDSW = 787` mean identities still swap during prolonged dense crowds (especially in distorted fisheye goalmouths). The UI displays this caveat permanently.
2. **Unlabelled match officials inflate false positives:** SoccerTrack labels only the 22 players. High-confidence detections of referees, assistant referees, and sideline personnel count as false positives; automated Lab color filtering was tested and rejected because shadows caused excessive player misses.
3. **Small/crowded player recall gap:** Phase I error analysis showed recall drops on players `< 30 px` tall (`0.757`) and heavily occluded/crowded players (`0.750` vs. `0.992` isolated).
4. **Domain shift & lighting sensitivity:** Trained on 4 daytime/stadium fisheye clips; night matches, heavy rain, different camera heights, or non-static cameras will degrade accuracy and should be validated before use.
