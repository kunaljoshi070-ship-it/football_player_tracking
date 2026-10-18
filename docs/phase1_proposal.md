# Phase I proposal — Project 36

**Problem and user.** Football coaches and analysts need a repeatable way to review player movement from match footage. This project will evaluate a small Python pipeline that detects players in soccer clips and, when compatible labels are available, follows their trajectories.

**Question and hypothesis.** On a fixed sample of annotated soccer clips, does adding a simple IoU tracker to a player detector improve trajectory continuity over frame-by-frame detections? The tracker should reduce fragmentation or identity switches while preserving frame-level detection quality. Final metrics and a numeric target will be selected after checking the labels, then fixed before evaluation.

**Method.** Use Python, PyTorch, and OpenCV, as permitted by the professor. Train or fine-tune one detector from the project brief's prescribed family, preferably RetinaNet with a ResNet-50 feature backbone, if the selected boxes and available compute support it. Compare detector-only output with simple IoU association on identical frames. Report precision, recall and AP for detection; use HOTA/DetA/AssA or clearly defined tracking measures when labels support them. Include learning curves, representative errors, runtime and limitations.

**Data.** Use SoccerTrack v2 v1.2, initially only a few short videos and their matching MOT boxes/IDs. Split by match, not by nearby frames. Record exact files and splits in `data/manifest.csv`. Review `docs/dataset_card.md` for access, license, download size, and dataset limitations before obtaining data.

**Scope.** Phase I covers the data audit, player-detection baseline, initial training evidence, a small controlled tracking comparison and reproducibility. Ball detection, field segmentation, action classification, a second deep architecture, sequence model, VAE and finished app are deferred to Phase II unless the instructor changes the boundary. Trajectories are image-plane observations unless calibrated; do not infer player identities or tactical intent.

**Reproducibility and risks.** Verify that the selected MOT clips have aligned video and labels and allow a match-separated evaluation. If not, ask the instructor before changing datasets or splitting frames. Record dataset release/license, clip and split IDs, preprocessing, seed, model/weight source, loss, optimizer, learning rate, batch size, epochs, hardware, software versions, commands, git revision, metrics and error cases. Keep raw video, crops, weights and credentials out of Git and external AI services.

**Phase I evidence.** Approved one-page proposal; dataset card and leakage-safe manifest; working baseline with initial metric; architecture/tensor-flow explanation; learning curves; controlled comparison; error analysis; reproducible instructions; individual contribution log; and timestamped `Phase_I` snapshot.
