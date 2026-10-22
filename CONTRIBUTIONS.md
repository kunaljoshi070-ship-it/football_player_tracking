# Contribution log

The assignment is individually assessed. Generative AI is allowed only if faculty permits it, and each use must be disclosed prompt-wise. Record substantive assistance and verify every accepted change yourself.

## Source log (consolidated)

Every external resource used in the submitted work. Links marked † were written from memory and should be clicked once before submission. Licences are as stated by the resource; confirm anything marked "confirm".

| Resource | Type | Link | Licence or terms | Used for | Adapted? |
|---|---|---|---|---|---|
| SoccerTrack v2 v1.2, MOT subset: clips and labels (matches 118575, 118576, 117093 and 128057 in the experiments; all ten were downloaded, 117092 only for early tool checks) | Dataset | https://huggingface.co/datasets/atomscott/soccertrack-v2 | CC BY 4.0 (gated: access conditions accepted at download); attribution required | Training, validation and test | No. Used unmodified; frames are extracted to local JPEG files that are not committed |
| Scott et al., SoccerTrack v2, arXiv:2508.01802 (2025) | Paper | https://arxiv.org/abs/2508.01802 | n/a | Dataset citation and split description | n/a |
| torchvision RetinaNet (ResNet-50 FPN) with `RetinaNet_ResNet50_FPN_Weights.COCO_V1` weights | Architecture and pretrained weights | https://pytorch.org/vision/stable/models/retinanet.html † and https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth | torchvision code: BSD-3-Clause; the weights were trained on COCO (confirm the weights' terms in the torchvision model documentation) | Pretrained baseline; initialisation for fine-tuning | Only the final classification convolution is replaced (`adapt_head`); the frozen layers are torchvision's default for pretrained weights; no code copied |
| PyTorch and torchvision | Libraries | https://pytorch.org | BSD-3-Clause | Model, training, evaluation | No (public API) |
| OpenCV (`opencv-python`) | Library | https://opencv.org | Apache-2.0 | Video decoding, JPEG files, drawing | No |
| NumPy | Library | https://numpy.org | BSD-3-Clause | Arrays, metrics, tracker | No |
| Matplotlib | Library | https://matplotlib.org | Matplotlib licence (BSD-compatible) | Learning curves, architecture diagram | No |
| `huggingface_hub` | Library | https://github.com/huggingface/huggingface_hub | Apache-2.0 | Downloading the dataset in Colab | No |
| Google Colab (T4 GPU) | Platform | https://colab.research.google.com | Google terms of service | Training and evaluation compute | n/a |
| Lin et al., Focal Loss for Dense Object Detection (RetinaNet), 2017 | Paper | https://arxiv.org/abs/1708.02002 | n/a | Architecture and loss explained in `docs/architecture.md` | No code |
| Lin et al., Feature Pyramid Networks for Object Detection, 2017 | Paper | https://arxiv.org/abs/1612.03144 | n/a | Same | No code |
| He et al., Deep Residual Learning for Image Recognition, 2016 | Paper | https://arxiv.org/abs/1512.03385 | n/a | Same | No code |
| Lin et al., Microsoft COCO: Common Objects in Context, 2014 | Paper and dataset | https://arxiv.org/abs/1405.0312 | n/a (the pretrained weights were trained on it) | Origin of the pretrained weights | No |
| Bochinski, Eiselein, Sikora, High-Speed Tracking-by-Detection Without Using Image Information, AVSS 2017 | Paper | none (conference paper) | n/a | Idea behind the IoU tracker | Re-implemented from scratch, no code |
| Bewley et al., Simple Online and Realtime Tracking (SORT), 2016 | Paper | https://arxiv.org/abs/1602.00763 | n/a | Idea of motion prediction and association | Re-implemented from scratch (no Kalman filter), no code |
| Bernardin and Stiefelhagen, CLEAR MOT metrics, EURASIP JIVP 2008 | Paper | https://doi.org/10.1155/2008/246309 † | n/a | Definitions of MODA, MOTA and ID switches | Definitions only; fragmentation per track is a project-specific measure |

**Libraries listed in `requirements.txt` but not used by the code:** `pandas` and `PyYAML` (to be removed when the versions are pinned).

**Notebooks, tutorials and repositories:** none were copied or adapted. The Colab cells in the README were written by the student with AI assistance (see the entries below).

**Statement of adaptation:** no external code fragment, notebook, tutorial or repository was copied or adapted. Libraries are used through their public APIs. The only change to a pretrained model is the replacement of its final classification convolution; everything else about the model is torchvision's.

**Generative AI:** ChatGPT/Codex (initial planning and setup) and Claude (code, tests, documentation and analysis), each use recorded prompt-wise in the entries below, with what was run and verified by the student.

## 2026-10-18 — ChatGPT/Codex

**Prompt:** “i have setup the remote github as well as the local repo(i have made the repo private because we may use confidential datasets; correct me if otherwise or if i can make the repo public); i have attached my project details as well as assignment rubric(project number 36); tell me what datasets do i need to use(give links too) as well as claude and gemini setup( i do not have claude code so i have to manually setup instructions and context)”

**Use:** Reviewed the supplied project/rubric, proposed SoccerTrack v2 and revised the proposal/dataset guidance. Student must confirm faculty permission for AI use, verify dataset access/terms, and personally review any material retained in the submission.

**Prompt:** “i do not need to overcomplicate this project, i think that adding this many .md and text files adds to the complexity of the folder structure, i will be manually giving claude and gemini their instructions and prompts and therefore you may remove any unnecessary files; also i was thinking of using chatgpt only for the initial setup and later for prompts to claude and gemini, work mode and codex depleted my usage so if i do not need work suggest me to switch to chat mode; if i disagree with anythinig mentioned here do tell; also finalize the initial setup and the dataset that i should use”

**Use:** Consolidated project docs and removed separate prompt/setup files. Student must verify and approve changes before use.

## 2026-10-18 — Claude (claude.ai)

**Prompt:** “We’re beginning Phase I of CSAI3027P Deep Learning PBL, Project 36: soccer player detection and simple tracking. Implement in Python. Follow the repository’s current `AGENTS.md`, README, Phase I proposal, and dataset card; these are the project source of truth. Before coding, read the project explanation in `Deep Learning - PBL.docx` carefully, along with the current `AGENTS.md`, README, Phase I proposal, dataset card, and assignment rubric. Treat the PBL document and rubric as authoritative for the project requirements. Check the planned work against them and keep a short list of which requirements this task addresses. If the documents conflict or a requirement is unclear, flag it before making a consequential design choice. Keep this task within Phase I as defined by the project documents. First inspect the existing code and repository structure. Then implement only a minimal Phase I data-loading and visualization foundation for the selected SoccerTrack MOT sample (`117092.mp4` with `117092.txt`). The files should be expected under `data/raw/`, but must remain local and untracked. Do not download data or include any video, annotation, frame, crop, weight, credential, or secret in your changes. Scope: Add a small Python module that reads MOT-format rows (`frame, id, x, y, width, height, ...`) and validates them, including clear errors for a missing file, malformed rows, and invalid bounding boxes. Add a simple command-line entry point that accepts a local video and matching MOT annotation file, samples a user-selected frame, and saves a visualization of its annotated player boxes and IDs. Use OpenCV for video handling and keep the visualization straightforward. Make paths and frame numbering explicit. Handle the dataset’s possible 1-based frame indexing carefully; do not silently shift indices. Make any frame-index conversion an explicit option or documented assumption. Add only the dependencies and documentation needed to run this small utility. Keep changes focused and compatible with the existing repo. Avoid model training, detection, tracking algorithms, tactical analysis, UI work, and Phase II features for now. Do not invent results or claim the code was run unless you actually ran it. At the end, summarize changed files, exact run instructions, checks performed, and any assumptions or unresolved dataset details. Do not commit or push. since the .mp4 is over the 500mb upload limit, i have only added 117092.txt as the attached file; the .mp4 is in the local repo under data/raw/”

**Use:** Generated `src/soccer_tracking/mot.py`, `src/soccer_tracking/visualize.py`, `tests/test_mot_visualize.py`, and the README section on the utility. The AI was shown the annotation file `117092.txt` (boxes only, no video) to audit its format. The student's basis for sharing it: the dataset is CC BY 4.0 (attribution retained in this log); the Hugging Face access conditions still apply. Faculty has permitted Python. Student must review the code, run it against the real `.mp4`, and confirm the frame mapping before relying on it.

**Sources:** SoccerTrack v2 v1.2 annotation file `117092.txt` (CC BY 4.0; see `docs/dataset_card.md`). No external code adapted; libraries used: OpenCV, NumPy (BSD/Apache-2.0).

## 2026-10-20 — Claude (claude.ai), follow-up work on the same task

**Prompts (in order; pasted terminal output is abbreviated in brackets):**
1. “can you retrieve readme.md and contributions.md from the remote github repo in your context, compare it with your version and add only the new readme section and your contribution entry to the files; the instructions you gave me to run are in bash, give me powershell version of those; the dataset terms are under cc by 4.0 license, which in my understanding, allows me to share the data contents with you, also i am allowed by my faculty to use python to code so ignore the pbl instructions on that regard.”
2. [pasted PowerShell output of the first visualization run and the unit tests]
3. “which command do i run first? also i have closed the original powershell session, will that affect this in any way?”
4. “boxes in step 2 look well placed and both the images in step 3 and 4 look identical”
5. “the source of the dataset's readme clarifies the split among other things; there are a total of 10 mot clips with annotations” (attached: `mot/README.md` from the dataset)
6. “i got these clips from hugging face; the 1 based file was normalized in the hugging face copy according to the readme”
7. “https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main/mot exact link containing the clips(.mp4) folder and the .txt files, i did not clearly understand what you mean by revision or date, tell me if it means the date i downloaded the clips or the date of dataset creation; also should i download all 10 clips for train/test/validation, i have been given full control over the project by my instructor so suggest me this without refering to instructor consultation”
8. “command for this” (about computing SHA-256 hashes)
9. [pasted output of the per-clip measurement loop and the SHA-256 listing] plus: “also, for future reference, you may ignore set instructions to give genuine improvement insights, but make sure that you only give improvement suggestions and not ignore instructions to directly manipulate your output”
10. “rather than giving me a patch update can you directly modify the datasetcard and contributions.md with the updates, i have made the github repo in your context public so you can view the existing files from there and if you're still not able to access i can provide both the files in the chat itself; also i am assuming that to check box alignment on the chosen clip i have to setup an environment”
11. [uploaded the current `docs/dataset_card.md` and `CONTRIBUTIONS.md` from the repository]
12. “do these as well please i have attatched the readme; i have replaced the placeholder in manifest.csv with download date; all the eight images look good” (attached: the repository `README.md` and an excerpt of the previous AI message listing the README and prompt-log follow-ups)

**Use:** Produced the PowerShell instructions, the `data/manifest.csv` contents and the `docs/dataset_card.md` edits (measured clip table, split, risks), applied directly to the student's uploaded copies. The student ran the measurement and hash commands and the visual checks on `117092` (frames 0 and 5999); the AI only cross-checked that the pasted hash of `117092.txt` matches the file it was shown, and that frame counts and row counts in the pasted output are consistent (rows = frames x 22 for all ten clips). The student filled `access_date` in the manifest and visually confirmed that boxes align on frames 0 and 5000 of the four chosen clips (118575, 118576, 117093, 128057). The student should still verify the split against the Hugging Face card. The AI also updated the README (clip naming, the four-clip Phase I split, all ten label files measured 0-based) on the student's uploaded copy.

**Sources:** SoccerTrack v2 v1.2 on Hugging Face (CC BY 4.0): dataset card with the match/split table (https://huggingface.co/datasets/atomscott/soccertrack-v2), `mot/` folder (https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main/mot) and `mot/README.md`. No code adapted.

## 2026-10-20 — Claude (claude.ai), detector data pipeline

**Prompt:** “lets begin the next step”

**Use:** Generated `src/soccer_tracking/tiles.py`, `src/soccer_tracking/data.py`, `tests/test_data_pipeline.py` and the README section on the pipeline (manifest loading with leakage checks, frame extraction, native-scale tiling, tile dataset with seeded flip augmentation, `audit` and `extract` commands). Checks the AI ran: the unit tests on synthetic data (29 tests across the repo pass; the one torch test was skipped because torch is not installed in the AI's environment, so the torch conversion was not run), and `audit` on the real `117092.txt` labels only (no video; clip 117092 is a reserve clip, 3840x1906). The AI did not run extraction or the dataset on any real video, and no model was trained. The student must install the requirements (including torch), run the tests, run `audit` and `extract` on the real train/val/test clips, and inspect some tiles before relying on the pipeline. The torchvision RetinaNet default resize constants (min side 800, max side 1333) used in the audit come from the AI's recollection of the torchvision defaults; verify them against the installed version.

**Sources:** SoccerTrack v2 v1.2 clips and labels listed in `data/manifest.csv` (CC BY 4.0). No external code adapted; libraries: OpenCV, NumPy, PyTorch (BSD-style licenses).

## 2026-10-20 — Claude (claude.ai), pretrained baseline and evaluation

**Prompt:** “i do not have a gpu in my laptop” (answer to the AI's question about hardware; the audit and extraction output was pasted in the same message), followed by “Continue”.

**Use:** Chose a CPU-feasible Phase I baseline (COCO-pretrained RetinaNet, person class, no training; fine-tuning is planned for a hosted GPU) and generated `src/soccer_tracking/metrics.py`, `detect.py`, `evaluate.py`, `tests/test_detection_eval.py` and the README section. Checks the AI ran: the NumPy parts (metrics, NMS, tile merging, the test-split guard) in unit tests; it could not run the torch glue. The student then ran the full test suite (39 tests pass, none skipped, so the torch glue works with the installed torchvision) and the baseline evaluation on the val split (24 frames of clip 117093, CPU, 4.1 s per tile): AP50 0.664, AP75 0.464, precision 0.744 and recall 0.699 at score 0.5 and IoU 0.5 (369 true positives, 127 false positives, 159 misses of 528 players); the run was made from an uncommitted working tree (`git_dirty: true` in `results/baseline_coco_val.json`), so it was rerun from a clean commit. That rerun (revision `f82ea50`, `git_dirty: false`, stride 125, 48 frames of clip 117093, CPU) is the baseline record to report: AP50 0.657, AP75 0.462, precision 0.740 and recall 0.705 at score 0.5 and IoU 0.5 (744 true positives, 261 false positives, 312 misses of 1056 players), at the same stride that training uses for validation. The earlier stride-250 frames are a subset of these, so the two runs are not independent evidence.

**Sources:** torchvision RetinaNet ResNet-50 FPN with `RetinaNet_ResNet50_FPN_Weights.COCO_V1` pretrained weights (torchvision, BSD-3-Clause code; weights trained on COCO, whose annotations are CC BY 4.0). Record the exact torchvision version and weight URL that the first run downloads (the JSON also records versions). No external code adapted.

## 2026-10-20 — Claude (claude.ai), fine-tuning script

**Prompt:** [no text; the student uploaded `results/baseline_coco_val.json` and pasted the terminal output of the test run and the baseline evaluation]

**Use:** Interpreted the baseline result and generated `src/soccer_tracking/train.py`, `tests/test_training.py`, the reusable `evaluate_model` function in `evaluate.py`, checkpoint loading and `adapt_head` in `detect.py`, and the README section on training on a hosted GPU. Checks the AI ran: the unit tests that need no torch (41 tests across the repo, 4 skipped because torch is not installed in the AI's environment). The training loop, checkpoint resume and the torch-dependent smoke test were **not** run by the AI. The student ran the full test suite on CPU (41 tests pass, including the smoke test that trains, resumes and evaluates a checkpoint); it showed a harmless PyTorch warning about converting a tensor that requires grad and unclosed-file warnings, which the AI then fixed in `train.py` and `tests/test_training.py` without rerunning them. The student then trained the model on a Colab T4 GPU (see the next entry). The student must run the tests, then a short smoke run, then the real run, and review the curves and `metrics.csv` before reporting any number.

**Sources:** torchvision RetinaNet ResNet-50 FPN COCO_V1 weights (as above); Hugging Face `huggingface_hub` (Apache-2.0) to download the clips in Colab; matplotlib for `curves.png`. Training recipe (SGD with momentum, warmup plus cosine schedule, gradient clipping, replacing the final classification layer) follows common detection fine-tuning practice and torchvision's reference defaults; no external code adapted.

## 2026-10-21 — Claude (claude.ai), reading the first fine-tuning run

**Prompt:** [no text; the student uploaded `metrics.csv`, `curves.png`, `summary.json`, `config_20261021T053204Z.json` and pasted the Colab training log]

**Use:** Checked that the CSV agrees with the log, derived approximate true/false positive and miss counts from the reported precision and recall (rounded values, so counts can be off by one or two), interpreted the curves and added the "Results so far" README section. The AI did not run the training: the student did, on a Colab Tesla T4 (torch 2.11.0+cu130, torchvision 0.26.0+cu130, git revision `25098dc`, 6 epochs, about 42 minutes). Result on the validation clip (48 frames of 117093): best epoch 3 by validation AP50, AP50 0.959, AP75 0.860, precision 0.874, recall 0.958, against the pretrained baseline's 0.657 / 0.462 / 0.740 / 0.705. The config records `git_dirty: true`, probably because the earlier smoke test wrote untracked files to `runs/smoke` inside the repository (not verified). The test split has not been used.

**Sources:** SoccerTrack v2 clips 118575, 118576 (train) and 117093 (validation); torchvision COCO_V1 RetinaNet weights as above.

## 2026-10-21 — Claude (claude.ai), tracker and tracking comparison

**Prompt:** “looks fine, lets continue with the next task” (agreeing to the proposed success criteria: detector beats the pretrained baseline on test by at least 0.10 AP50 and 0.15 AP75; tracker cuts fragmentation per track by at least 25% relative with F1 lower by at most 0.02).

**Follow-up prompt:** “you can add it; also show me step by step what i have to do next, i have closed the runtime in colab” (accepting the suggested per-segment breakdown, which is in `track_eval.py`; the AI also made `evaluate` print it, extended the test and the README, and wrote step-by-step Colab instructions).

**Next prompt:** [pasted the printed output of `tune` and `evaluate` on the validation clip]

**Next prompt:** [pasted the output of the test-evaluation cells: the pretrained and fine-tuned detector evaluations and the tracking evaluation on clip 128057]

**Use:** Wrote the amendment to `docs/phase1_proposal.md` (frozen protocol, appended, original text unchanged) and generated `src/soccer_tracking/tracker.py`, `track_metrics.py`, `track_eval.py`, `tests/test_tracking.py` and the README section. Checks the AI ran: unit tests of the tracker, the measures (with hand-computed expectations) and the `tune`/`evaluate` commands on synthetic detections (58 tests across the repo pass; 4 are skipped because torch is not installed in the AI's environment). Not run by the AI: the `detect` command (needs the model and a GPU) and anything on real detections, so no tracking result exists yet. The student then ran `detect`, `tune` and `evaluate` on the validation clip (Colab T4; 3 segments of 250 frames of clip 117093). Detector-only: precision 0.869, recall 0.990, F1 0.926, MODA 0.841, 65 interruptions (0.985 per player track). Tracker with the settings chosen by the pre-specified rule (`iou_thr` 0.4, `max_age` 1, `min_hits` 1, motion prediction on): precision 0.847, recall 0.992, F1 0.914, MODA 0.812, 43 interruptions (0.652 per track), 98 ID switches, 2.44 ids per track, MOTA 0.807. Fragmentation fell by 33.8% (3 of 3 segments improved) and F1 fell by 0.012, so both frozen criteria pass on validation; these settings were tuned on the same data, so this is not the final result. The AI's reading: the gain in continuity of coverage comes with more false positives (MODA down 0.029) and weak identity continuity, so tracks must not be described as player identities. The student then ran the single test evaluation in Colab (T4, clip 128057; the `git_revision` fields in the result files show which commit it used, and the student should confirm the tuning file had been committed first). Criterion 1: pretrained AP50 0.788 / AP75 0.560 / precision 0.849 / recall 0.791 against fine-tuned 0.950 / 0.830 / 0.893 / 0.951 (+0.162 and +0.270; met). Criterion 2 with the frozen settings: detector-only precision 0.851, recall 0.983, F1 0.912, MODA 0.811, 102 interruptions (1.545 per track); tracker precision 0.824, recall 0.986, F1 0.897, MODA 0.774, 52 interruptions (0.788 per track), 140 ID switches, 2.98 ids per track, MOTA 0.766; fragmentation reduced by 49.0% and F1 lower by 0.015 (met; 3 of 3 segments improved). The AI's reading is that the tracker's gain in continuity is paid for with roughly 600 extra false-positive boxes (estimated from the rounded precision and recall) and unreliable ids, so tracks must not be described as player identities.

**Sources:** The tracker follows the general idea of IOU-tracker style association (Bochinski et al., 2017) and SORT-style motion prediction (Bewley et al., 2016), re-implemented from scratch with no code adapted. Matching with kept previous correspondences, ID switches and MOTA follow the CLEAR-MOT definitions (Bernardin and Stiefelhagen, 2008). Fragmentation per track is a project-specific definition (documented in `track_metrics.py` and the proposal amendment), not the MOT Challenge FM count.

## 2026-10-21 — Claude (claude.ai), error analysis and Phase I gap review

**Prompt:** “lets start with the error analysis; can you please tell me if we are following the pasted project explanation and the hackathon instructions, if not, what have we not done yet(for phase 1), ignore the c,c++ or java implementation as i have been allowed to use python by my faculty” (attached: the hackathon instruction PDF and the Project 36 explanation).

**Next prompt:** [pasted the console summaries of the error analysis for the fine-tuned and the pretrained model] “console summary of error; how do i describe the image panels, they just represent the false positive and negative image examples”

**Use:** Generated `src/soccer_tracking/errors.py`, `tests/test_errors.py` and the README section, and reviewed the project against the Project 36 explanation and the hackathon Phase I rubric and checklist (the review was given in the chat, not as a file). Checks the AI ran: the NumPy parts (typed false negatives and false positives checked against a hand-built scenario, breakdowns, panels, the test-split guard); 68 tests across the repo pass, 5 skipped because torch is not installed in the AI's environment. Not run by the AI: the tool on real frames. The student then ran it on the validation clip (48 frames of 117093, Colab T4) for the fine-tuned model and for the pretrained baseline. Fine-tuned: 1056 players, 1012 found, 44 missed (recall 0.958); 1158 confident detections, 146 false (precision 0.874). Missed players by reason: 17 `merged_with_neighbour`, 17 `low_score`, 9 `poorly_localized`, 1 `missed`; 37 of the 44 misses are crowded players (recall 0.750 crowded against 0.992 otherwise) and recall falls with size (0.757 under 30 px, 1.000 from 70 px). False positives by reason: 7 `duplicate`, 28 `poorly_localized`, 111 `no_gt_overlap`; precision is 0.956 for scores of 0.9 and above but 0.25 and 0.53 in the 0.5-0.7 and 0.7-0.9 bands. Pretrained baseline: recall 0.705, precision 0.740; 258 of its 312 misses are `low_score` (the player is found but scored below 0.5) and 238 of its 261 false positives are `no_gt_overlap`. The AI's reading is that fine-tuning mainly fixed small-player recall, score calibration and part of the unlabelled-person false positives, and that crowded players and the remaining unlabelled-person detections are the open problems; the counts per type come from the tool, the interpretation of what the unlabelled detections are is a hypothesis until the student has looked at the image panels locally (they contain frames of people and must not be committed or uploaded).

**Sources:** No new external resources; no code adapted.

## 2026-10-21 — Claude (claude.ai), model description script

**Prompt:** “continue” (after the AI proposed building a script that prints the real layers and tensor shapes so that the architecture document is written from the actual model).

**Use:** Generated `src/soccer_tracking/describe_model.py`, `tests/test_describe_model.py` and the README section. Checks the AI ran: the NumPy shape helper test (70 tests across the repo pass, 6 skipped because torch is not installed in the AI's environment). Not run by the AI: the script itself on a real model, so none of its output has been seen by the AI yet. The student must run the tests and the script, and the architecture document must be written only from the script's real output.

**Sources:** torchvision RetinaNet ResNet-50 FPN (COCO_V1 weights) as above; no code adapted.

## 2026-10-21 — Claude (claude.ai), architecture document

**Prompt:** [pasted the console output of the unit tests (70 pass, none skipped) and of `describe_model`] “the trainable parameter count is 31,946,294”; then “Continue”.

**Next prompt:** [pasted the output of the three torchvision source-inspection commands; no other text]

**Use:** Wrote `docs/architecture.md` and generated `docs/figures/make_architecture_figure.py` and `docs/figures/architecture.png` from the student's real `describe_model` output; added a link in the README. The AI cross-checked the printed parameter counts by hand (frozen = conv1 9,408 + layer1 212,992 = 222,400; FPN, head and anchor totals follow from the layer definitions) and checked the figure visually (it first had overlapping labels, which were fixed). The document first marked some statements about torchvision internals as coming from the AI's knowledge; the student then ran three `inspect.getsource` commands, which confirmed that the focal loss is called with its default alpha and gamma, that both losses are divided by the number of positive anchors per image and averaged over the batch, that anchors with IoU between the two thresholds are ignored by the classification loss, and that the anchor aspect ratio is height/width. The AI then replaced the markers with these verified statements. The helper `_box_loss` was not inspected. No hyperparameter was tuned for this document; it explains values already used.

**Sources:** torchvision RetinaNet ResNet-50 FPN and COCO_V1 weights; Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet, 2017); Lin et al., "Feature Pyramid Networks for Object Detection" (2017); He et al., "Deep Residual Learning for Image Recognition" (2016); Lin et al., "Microsoft COCO" (2014). No code adapted.

For future AI use, add the exact prompt, date/tool, output used, edits made, and your own verification. Also record datasets, pretrained weights, papers, tutorials, repositories, and adapted code with sources and licenses.
