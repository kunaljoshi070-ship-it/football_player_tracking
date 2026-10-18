# CSAI3027P Deep Learning PBL — Project 36

**Working title:** Soccer Player Detection, Tracking, and Trajectory-Based Analysis  
**Language:** Python  
**Current stage:** Phase I setup and proposal; Phase II is out of scope for now.

This repository is the shared project record for planning, code, experiments, review, and handoffs across ChatGPT, Claude, Gemini, and the independent reviewer. Keep data, video, model weights, and secrets out of Git.

## Phase I goal

Build a feasible baseline that detects people/players in a small, fixed set of soccer video clips. If permitted video and compatible tracking annotations are available, add a simple tracker and visualize image-plane trajectories. Test whether a lightweight temporal association/smoothing step improves trajectory continuity over frame-by-frame detections on the selected clips. Report detection and tracking metrics separately and state the limitations of image-plane tactical summaries.

The experiment plan defines the measurable hypothesis and the proposal records current assumptions. Rubric alignment is provisional until the two authoritative course files are added and mapped.

## Start here

1. Read [`AGENTS.md`](AGENTS.md), [`docs/phase1_proposal.md`](docs/phase1_proposal.md), [`docs/dataset_card.md`](docs/dataset_card.md), and [`docs/experiment_plan.md`](docs/experiment_plan.md).
2. Check [`docs/rubric_map.md`](docs/rubric_map.md) against the course DOCX and PDF; add them locally or to the team's approved private course-material location, not blindly to a public repository.
3. Review [`data/README.md`](data/README.md) and [`data/manifest.csv`](data/manifest.csv) before obtaining any data.
4. Create a Python environment and install `requirements.txt`. Record the Python/platform and library versions before experiments.
5. Keep implementation in `src/`, exploratory work in `notebooks/`, configuration in `configs/`, and metrics/plots in `results/`.

## Data access and ethics

SoccerNet is the preferred candidate, subject to confirming that the course team can obtain the relevant tracking video and that the intended use complies with the current terms. The official tracking kit documents clips and annotation files; the official SoccerNet pages currently list raw tracking video as gated. Do not assume that public annotations imply public video access. See the dataset card for the verified status, access steps, citations, and fallback decision.

## Repository layout

```text
AGENTS.md
README.md
CONTRIBUTIONS.md
requirements.txt
configs/phase1.yaml
data/README.md
data/manifest.csv
docs/{phase1_proposal,dataset_card,experiment_plan,rubric_map}.md
notebooks/README.md
prompts/{chatgpt,claude,gemini,reviewer}.md
results/README.md
src/soccer_tracking/
tests/                  # reserved for Phase I unit checks
```

## Team workflow

- Push design decisions and results to GitHub so all four AI roles and teammates work from the same version.
- Use pull requests for code or document changes. Attach experiment configuration and evidence to result updates.
- Use the role prompts in `prompts/`; adapt each task to a small, bounded issue.
- Log meaningful contributions in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
