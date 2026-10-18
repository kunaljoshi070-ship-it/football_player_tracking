# Project-wide instructions

## Scope and source of truth
- This is CSAI3027P Deep Learning PBL Project 36: football/soccer player tracking and basic tactical analysis.
- GitHub is the shared source of truth for ChatGPT, Claude (primary implementation), Gemini (frontend/visualization), and the independent reviewer.
- Implement in Python. Prefer PyTorch for learned models and OpenCV for video handling. Do not revive the earlier C/C++/Java assumption.
- Work on Phase I only until the team and instructor approve moving to Phase II. Do not implement Phase II requirements in this phase.
- The authoritative files are `Deep Learning - PBL.docx` and `CSAI3027P_Two_Phase_Hackathon_Instructions_for_Students.pdf`. They were not available in the initial workspace; when added, read them and update the rubric map and proposal before claiming rubric alignment.

## Phase I boundaries
- Start with a small, fixed, documented subset of soccer clips.
- Establish a person/player detection baseline. Add tracking and trajectory visualization only if compatible annotations and permitted video are available.
- Keep tactical analysis descriptive and limited to what trajectories support; do not claim full tactical understanding or player identity.
- Include an explicit testable hypothesis, baseline comparison, controlled experiment, reproducibility details, and measured results.
- Explain neural-network fundamentals and model breadth at the level required by the rubric, while labeling each deliverable Phase I or Phase II in `docs/rubric_map.md`.

## Data and rights
- Never commit raw video, datasets, model weights, access tokens, or other secrets.
- Store only manifests, scripts, metadata that may legally be shared, and instructions in Git.
- Check the current dataset terms and access status before downloading or sharing anything. SoccerNet broadcast video is copyrighted and access-gated; do not redistribute it. Treat any annotations separately and verify their repository-specific terms.
- Record dataset version, split, clip/frame selection, source URL, access date, checksums where appropriate, and citation in the manifest and experiment log.

## Collaboration rules
- Use focused branches and pull requests; keep changes reviewable and document who/which tool contributed in `CONTRIBUTIONS.md`.
- Claude may implement approved tasks; Gemini may propose frontend/visualization changes; ChatGPT may plan, write prompts, and review; the separate reviewer should inspect diffs and evidence independently.
- AI-generated changes require human review. Never paste secrets or gated/copyrighted video into external AI services.
- Keep experiments reproducible: fixed seeds where applicable, recorded environment/configuration, exact commands, and saved metrics/plots without committing prohibited assets.
- Before coding, read this file and the relevant proposal, dataset card, and experiment plan. Do not invent results or claim tests were run when they were not.
