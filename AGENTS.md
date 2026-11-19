# Project instructions

- Project: CSAI3027P Deep Learning PBL, Project 36, soccer player tracking and tactical event analysis.
- Use Python as approved by the professor; keep the approval available because the generic project compendium says C/C++/Java.
- Phase I is finished and frozen: its results, split (train 118575/118576, val 117093, test 128057) and records in `results/` and `runs/finetune/` must not be edited or overwritten. Work now is **Phase II**: beat the published GTATrack baseline (ACM MMSports '25) on SoccerTrack. The plan, targets, protocol, gates and the current status board are in `docs/phase2_plan.md`; read it before starting and update its status board when you finish.
- GitHub is the source of truth. AI chats are not synced; pass relevant context manually and review every change.
- Never commit raw video, dataset files, person crops, error-analysis image panels, model weights, credentials, or secrets. Do not upload restricted footage/data to AI services.
- Generative AI is allowed only if faculty permits it. If permitted, record every prompt and use in `CONTRIBUTIONS.md`; understand and verify all submitted work yourself.
- Keep results reproducible: record split, seed, preprocessing, model and weight source, training settings, metrics, hardware, versions and command. Never invent results or say a check ran if it did not.
- Evaluation discipline: choose settings on validation clips only; write the final configuration into a dated amendment before the first test run; run the test clips once, only through commands that require `--allow-test`; keep the reserve clips (118578, 132877) unused until the live unseen test.
- End every working session with a handoff: work done, files created or edited, next steps with exact commands, open decisions and blockers, notes for the next model (format in `docs/phase2_plan.md`, section 8).
