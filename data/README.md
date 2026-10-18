# Data access and handling

This directory is for a small manifest and documentation only. Put downloaded datasets, video, extracted frames, caches, and model weights in an ignored local path such as `data/raw/` or outside the repository. Never add gated/copyrighted video or secrets to Git or external AI tools.

For the current SoccerNet Tracking access assessment and official references, read [`../docs/dataset_card.md`](../docs/dataset_card.md). The primary candidate is conditional on authorization for raw video and confirmation of annotation terms. Publicly available code or annotation files do not by themselves establish permission to use or redistribute broadcast footage.

## Download outline

Use the official SoccerNet development kit and follow its current access route. Its README documents the `SoccerNet` Python downloader for tracking annotations. Raw single-camera tracking video is currently listed separately as gated; request approval and follow its NDA/terms. Do not hard-code passwords or tokens. Keep them in your local environment or credential manager, never in repository files or AI prompts.

## Manifest fields

Record dataset edition, split, clip ID, source URI, access date, local relative path (ignored), frame rate/size, annotation file and license/terms URL, checksum, and notes on selection. Do not record signed NDA contents or credentials.
