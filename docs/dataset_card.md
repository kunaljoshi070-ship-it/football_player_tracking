# Dataset card — SoccerNet Tracking (candidate)

**Decision:** Preferred candidate, but conditional. Do not begin a video-based experiment until the team has confirmed access, terms, annotation/video alignment, and suitability for the course.

## What it contains

The official SoccerNet tracking task describes 100 main-camera clips of 30 seconds at 1080p across its 2022/2023 challenge data, with tracking annotations, and distinguishes association using ground-truth detections from full tracking using raw video. The development-kit README also documents downloadable tracking CSVs and a larger collection. Confirm the exact release/split and counts against the chosen kit version before freezing the sample.

## Access status checked 2026-10-18

- Official tracking annotations/dev-kit data are described as downloadable through the `SoccerNet` Python package and its task downloader. The older tracking README gives `downloadDataTask(task="tracking", split=[...])` and `task="tracking-2023"` examples.
- The official SoccerNet Hugging Face organization currently lists `SoccerNet-Tracking-RAW-Video` as a **gated** repository. The official SoccerNet site says game videos are copyrighted and require an NDA to prevent redistribution. The official organization describes an account request, NDA acknowledgement, and manual approval flow for gated content.
- Therefore SoccerNet is not an unconditional primary video dataset. If access is unavailable or approval takes too long, use only an instructor-approved, legally accessible alternative with compatible annotations; do not replace the video with an unlicensed broadcast download.
- Verify whether the relevant annotation repository has separate access conditions and license before using it. Keep data files and credentials local; record only permitted metadata in Git.

## Access workflow

1. Visit the official SoccerNet tracking task page and development kit linked below.
2. Identify the exact tracking edition/split needed for the proposed sample.
3. Request access to the relevant tracking raw-video repository on Hugging Face; complete its current terms/NDA and wait for approval if requested.
4. Obtain annotations through the official downloader or approved repository and video through the currently authorized official route.
5. Verify file correspondence, frame rate/resolution, annotation format, and permitted academic use. Record date, version, split, clip identifiers, and checksums in the local manifest.
6. Never commit or share gated video, redistributed copies, access tokens, or model weights derived from restricted assets.

## Proposed sample protocol

After access is confirmed, select a small fixed sample (initial target: 3–5 clips, 30 seconds each) spanning ordinary play and at least one moderately crowded/occluded sequence. Freeze clip IDs before evaluating. Sample at a documented fixed rate for computational feasibility; preserve the official frame mapping. Use match/sequence-separated splits if sufficient clips exist. This is a proposal, not a claim that the selected clips are already accessible.

## Fields and limitations

The official kit describes MOT-style CSVs with frame ID, track ID, bounding-box coordinates, dimensions, confidence, and placeholder fields. Tracking classes include players, goalkeepers, referees, staff, and ball; the benchmark may not evaluate object class. Decide whether this project evaluates all annotated objects or filters to player categories, and document that choice. Broadcast camera movement, occlusion, scale changes, and visually similar uniforms limit image-plane tactical interpretation.

## License and citation

Video is copyrighted by the relevant rights holders; SoccerNet says the data is research-focused and not for commercial use. A repository's code license does not grant a license to the videos. Review the current terms/NDA and annotation repository license directly before use. Cite the dataset paper:

Cioppa et al., “SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos,” CVPR Workshops 2022, pp. 3491–3502.

## Official references

- [SoccerNet Tracking task](https://www.soccer-net.org/tasks/tracking)
- [Official SoccerNet tracking development kit](https://github.com/SoccerNet/sn-tracking)
- [Official SoccerNet FAQ](https://www.soccer-net.org/faq)
- [Official SoccerNet Hugging Face organization and dataset access guide](https://huggingface.co/SoccerNet)
- [SoccerNet-Tracking paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Cioppa_SoccerNet-Tracking_Multiple_Object_Tracking_Dataset_and_Benchmark_in_Soccer_Videos_CVPRW_2022_paper.pdf)
