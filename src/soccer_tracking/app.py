"""Thin Gradio UI for the SoccerTrack Offline Tactical Analyzer (Phase II, WS-E).

Launch in Colab::

    !pip install -q gradio
    !cd /content/repo && PYTHONPATH=src:/content/deep-person-reid python3 -m soccer_tracking.app --share
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .product import DISCLAIMER_TEXT, InputValidationError, run_cached_analysis, run_live_analysis
from .score_final_test import FROZEN_FINAL_PARAMS


def build_demo():
    import gradio as gr

    def _run_cached(video_path, det_dir, clip_id, start, end, det_high, det_low, merge_cos, use_gta):
        try:
            out_dir = Path(tempfile.mkdtemp(prefix="soccer_cached_"))
            res = run_cached_analysis(
                video_path=Path(video_path.strip()),
                det_dir=Path(det_dir.strip()),
                clip_id=clip_id.strip(),
                out_dir=out_dir,
                start_frame=int(start),
                end_frame=int(end),
                tracker_overrides={
                    "det_high": float(det_high),
                    "det_low": float(det_low),
                    "merge_cos_thresh": float(merge_cos),
                    "use_gta": bool(use_gta),
                },
            )
            return (
                res["outputs"]["video"],
                res["outputs"]["heatmap"],
                res["outputs"]["csv"],
                json.dumps(res["stats"], indent=2),
                f"Done in {res['runtime_seconds']}s ({res['stats']['unique_tracks']} tracks).",
            )
        except (InputValidationError, FileNotFoundError, ValueError, RuntimeError) as exc:
            return None, None, None, "{}", f"Validation / Input Error: {exc}"

    def _run_live(video_file, weights_path, reid_path, start, end, det_high, det_low, merge_cos, use_gta):
        if not video_file:
            return None, None, None, "{}", "Validation Error: Please upload a fixed-camera video clip."
        try:
            out_dir = Path(tempfile.mkdtemp(prefix="soccer_live_"))
            res = run_live_analysis(
                video_path=Path(video_file),
                weights_path=Path(weights_path.strip()),
                reid_weights=Path(reid_path.strip()) if reid_path and reid_path.strip() else None,
                out_dir=out_dir,
                start_frame=int(start),
                end_frame=int(end),
                tracker_overrides={
                    "det_high": float(det_high),
                    "det_low": float(det_low),
                    "merge_cos_thresh": float(merge_cos),
                    "use_gta": bool(use_gta),
                },
            )
            return (
                res["outputs"]["video"],
                res["outputs"]["heatmap"],
                res["outputs"]["csv"],
                json.dumps(res["stats"], indent=2),
                f"Done in {res['runtime_seconds']}s on {res['device']}.",
            )
        except (InputValidationError, FileNotFoundError, ValueError, RuntimeError) as exc:
            return None, None, None, "{}", f"Validation / Input Error: {exc}"

    with gr.Blocks(title="SoccerTrack Offline Tactical Analyzer (Phase II)") as demo:
        gr.Markdown(
            f"# SoccerTrack Offline Tactical Analyzer (Phase II)\n"
            f"> **Interpretation Notice:** {DISCLAIMER_TEXT}"
        )

        with gr.Tab("Cached Demo Mode (Instant — CPU)"):
            with gr.Row():
                c_video = gr.Textbox(label="Video Path", value="data/raw/117093.mp4")
                c_det = gr.Textbox(label="Cached Detection Run Dir", value="/content/drive/MyDrive/soccer_runs/detections_p2/p2more_full")
                c_clip = gr.Textbox(label="Clip ID", value="117093")
            with gr.Row():
                c_start = gr.Number(label="Start Frame", value=4000, precision=0)
                c_end = gr.Number(label="End Frame (max 600 frames)", value=4200, precision=0)
                c_dh = gr.Slider(0.50, 0.95, value=float(FROZEN_FINAL_PARAMS["det_high"]), step=0.02, label="det_high")
                c_dl = gr.Slider(0.10, 0.70, value=float(FROZEN_FINAL_PARAMS["det_low"]), step=0.05, label="det_low")
                c_cos = gr.Slider(0.10, 0.50, value=float(FROZEN_FINAL_PARAMS["merge_cos_thresh"]), step=0.05, label="GTA merge_cos")
                c_gta = gr.Checkbox(value=True, label="Enable GTA-Link")
            c_btn = gr.Button("Run Instant Analysis", variant="primary")
            c_status = gr.Textbox(label="Status")
            with gr.Row():
                c_vid_out = gr.Video(label="Annotated Video")
                c_hm_out = gr.Image(label="Pitch Occupancy Heat Map")
            with gr.Row():
                c_csv_out = gr.File(label="Download Trajectories CSV")
                c_json_out = gr.Code(label="Summary Stats (JSON)", language="json")
            c_btn.click(_run_cached, [c_video, c_det, c_clip, c_start, c_end, c_dh, c_dl, c_cos, c_gta],
                        [c_vid_out, c_hm_out, c_csv_out, c_json_out, c_status])

        with gr.Tab("Live Video Segment Mode (GPU Recommended)"):
            l_video = gr.Video(label="Upload Fixed-Camera Soccer Clip (.mp4/.mov/.avi/.mkv)")
            with gr.Row():
                l_weights = gr.Textbox(label="Detector Weights (P2-more best.pt)", value="/content/drive/MyDrive/soccer_runs/p2_more_data/best.pt")
                l_reid = gr.Textbox(label="Fine-Tuned OSNet Weights (optional)", value="/content/drive/MyDrive/soccer_runs/reid_osnet/best_osnet_soccer.pt")
            with gr.Row():
                l_start = gr.Number(label="Start Frame", value=0, precision=0)
                l_end = gr.Number(label="End Frame (max 250 frames)", value=150, precision=0)
                l_dh = gr.Slider(0.50, 0.95, value=float(FROZEN_FINAL_PARAMS["det_high"]), step=0.02, label="det_high")
                l_dl = gr.Slider(0.10, 0.70, value=float(FROZEN_FINAL_PARAMS["det_low"]), step=0.05, label="det_low")
                l_cos = gr.Slider(0.10, 0.50, value=float(FROZEN_FINAL_PARAMS["merge_cos_thresh"]), step=0.05, label="GTA merge_cos")
                l_gta = gr.Checkbox(value=True, label="Enable GTA-Link")
            l_btn = gr.Button("Run Live Segment Analysis", variant="primary")
            l_status = gr.Textbox(label="Status")
            with gr.Row():
                l_vid_out = gr.Video(label="Annotated Video")
                l_hm_out = gr.Image(label="Pitch Occupancy Heat Map")
            with gr.Row():
                l_csv_out = gr.File(label="Download Trajectories CSV")
                l_json_out = gr.Code(label="Summary Stats (JSON)", language="json")
            l_btn.click(_run_live, [l_video, l_weights, l_reid, l_start, l_end, l_dh, l_dl, l_cos, l_gta],
                        [l_vid_out, l_hm_out, l_csv_out, l_json_out, l_status])

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Launch the Gradio MVP for SoccerTrack Phase II.")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link (recommended in Colab)")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    demo = build_demo()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
