"""Football Tracker — Offline Football Player Tracking & Tactical Analyzer (Streamlit UI).

Launch locally (Windows PowerShell)::

    pip install streamlit imageio-ffmpeg pandas
    $env:PYTHONPATH = "src"
    streamlit run src/soccer_tracking/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src/ is importable even if launched without PYTHONPATH
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from soccer_tracking.product import (  # noqa: E402
    DISCLAIMER_TEXT,
    InputValidationError,
    run_cached_analysis,
    run_live_analysis,
)
from soccer_tracking.score_final_test import FROZEN_FINAL_PARAMS  # noqa: E402

st.set_page_config(
    page_title="Football Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; max-width: 1260px; }
    .ft-title {
        font-size: 2.05rem; font-weight: 750; letter-spacing: -0.02em; margin: 0 0 0.2rem 0;
        background: linear-gradient(90deg, #10b981 0%, #34d399 55%, #e5e7eb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .ft-sub { font-size: 0.95rem; color: #9ca3af; margin: 0 0 1.25rem 0; }
    .ft-section-label {
        font-size: 0.82rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        color: #9ca3af; margin: 1.6rem 0 0.5rem 0;
    }
    .ft-footer-note {
        margin-top: 2rem; padding: 0.65rem 0.9rem; border-radius: 6px;
        background: rgba(255, 255, 255, 0.025); border-left: 3px solid #10b981;
        font-size: 0.82rem; color: #9ca3af;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px; padding: 0.85rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    '<h1 class="ft-title">Football Tracker</h1>'
    '<p class="ft-sub">Offline Football Player Tracking &amp; Tactical Analyzer</p>',
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Analysis Setup")
    mode = st.radio(
        "Execution Mode",
        ["Cached Demo (Instant · CPU)", "Live Video Segment (GPU)"],
        index=0,
        help="Cached Demo re-tracks saved detections in seconds on CPU. Live mode runs RetinaNet + OSNet.",
    )
    is_cached = mode.startswith("Cached")

    if is_cached:
        video_path = st.text_input("Match Video Path", value="data/raw/117093.mp4")
        det_dir = st.text_input("Cached Detections Folder", value="data/processed/detections_p2/p2more_full")
        clip_id = st.text_input("Clip ID", value="117093")
        c1, c2 = st.columns(2)
        start_frame = int(c1.number_input("Start Frame", min_value=0, value=4000, step=50))
        seg_len = int(c2.number_input("Frames", min_value=25, max_value=600, value=150, step=25))
        end_frame = start_frame + seg_len
        uploaded_video = None
        weights_path = ""
        reid_path = ""
    else:
        uploaded_video = st.file_uploader("Upload Fixed-Camera Football Clip", type=["mp4", "mov", "avi", "mkv"])
        weights_path = st.text_input("Detector Weights (.pt)", value="runs/p2_more_data/best.pt")
        reid_path = st.text_input("Soccer OSNet Weights (optional)", value="runs/reid_osnet/best_osnet_soccer.pt")
        c1, c2 = st.columns(2)
        start_frame = int(c1.number_input("Start Frame", min_value=0, value=0, step=25))
        seg_len = int(c2.number_input("Frames (max 250)", min_value=25, max_value=250, value=125, step=25))
        end_frame = start_frame + seg_len
        video_path, det_dir, clip_id = "", "", ""

    with st.expander("Tracker & Threshold Settings", expanded=False):
        det_high = st.slider("High-Confidence Threshold (det_high)", 0.50, 0.95, float(FROZEN_FINAL_PARAMS["det_high"]), 0.02)
        det_low = st.slider("Continuation Threshold (det_low)", 0.15, 0.70, float(FROZEN_FINAL_PARAMS["det_low"]), 0.05)
        merge_cos = st.slider("GTA-Link Cosine Threshold", 0.10, 0.45, float(FROZEN_FINAL_PARAMS["merge_cos_thresh"]), 0.05)
        use_gta = st.toggle("Enable Offline GTA-Link Stitching", value=True)

    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

# --- Execute Analysis ---
if run_clicked:
    overrides = {
        "det_high": float(det_high),
        "det_low": float(det_low),
        "merge_cos_thresh": float(merge_cos),
        "use_gta": bool(use_gta),
    }
    try:
        out_dir = Path(tempfile.mkdtemp(prefix="football_tracker_"))
        with st.spinner("Running football tracking pipeline and rendering replay..."):
            if is_cached:
                res = run_cached_analysis(
                    video_path=Path(video_path.strip()),
                    det_dir=Path(det_dir.strip()),
                    clip_id=clip_id.strip(),
                    out_dir=out_dir,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    tracker_overrides=overrides,
                )
            else:
                if uploaded_video is None:
                    raise InputValidationError("Please upload a fixed-camera football video clip first.")
                tmp_vid = out_dir / uploaded_video.name
                tmp_vid.write_bytes(uploaded_video.getbuffer())
                res = run_live_analysis(
                    video_path=tmp_vid,
                    weights_path=Path(weights_path.strip()),
                    reid_weights=Path(reid_path.strip()) if reid_path.strip() else None,
                    out_dir=out_dir,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    tracker_overrides=overrides,
                )
        st.session_state["ft_result"] = res
        st.toast(f"Analysis finished in {res['runtime_seconds']}s")
    except (InputValidationError, FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(f"Input / Validation Error: {exc}")

# --- Render Results ---
res = st.session_state.get("ft_result")
if res is None:
    st.info("Configure your clip in the sidebar and click **Run Analysis** to generate the annotated replay, pitch heat map, and data exports.")
else:
    stats = res["stats"]
    tiers = stats["track_confidence_counts"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique Tracks", stats["unique_tracks"])
    m2.metric("Avg Players / Frame", f"{stats['mean_tracks_per_frame']:.1f}")
    m3.metric("High-Confidence Tracks", f"{tiers['high']} / {stats['unique_tracks']}")
    m4.metric("Runtime", f"{res['runtime_seconds']:.1f}s")

    # Main visual attraction: Annotated Replay & Pitch Heat Map
    tab_video, tab_heat = st.tabs(["Annotated Replay", "Pitch Heat Map"])

    with tab_video:
        vid_path = Path(res["outputs"]["video"])
        if vid_path.is_file():
            vid_bytes = vid_path.read_bytes()
            st.video(vid_bytes, format="video/mp4")
            st.download_button(
                "Download Annotated MP4",
                data=vid_bytes,
                file_name=f"football_tracker_{res.get('clip_id', 'segment')}.mp4",
                mime="video/mp4",
            )
        prev_path = Path(res["outputs"].get("preview", ""))
        if prev_path.is_file():
            with st.expander("Mid-Segment Annotated Keyframe", expanded=False):
                st.image(str(prev_path), use_container_width=True)

    with tab_heat:
        hm_path = Path(res["outputs"]["heatmap"])
        if hm_path.is_file():
            st.image(str(hm_path), use_container_width=True)
            st.download_button(
                "Download Heat Map PNG",
                data=hm_path.read_bytes(),
                file_name=f"football_tracker_{res.get('clip_id', 'segment')}_heatmap.png",
                mime="image/png",
            )

    # Secondary data & diagnostics sections below the main visuals
    st.markdown('<div class="ft-section-label">Data &amp; Run Diagnostics</div>', unsafe_allow_html=True)

    csv_path = Path(res["outputs"]["csv"])
    with st.expander("Trajectories & Confidence Flags", expanded=False):
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            c_left, c_right = st.columns([3, 1])
            c_left.caption(
                f"Confidence tiers — High: {tiers['high']}, Medium: {tiers['medium']}, Low: {tiers['low']} | "
                f"Crowded events: {stats['crowded_box_events']} | Low-score continuations: {stats['low_score_continuation_events']}"
            )
            c_right.download_button(
                "Download CSV",
                data=csv_path.read_bytes(),
                file_name=f"football_tracker_{res.get('clip_id', 'segment')}_trajectories.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.dataframe(df.head(500), use_container_width=True, height=340)

    sum_path = Path(res["outputs"]["summary"])
    with st.expander("Run Summary (JSON)", expanded=False):
        st.download_button(
            "Download Summary JSON",
            data=sum_path.read_bytes() if sum_path.is_file() else json.dumps(res, indent=2).encode("utf-8"),
            file_name=f"football_tracker_{res.get('clip_id', 'segment')}_summary.json",
            mime="application/json",
        )
        st.json(res)

# --- Small Footer Notice ---
st.markdown(
    f'<div class="ft-footer-note"><strong>Scope &amp; Confidence Notice:</strong> {DISCLAIMER_TEXT}</div>',
    unsafe_allow_html=True,
)