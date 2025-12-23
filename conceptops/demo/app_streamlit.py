"""
Streamlit demo app for ConceptOps (Phase 4).

High-level UX:
- Choose segmentation backend preference: sam3 (preferred) or dummy
- Run pipeline end-to-end using the Phase 3 canonical entrypoint:
  conceptops/pipelines/integrated_pipeline.py::process_video_to_dataset(...)
- Visualize outputs:
  - segmentation overlay playback
  - event timeline from pred_events.json (hero feature)
  - event slice viewer: select an event -> jump to its frames
- Export + download:
  - "Download episode.zip" (always works)
  - Export buttons call existing exporters if available

Constraints:
- Do NOT refactor core pipeline
- Do NOT add new ML models
- Do NOT integrate Ego2Robot in Phase 4
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

import streamlit as st

from conceptops.demo.demo_runner import DemoBackendChoice, DemoRunResult, run_conceptops_pipeline_for_demo
from conceptops.demo.rendering import (
    build_event_timeline_plotly,
    load_pred_events,
    overlay_frame_with_mask,
    render_event_slice_gallery,
    safe_read_json,
    zip_directory_to_bytes,
)

st.set_page_config(page_title="ConceptOps Demo", page_icon="🎛️", layout="wide")
st.title("🎛️ ConceptOps Demo")
st.caption("Upload → pipeline → segmentation + event timeline → slice viewer → export")

RUN_KEY = "conceptops_demo_run"
ERROR_KEY = "conceptops_demo_error"
UPLOAD_PATH_KEY = "conceptops_demo_upload_path"
EXPORT_PATHS_KEY = "conceptops_demo_export_paths"


def _reset_run_state() -> None:
    st.session_state.pop(RUN_KEY, None)
    st.session_state.pop(ERROR_KEY, None)
    st.session_state.pop(EXPORT_PATHS_KEY, None)


def _persist_uploaded_file(uploaded) -> Path:
    cache_dir = Path(".conceptops_demo_cache/uploads").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    src = Path(uploaded.name)
    out_path = cache_dir / f"{src.stem}__{uuid4().hex[:8]}{src.suffix}"
    out_path.write_bytes(uploaded.getvalue())
    st.session_state[UPLOAD_PATH_KEY] = str(out_path)
    return out_path


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "mkv", "avi", "mpeg4"],
        on_change=_reset_run_state,
    )

    st.divider()
    st.header("Segmentation Backend")
    backend_label = st.radio(
        "Backend preference",
        ["sam3 (GPU env)", "dummy (local CPU)"],
        index=1,
        on_change=_reset_run_state,
        help="Dummy is a local preview. SAM-3 is the real demo; run Streamlit on a GPU machine for that.",
    )
    backend_choice = DemoBackendChoice.SAM3 if backend_label.startswith("sam3") else DemoBackendChoice.DUMMY

    st.divider()
    st.header("Event Detector")
    event_mode = st.radio(
        "Event mode",
        ["model", "motion", "motion_mask", "fixed"],
        index=0,
        on_change=_reset_run_state,
    )

    e2r_config = {"mode": event_mode}

    # ---- Model tuning: make the local demo produce >1 event more often ----
    if event_mode == "model":
        st.caption("Model mode (tune to avoid single-event collapse)")
        # These defaults are more permissive than before to increase event count.
        inference_profile = st.text_input("inference_profile", value="demo_clean_v2")
        window_size = st.number_input("window_size", min_value=1, value=8, step=1)
        stride = st.number_input("stride", min_value=1, value=4, step=1)

        topk = st.number_input("topk (proposals)", min_value=1, value=50, step=5)
        min_score = st.number_input("min_score", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        nms_iou = st.number_input("nms_iou", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

        # Optional: point at artifacts
        model_name = st.text_input("model_name (optional)", value="event_model_demo3_wt")
        model_dir = st.text_input("model_dir (optional override)", value="")

        if model_dir.strip():
            e2r_config["model_dir"] = model_dir.strip()
        elif model_name.strip():
            e2r_config["model_name"] = model_name.strip()

        e2r_config.update(
            {
                "inference_profile": inference_profile,
                "window_size": int(window_size),
                "stride": int(stride),
                "topk": int(topk),
                "min_score": float(min_score),
                "nms_iou": float(nms_iou),
            }
        )

        st.caption("Tip: if you still get 1 event, try min_score=0.05 and topk=100.")

    elif event_mode in ("motion", "motion_mask"):
        threshold_multiplier = st.number_input("threshold_multiplier", min_value=0.1, value=1.5, step=0.1)
        min_event_length = st.number_input("min_event_length", min_value=1, value=2, step=1)
        base_label = st.text_input("base_label", value="move")
        min_area_ratio = st.number_input("min_area_ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        e2r_config.update(
            {
                "threshold_multiplier": float(threshold_multiplier),
                "min_event_length": int(min_event_length),
                "base_label": base_label,
                "min_area_ratio": float(min_area_ratio),
            }
        )

    else:
        frames_per_event = st.number_input("frames_per_event", min_value=1, value=8, step=1)
        base_label = st.text_input("base_label", value="segment")
        e2r_config.update({"frames_per_event": int(frames_per_event), "base_label": base_label})

    st.divider()
    run_btn = st.button("▶ Run pipeline")


# ---------------- Upload resolution ----------------
video_path: Optional[Path] = None
input_description = ""

if uploaded is not None:
    video_path = _persist_uploaded_file(uploaded)
    input_description = f"Uploaded: {Path(uploaded.name).name}"
else:
    prev = st.session_state.get(UPLOAD_PATH_KEY)
    if prev and Path(prev).exists():
        video_path = Path(prev)
        input_description = f"Uploaded (cached): {video_path.name}"


# ---------------- Run ----------------
if run_btn:
    if video_path is None or not video_path.exists():
        st.warning("Upload a video first.")
    else:
        with st.spinner("Running pipeline…"):
            try:
                run: DemoRunResult = run_conceptops_pipeline_for_demo(
                    video_path=video_path,
                    backend_choice=backend_choice,
                    input_description=input_description or video_path.name,
                    e2r_config=e2r_config,
                )
                st.session_state[RUN_KEY] = run
                st.session_state.pop(ERROR_KEY, None)
                st.session_state.pop(EXPORT_PATHS_KEY, None)
            except Exception as e:
                st.session_state[ERROR_KEY] = str(e)
                st.session_state.pop(RUN_KEY, None)

if st.session_state.get(ERROR_KEY):
    st.error(st.session_state[ERROR_KEY])

run: Optional[DemoRunResult] = st.session_state.get(RUN_KEY)
if run is None:
    st.info("Upload a video and click **Run pipeline**.")
    st.stop()

episode_dir = Path(run.episode_dir)
episode_json = safe_read_json(episode_dir / "episode.json")

st.write(f"**Input:** {run.input_description}")
st.write(f"**Episode dir:** `{episode_dir}`")
st.write(f"**Segmentation backend used:** `{run.backend_used.value}`")
if run.backend_message:
    st.warning(run.backend_message)

# Provenance badge: show detector mode actually used
with st.expander("Run Provenance", expanded=False):
    cfg = (episode_json.get("extra", {}) or {}).get("event_config", {})
    st.json(
        {
            "event_config_from_episode": cfg,
            "num_events_in_episode_json": len(episode_json.get("events", []) or []),
            "note": "This is the ground truth of what ran (not UI intent).",
        }
    )

pred_path = episode_dir / "pred_events.json"
pred_events = load_pred_events(pred_path)

# ---------------- Layout ----------------
col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    st.subheader("Segmentation Playback")
    num_frames = int((episode_json.get("video", {}) or {}).get("num_frames", run.num_frames) or run.num_frames)
    if num_frames <= 0:
        st.error("No frames detected.")
    else:
        frame_idx = st.slider("Frame index", min_value=0, max_value=max(0, num_frames - 1), value=0)
        img = overlay_frame_with_mask(episode_dir=episode_dir, frame_idx=frame_idx, overlay_alpha=0.55)
        if img is not None:
            st.image(img)

with col_right:
    st.subheader("Event Timeline (Hero Feature)")
    st.caption("If you have only 1 event, you will see a single bar. Tune min_score/topk to increase event density.")

    fig = build_event_timeline_plotly(pred_events, total_frames=max(num_frames, 1))
    st.plotly_chart(fig)

    chosen_event = None
    if pred_events:
        opts = []
        for i, e in enumerate(pred_events):
            s = e.get("start_frame", 0)
            t = e.get("end_frame", s)
            opts.append(f"#{i} {e.get('label','?')} [{s}–{t}] score={e.get('score','n/a')}")
        chosen = st.selectbox("Select an event", opts, index=0)
        chosen_event = pred_events[opts.index(chosen)]
    else:
        st.warning("No predicted events found in pred_events.json.")

    st.subheader("Event Slice Viewer")
    if chosen_event:
        render_event_slice_gallery(episode_dir=episode_dir, event=chosen_event, max_frames=12)

# ---------------- Export + Download ----------------
st.divider()
st.subheader("Export + Download")

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.download_button(
        "⬇ Download episode.zip",
        data=zip_directory_to_bytes(episode_dir),
        file_name=f"{episode_dir.name}.zip",
        mime="application/zip",
    )

export_paths = st.session_state.get(EXPORT_PATHS_KEY, {}) or {}

with c2:
    if st.button("Export: COCO"):
        ok, msg = run.try_export("coco")
        (st.success if ok else st.warning)(msg)
        if ok:
            export_paths["coco"] = str(episode_dir / "export_coco" / "annotations.json")
            st.session_state[EXPORT_PATHS_KEY] = export_paths

with c3:
    if st.button("Export: RLDS"):
        ok, msg = run.try_export("rlds")
        (st.success if ok else st.warning)(msg)
        if ok:
            export_paths["rlds"] = str(episode_dir / "export_rlds" / "rlds.json")
            st.session_state[EXPORT_PATHS_KEY] = export_paths

with c4:
    if st.button("Export: LeRobot"):
        ok, msg = run.try_export("lerobot")
        (st.success if ok else st.warning)(msg)
        if ok:
            export_paths["lerobot"] = str(episode_dir / "export_lerobot" / "lerobot.json")
            st.session_state[EXPORT_PATHS_KEY] = export_paths

if export_paths:
    st.caption("Downloads (generated this session):")
    d1, d2, d3 = st.columns(3)
    if "coco" in export_paths:
        p = Path(export_paths["coco"])
        if p.exists():
            d1.download_button("⬇ COCO annotations.json", data=p.read_bytes(), file_name="annotations.json", mime="application/json")
    if "rlds" in export_paths:
        p = Path(export_paths["rlds"])
        if p.exists():
            d2.download_button("⬇ RLDS rlds.json", data=p.read_bytes(), file_name="rlds.json", mime="application/json")
    if "lerobot" in export_paths:
        p = Path(export_paths["lerobot"])
        if p.exists():
            d3.download_button("⬇ LeRobot lerobot.json", data=p.read_bytes(), file_name="lerobot.json", mime="application/json")