# app.py
# ═══════════════════════════════════════════════════════════
# SECUREGATE — STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════

import streamlit as st
import cv2
import json
import pandas as pd
import numpy as np
import tempfile
import os
from src.video_processor import VideoProcessor

st.set_page_config(
    page_title="SecureGate — AI Security",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Base ── */
    .stApp { background-color: #0a0a0f; color: #f1f5f9; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #12121a;
        border-right: 1px solid #1e1e2e;
    }

    /* ── SIDEBAR FIX: Always show toggle, never fully hide ── */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    /* When collapsed, keep a small strip so user can reopen */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 20px !important;
    }
    /* Make the collapse/expand arrow always red and visible */
    [data-testid="collapsedControl"] svg {
        fill: #ef4444 !important;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a0a0a, #2d1515);
        border: 1px solid #ef4444; border-radius: 8px;
        padding: 16px 24px; margin-bottom: 20px;
    }
    .main-header h1 {
        color: #ef4444; font-size: 28px; margin: 0;
        font-weight: 700; letter-spacing: 2px;
    }
    .main-header p { color: #94a3b8; margin: 4px 0 0 0; font-size: 13px; }

    /* ── Status banners ── */
    .status-normal {
        background: linear-gradient(90deg,#052e16,#14532d);
        border: 1px solid #22c55e; border-left: 4px solid #22c55e;
        border-radius: 6px; padding: 12px 16px; color: #86efac;
        font-weight: 600; font-size: 16px; letter-spacing: 1px;
    }
    .status-tailgating {
        background: linear-gradient(90deg,#2d0a0a,#450a0a);
        border: 1px solid #ef4444; border-left: 4px solid #ef4444;
        border-radius: 6px; padding: 12px 16px; color: #fca5a5;
        font-weight: 700; font-size: 16px; letter-spacing: 1px;
    }
    .status-piggybacking {
        background: linear-gradient(90deg,#1e0a2d,#2d1452);
        border: 1px solid #a855f7; border-left: 4px solid #a855f7;
        border-radius: 6px; padding: 12px 16px; color: #d8b4fe;
        font-weight: 700; font-size: 16px; letter-spacing: 1px;
    }

    /* ── Cards ── */
    .stat-card {
        background: #12121a; border: 1px solid #1e1e2e;
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .stat-value { font-size: 32px; font-weight: 700; color: #ef4444; }
    .stat-label {
        font-size: 12px; color: #64748b;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .alert-card {
        background: #2d0a0a; border: 1px solid #ef4444;
        border-radius: 8px; padding: 16px; margin: 8px 0;
    }

    /* ── Section headers ── */
    .section-header {
        color: #64748b; font-size: 11px; font-weight: 600;
        letter-spacing: 2px; text-transform: uppercase;
        margin: 16px 0 8px 0; border-bottom: 1px solid #1e1e2e;
        padding-bottom: 4px;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #ef4444; color: white; border: none;
        border-radius: 6px; font-weight: 600; letter-spacing: 1px;
    }
    .stButton > button:hover { background-color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_scenarios():
    with open("scenarios.json", "r") as f:
        return json.load(f)["scenarios"]


# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "is_running":          False,
        "alert_log":           [],
        "event_log":           [],
        "final_results":       None,
        "session_results":     {},
        "custom_line_y":       300,
        "custom_line_x_start": 50,
        "custom_line_x_end":   700,
        "custom_video_path":   None,
        "custom_video_name":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def get_scenario_options(scenarios):
    opts = {}
    for s in scenarios:
        emoji  = "✅" if s["expected_result"] == "NORMAL" else "🚨"
        source = "CP" if s["source"] == "ChokePoint Dataset" else "SR"
        label  = f"{emoji} {s['id']} — {s['name']} [{source}]"
        opts[label] = s
    return opts


def format_event_row(event: dict) -> dict:
    t = event.get("type", "")
    if t == "TAILGATING":
        return {
            "Time":      event.get("timestamp", ""),
            "Type":      "🚨 TAILGATING",
            "Person":    f"ID:{event.get('tailgater_id','?')}",
            "Followed":  f"ID:{event.get('authorized_id','?')}",
            "Gap (s)":   event.get("time_gap_seconds", ""),
            "Dist (px)": event.get("distance_pixels", ""),
            "Conf":      event.get("confidence", ""),
        }
    elif t == "PIGGYBACKING":
        return {
            "Time":      event.get("timestamp", ""),
            "Type":      "🚨 PIGGYBACK",
            "Person":    f"ID:{event.get('track_id','?')}",
            "Followed":  "—",
            "Gap (s)":   "—",
            "Dist (px)": f"h:{event.get('bbox_height','?')}",
            "Conf":      "HIGH",
        }
    return {}


def calculate_session_metrics(scenarios, session_results):
    TP = FP = TN = FN = 0
    for s in scenarios:
        sid = s["id"]
        if sid not in session_results:
            continue
        expected   = s["expected_result"]
        actual     = session_results[sid]
        want_alert = expected in ["TAILGATING", "PIGGYBACKING"]
        got_alert  = actual   in ["TAILGATING", "PIGGYBACKING"]
        if   want_alert and     got_alert:  TP += 1
        elif want_alert and not got_alert:  FN += 1
        elif not want_alert and got_alert:  FP += 1
        else:                               TN += 1
    p  = TP/(TP+FP)    if (TP+FP)       > 0 else 0
    r  = TP/(TP+FN)    if (TP+FN)       > 0 else 0
    f1 = 2*p*r/(p+r)   if (p+r)         > 0 else 0
    ac = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Precision": round(p*100,1), "Recall": round(r*100,1),
        "F1": round(f1*100,1),       "Accuracy": round(ac*100,1),
    }


def get_video_first_frame(video_path: str,
                           max_w: int = 680) -> tuple:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 3, 0))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, 1.0, 640, 480
    orig_h, orig_w = frame.shape[:2]
    if orig_w > max_w:
        scale  = max_w / orig_w
        frame  = cv2.resize(frame, (max_w, int(orig_h * scale)))
    else:
        scale = 1.0
    return frame, scale, orig_w, orig_h


def draw_gate_line_preview(frame_bgr, line_y, x_start, x_end):
    preview = frame_bgr.copy()
    h, w    = preview.shape[:2]

    # Subtle grid every 50px
    for y in range(0, h, 50):
        cv2.line(preview, (0,y), (w,y), (35,35,35), 1)
        if y % 100 == 0 and y > 0:
            cv2.putText(preview, str(y), (2, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60,60,60), 1)
    for x in range(0, w, 50):
        cv2.line(preview, (x,0), (x,h), (35,35,35), 1)
        if x % 100 == 0 and x > 0:
            cv2.putText(preview, str(x), (x+2, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60,60,60), 1)

    line_y  = max(0, min(line_y,  h-1))
    x_start = max(0, min(x_start, w-1))
    x_end   = max(0, min(x_end,   w-1))

    # Glow strip
    overlay = preview.copy()
    cv2.rectangle(overlay, (x_start, line_y-4), (x_end, line_y+4),
                  (0,0,180), -1)
    cv2.addWeighted(overlay, 0.3, preview, 0.7, 0, preview)

    # Solid line + handles
    cv2.line(preview, (x_start, line_y), (x_end, line_y), (0,0,255), 2)
    cv2.circle(preview, (x_start, line_y), 5, (0,220,255), -1)
    cv2.circle(preview, (x_end,   line_y), 5, (0,220,255), -1)

    label = f"ACCESS LINE  y={line_y}"
    (lw, lh), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    lx = x_start + 6
    ly = max(line_y - 10, lh + 4)
    cv2.rectangle(preview, (lx-2, ly-lh-2), (lx+lw+2, ly+2), (0,0,0), -1)
    cv2.putText(preview, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)
    return preview


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    init_session_state()
    scenarios        = load_scenarios()
    scenario_options = get_scenario_options(scenarios)

    # ── Header ─────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔴 SECUREGATE</h1>
        <p>AI-Powered Tailgating &amp; Piggybacking Detection System</p>
        <p style="color:#475569;font-size:11px;">
            YOLOv8n Detection • ByteTrack Tracking •
            Spatio-Temporal Analysis • Real-Time Alerts
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 16px 0;">
            <span style="font-size:32px;">🔴</span>
            <h3 style="color:#ef4444;margin:4px 0;letter-spacing:2px;">
                SECUREGATE
            </h3>
            <p style="color:#475569;font-size:11px;margin:0;">
                Security Intelligence System
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Scenario selector ───────────────────────────────
        st.markdown('<p class="section-header">📹 Select Scenario</p>',
                    unsafe_allow_html=True)
        selected_label    = st.selectbox(
            "Scenario", list(scenario_options.keys()),
            index=0, label_visibility="collapsed"
        )
        selected_scenario = scenario_options[selected_label]

        exp = selected_scenario["expected_result"]
        ec  = "#22c55e" if exp == "NORMAL" else "#ef4444"
        ee  = "✅" if exp == "NORMAL" else "🚨"
        st.markdown(f"""
        <div style="background:#12121a;border:1px solid #1e1e2e;
                    border-radius:6px;padding:10px;margin:6px 0;">
            <p style="color:#94a3b8;font-size:11px;margin:0;">
                {selected_scenario['source']}
            </p>
            <p style="color:#f1f5f9;font-size:12px;margin:4px 0;">
                {selected_scenario['description'][:85]}...
            </p>
            <p style="color:{ec};font-size:12px;font-weight:600;margin:4px 0;">
                {ee} Expected: {exp}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Config (read-only) ──────────────────────────────
        st.markdown('<p class="section-header">⚙️ Config</p>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0a0a0f;border:1px solid #1e1e2e;
                    border-radius:6px;padding:10px;font-size:12px;">
            <p style="color:#64748b;margin:2px 0;">Time threshold
                <span style="color:#94a3b8;float:right;">
                    {selected_scenario.get('time_threshold',5.0)}s
                </span></p>
            <p style="color:#64748b;margin:2px 0;">Proximity
                <span style="color:#94a3b8;float:right;">
                    {selected_scenario.get('proximity_threshold',150)}px
                </span></p>
            <p style="color:#64748b;margin:2px 0;">Confidence
                <span style="color:#94a3b8;float:right;">
                    {selected_scenario.get('confidence',0.5)}
                </span></p>
            <p style="color:#64748b;margin:2px 0;">Frame skip
                <span style="color:#94a3b8;float:right;">
                    {selected_scenario.get('frame_skip',1)}
                </span></p>
            <p style="color:#64748b;margin:2px 0;">Piggyback
                <span style="color:#94a3b8;float:right;">
                    {'Yes ✅' if selected_scenario.get('check_piggyback') else 'No'}
                </span></p>
        </div>
        <p style="color:#475569;font-size:10px;margin:3px 0 0 0;">
            ℹ️ Same as test_pipeline.py
        </p>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Upload ──────────────────────────────────────────
        st.markdown('<p class="section-header">📤 Custom Video</p>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload", type=["mp4","mov","avi"],
            label_visibility="collapsed"
        )
        use_upload = False
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.custom_video_name:
                tfile = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{uploaded_file.name.split('.')[-1]}"
                )
                tfile.write(uploaded_file.read())
                tfile.flush()
                st.session_state.custom_video_path = tfile.name
                st.session_state.custom_video_name = uploaded_file.name
            use_upload = True
            st.success(f"✅ {uploaded_file.name}")

        st.divider()

        # ── Run / Stop ──────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            run_button  = st.button("▶ RUN",  use_container_width=True,
                                    type="primary")
        with c2:
            stop_button = st.button("⏹ STOP", use_container_width=True)

        if stop_button:
            st.session_state.is_running = False

        st.divider()

        # ── Session stats ───────────────────────────────────
        st.markdown('<p class="section-header">📊 Session Stats</p>',
                    unsafe_allow_html=True)
        al  = st.session_state.alert_log
        tot = len(al)
        tg  = sum(1 for a in al if a.get("type") == "TAILGATING")
        pb  = sum(1 for a in al if a.get("type") == "PIGGYBACKING")

        st.markdown(f"""
        <div style="background:#12121a;border:1px solid #1e1e2e;
                    border-radius:6px;padding:10px;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#94a3b8;font-size:12px;">Alerts</span>
                <span style="color:#ef4444;font-weight:700;">{tot}</span>
            </div>
            <div style="display:flex;justify-content:space-between;
                        margin-top:3px;">
                <span style="color:#94a3b8;font-size:12px;">Tailgating</span>
                <span style="color:#ef4444;font-weight:700;">{tg}</span>
            </div>
            <div style="display:flex;justify-content:space-between;
                        margin-top:3px;">
                <span style="color:#94a3b8;font-size:12px;">Piggybacking</span>
                <span style="color:#a855f7;font-weight:700;">{pb}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.session_results:
            m = calculate_session_metrics(
                scenarios, st.session_state.session_results)
            n = m["TP"]+m["FP"]+m["TN"]+m["FN"]
            st.markdown(f"""
            <div style="background:#0a0a0f;border:1px solid #1e1e2e;
                        border-radius:6px;padding:10px;margin-top:8px;
                        font-size:12px;">
                <p style="color:#22c55e;font-weight:600;margin:0 0 4px 0;">
                    Metrics ({n} videos)
                </p>
                <p style="color:#64748b;margin:2px 0;">Precision
                    <span style="color:#94a3b8;float:right;">
                        {m['Precision']}%</span></p>
                <p style="color:#64748b;margin:2px 0;">Recall
                    <span style="color:#94a3b8;float:right;">
                        {m['Recall']}%</span></p>
                <p style="color:#64748b;margin:2px 0;">F1-Score
                    <span style="color:#94a3b8;float:right;">
                        {m['F1']}%</span></p>
                <p style="color:#64748b;margin:2px 0;">Accuracy
                    <span style="color:#94a3b8;float:right;">
                        {m['Accuracy']}%</span></p>
                <p style="color:#475569;font-size:10px;margin:4px 0 0 0;">
                    TP:{m['TP']} TN:{m['TN']}
                    FP:{m['FP']} FN:{m['FN']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.alert_log       = []
            st.session_state.event_log       = []
            st.session_state.session_results = {}
            st.rerun()

    # ═══════════════════════════════════════════════════════
    # CUSTOM UPLOAD — GATE LINE CALIBRATION
    # ═══════════════════════════════════════════════════════
    if use_upload and st.session_state.custom_video_path:
        st.markdown("---")
        st.markdown(
            '<p class="section-header">📏 Set Gate Line</p>',
            unsafe_allow_html=True
        )
        st.caption(
            "Drag sliders to position the red ACCESS LINE at the doorway. "
            "Preview updates live. Then click RUN."
        )

        frame_bgr, disp_scale, orig_w, orig_h = get_video_first_frame(
            st.session_state.custom_video_path
        )

        if frame_bgr is not None:
            fh, fw = frame_bgr.shape[:2]

            sc1, sc2, sc3 = st.columns([2, 1, 1])
            with sc1:
                disp_line_y = st.slider(
                    "↕ Gate Line Y",
                    min_value=10, max_value=fh-10,
                    value=(st.session_state.custom_line_y
                           if st.session_state.custom_line_y < fh
                           else fh//2),
                    key="slider_line_y"
                )
            with sc2:
                disp_x_start = st.slider(
                    "← Left",
                    min_value=0, max_value=fw//2,
                    value=min(st.session_state.custom_line_x_start, fw//2),
                    key="slider_x_start"
                )
            with sc3:
                disp_x_end = st.slider(
                    "→ Right",
                    min_value=fw//2, max_value=fw-1,
                    value=(max(st.session_state.custom_line_x_end, fw//2+1)
                           if st.session_state.custom_line_x_end < fw
                           else int(fw*0.95)),
                    key="slider_x_end"
                )

            # KEY FIX: save to session_state so RUN gets correct values
            st.session_state.custom_line_y       = disp_line_y
            st.session_state.custom_line_x_start = disp_x_start
            st.session_state.custom_line_x_end   = disp_x_end

            preview     = draw_gate_line_preview(
                frame_bgr, disp_line_y, disp_x_start, disp_x_end)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb,
                     caption=(f"y={disp_line_y}  "
                               f"x=[{disp_x_start}→{disp_x_end}]  "
                               f"| original {orig_w}×{orig_h}"),
                     use_container_width=True)

        else:
            st.warning("Cannot read frame from video.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════
    # MAIN COLUMNS
    # ═══════════════════════════════════════════════════════
    col_video, col_info = st.columns([3, 2])

    with col_video:
        st.markdown('<p class="section-header">📹 Live Detection Feed</p>',
                    unsafe_allow_html=True)
        video_placeholder = st.empty()
        video_placeholder.markdown("""
        <div style="background:#12121a;border:1px solid #1e1e2e;
                    border-radius:8px;height:380px;display:flex;
                    flex-direction:column;align-items:center;
                    justify-content:center;">
            <span style="font-size:56px;">🎥</span>
            <p style="color:#475569;margin-top:12px;">
                Select a scenario and click RUN
            </p>
        </div>
        """, unsafe_allow_html=True)
        progress_bar  = st.progress(0)
        progress_text = st.empty()

    with col_info:
        st.markdown('<p class="section-header">⚡ Status</p>',
                    unsafe_allow_html=True)
        status_ph = st.empty()
        status_ph.markdown("""
        <div class="status-normal">
            ✅ &nbsp; SYSTEM READY
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">📊 Stats</p>',
                    unsafe_allow_html=True)
        stats_ph = st.empty()
        stats_ph.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div class="stat-card">
                <div class="stat-value">—</div>
                <div class="stat-label">People</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">—</div>
                <div class="stat-label">Crossings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">—</div>
                <div class="stat-label">Alerts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">—</div>
                <div class="stat-label">FPS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">🚨 Latest Alert</p>',
                    unsafe_allow_html=True)
        latest_ph = st.empty()
        latest_ph.markdown("""
        <div style="background:#12121a;border:1px solid #1e1e2e;
                    border-radius:6px;padding:16px;text-align:center;">
            <p style="color:#475569;margin:0;">No alerts yet</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">📋 Event Log</p>',
                unsafe_allow_html=True)
    event_log_ph = st.empty()
    results_ph   = st.empty()

    # ═══════════════════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════════════════
    if run_button:
        st.session_state.is_running    = True
        st.session_state.alert_log     = []
        st.session_state.event_log     = []
        st.session_state.final_results = None

        if use_upload and st.session_state.custom_video_path:
            _, disp_scale, orig_w, orig_h = get_video_first_frame(
                st.session_state.custom_video_path)
            orig_ly = int(st.session_state.custom_line_y       / disp_scale)
            orig_xs = int(st.session_state.custom_line_x_start / disp_scale)
            orig_xe = int(st.session_state.custom_line_x_end   / disp_scale)
            process_scenario = {
                "id":                  "CUSTOM",
                "name":                f"Upload: {st.session_state.custom_video_name}",
                "description":         "User-uploaded video",
                "video_path":          st.session_state.custom_video_path,
                "source":              "Custom Upload",
                "expected_result":     "UNKNOWN",
                "evaluation_type":     "Custom",
                "line_type":           "horizontal",
                "flip_direction":      False,
                "line_y":              orig_ly,
                "line_x_start":        orig_xs,
                "line_x_end":          orig_xe,
                "time_threshold":      5.0,
                "proximity_threshold": 150,
                "confidence":          0.5,
                "check_piggyback":     False,
                "frame_skip":          1,
                "tags":                ["custom"],
            }
        else:
            process_scenario = dict(selected_scenario)
            # ── Ensure video is available (download if needed) ──
            from src.video_manager import ensure_video_available
            actual_path = ensure_video_available(
                process_scenario["video_path"]
            )
            if actual_path is None:
                st.error("❌ Video not available. Check Google Drive link.")
                st.stop()
            process_scenario["video_path"] = actual_path

        processor     = VideoProcessor(scenario=process_scenario)
        src           = process_scenario.get("source", "")
        display_every = 2 if src == "Self-Recorded" else 1
        disp_count    = 0

        for result in processor.process():
            if not st.session_state.is_running:
                break

            if result["is_complete"]:
                final = processor.get_final_results()
                st.session_state.final_results = final
                sid    = process_scenario.get("id", "CUSTOM")
                tge    = final.get("tailgating_events", 0)
                pbe    = final.get("piggyback_events",  0)
                actual = ("TAILGATING"   if tge > 0 else
                          "PIGGYBACKING" if pbe > 0 else "NORMAL")
                st.session_state.session_results[sid] = actual
                break

            frame = result["frame"]
            if frame is None:
                continue

            disp_count += 1

            if disp_count % display_every == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    frame_rgb, channels="RGB",
                    use_container_width=True)

            if result["total_frames"] > 0:
                prog = result["frame_number"] / result["total_frames"]
                progress_bar.progress(min(prog, 1.0))
                progress_text.markdown(
                    f"<p style='color:#475569;font-size:11px;"
                    f"text-align:center;'>"
                    f"Frame {result['frame_number']} / "
                    f"{result['total_frames']} &nbsp;|&nbsp; "
                    f"{result['fps']} FPS</p>",
                    unsafe_allow_html=True)

            s = result["status"]
            if s == "TAILGATING":
                status_ph.markdown("""
                <div class="status-tailgating">
                    🚨 &nbsp; TAILGATING DETECTED
                </div>""", unsafe_allow_html=True)
            elif s == "PIGGYBACKING":
                status_ph.markdown("""
                <div class="status-piggybacking">
                    🚨 &nbsp; PIGGYBACKING DETECTED
                </div>""", unsafe_allow_html=True)
            else:
                status_ph.markdown("""
                <div class="status-normal">
                    ✅ &nbsp; SYSTEM NORMAL
                </div>""", unsafe_allow_html=True)

            stats_ph.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div class="stat-card">
                    <div class="stat-value" style="color:#3b82f6;">
                        {result['persons_detected']}
                    </div>
                    <div class="stat-label">People</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color:#f59e0b;">
                        {result['crossings']}
                    </div>
                    <div class="stat-label">Crossings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color:#ef4444;">
                        {result['alerts']}
                    </div>
                    <div class="stat-label">Alerts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color:#22c55e;">
                        {result['fps']}
                    </div>
                    <div class="stat-label">FPS</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            alert_log = result["alert_log"]
            if alert_log != st.session_state.alert_log:
                st.session_state.alert_log = alert_log
                if alert_log:
                    latest = alert_log[-1]
                    atype  = latest.get("type", "")
                    if atype == "TAILGATING":
                        latest_ph.markdown(f"""
                        <div class="alert-card">
                            <p style="color:#ef4444;font-weight:700;
                                      margin:0 0 6px 0;">
                                🚨 TAILGATING — {latest['timestamp']}
                            </p>
                            <p style="color:#fca5a5;font-size:13px;margin:2px 0;">
                                Person {latest['tailgater_id']} followed
                                Person {latest['authorized_id']}
                            </p>
                            <p style="color:#94a3b8;font-size:12px;margin:2px 0;">
                                Gap: {latest['time_gap_seconds']}s &nbsp;|&nbsp;
                                Dist: {latest['distance_pixels']}px
                            </p>
                            <p style="color:#64748b;font-size:11px;
                                      margin:4px 0 0 0;">
                                📧 Email &nbsp;|&nbsp; 🔊 Alarm
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif atype == "PIGGYBACKING":
                        latest_ph.markdown(f"""
                        <div style="background:#1e0a2d;
                                    border:1px solid #a855f7;
                                    border-radius:8px;padding:14px;">
                            <p style="color:#a855f7;font-weight:700;
                                      margin:0 0 6px 0;">
                                🚨 PIGGYBACKING — {latest['timestamp']}
                            </p>
                            <p style="color:#d8b4fe;font-size:13px;margin:2px 0;">
                                Person {latest['track_id']} carrying another
                            </p>
                            <p style="color:#94a3b8;font-size:12px;margin:2px 0;">
                                Ratio: {latest['height_ratio']}x normal
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                rows = [format_event_row(a) for a in alert_log
                        if format_event_row(a)]
                if rows:
                    event_log_ph.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True, hide_index=True)

        st.session_state.is_running = False
        progress_bar.progress(1.0)
        progress_text.markdown(
            "<p style='color:#22c55e;font-size:12px;text-align:center;'>"
            "✅ Complete</p>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════════════════════════
    if st.session_state.final_results:
        r         = st.session_state.final_results
        expected  = r.get("expected_result", "UNKNOWN")
        tg_ev     = r.get("tailgating_events", 0)
        pb_ev     = r.get("piggyback_events",  0)
        got_alert = tg_ev > 0 or pb_ev > 0

        if expected == "UNKNOWN":
            verdict = "ℹ️ CUSTOM VIDEO"; vc = "#3b82f6"
            vd = "No expected result"
        elif expected == "NORMAL" and not got_alert:
            verdict = "✅ PASS"; vc = "#22c55e"
            vd = "Correctly NORMAL"
        elif expected in ["TAILGATING","PIGGYBACKING"] and got_alert:
            verdict = "✅ PASS"; vc = "#22c55e"
            vd = f"Correctly detected {expected}"
        elif expected == "NORMAL" and got_alert:
            verdict = "❌ FALSE POSITIVE"; vc = "#f59e0b"
            vd = "Alert on normal scenario"
        else:
            verdict = "❌ MISSED"; vc = "#ef4444"
            vd = f"Did not detect {expected}"

        results_ph.markdown(f"""
        <div style="background:#12121a;border:1px solid #1e1e2e;
                    border-radius:8px;padding:20px;margin-top:20px;">
            <h3 style="color:#f1f5f9;margin:0 0 14px 0;letter-spacing:1px;">
                📊 RESULTS — {r.get('scenario_name','')}
            </h3>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);
                        gap:10px;margin-bottom:14px;">
                <div class="stat-card">
                    <div class="stat-value" style="font-size:22px;">
                        {r.get('total_crossings_in',0)}
                    </div>
                    <div class="stat-label">Crossings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="font-size:22px;
                                                    color:#22c55e;">
                        {r.get('authorized_entries',0)}
                    </div>
                    <div class="stat-label">Authorized</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="font-size:22px;
                                                    color:#ef4444;">
                        {tg_ev}
                    </div>
                    <div class="stat-label">Tailgating</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="font-size:22px;
                                                    color:#a855f7;">
                        {pb_ev}
                    </div>
                    <div class="stat-label">Piggyback</div>
                </div>
            </div>
            <div style="background:#0a0a0f;border-radius:6px;padding:10px;">
                <p style="margin:0 0 3px 0;">
                    <span style="color:#64748b;">Expected: </span>
                    <span style="color:#94a3b8;">{expected}</span>
                    &nbsp;&nbsp;
                    <span style="color:#64748b;">FPS: </span>
                    <span style="color:#94a3b8;">{r.get('avg_fps',0)}</span>
                </p>
                <p style="margin:0;font-size:17px;font-weight:700;
                          color:{vc};">
                    {verdict} — {vd}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if r.get("alert_log"):
            csv = pd.DataFrame(r["alert_log"]).to_csv(index=False)
            st.download_button(
                "📥 Download Alert Log (CSV)",
                data=csv,
                file_name=f"securegate_{r.get('scenario_id','')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()