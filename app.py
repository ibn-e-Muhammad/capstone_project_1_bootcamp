# =============================================================================
# app.py — KINETIC ENGINE | Machine Failure Prediction System
# Phase 2: Streamlit Deployment & Inference Engine
# =============================================================================
# Run with: streamlit run app.py
# Demo credentials: Operator ID = STATION_NODE_492  |  Key = kinetic2024
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timezone

# =============================================================================
# STEP 1: PAGE CONFIGURATION (must be the very first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="KINETIC ENGINE | Command Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STEP 2: GLOBAL CSS INJECTION — Kinetic Engine Override
# =============================================================================
GLOBAL_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>

<style>
/* ── Kinetic Engine: Global Resets ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #131313 !important;
    color: #E5E2E1 !important;
}

/* ── Hide Streamlit Chrome ── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer,
header {
    display: none !important;
    visibility: hidden !important;
}

/* ── Remove top padding added by hidden header ── */
[data-testid="stAppViewContainer"] > section:first-child {
    padding-top: 0 !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Force Space Grotesk on all headings ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E5E2E1 !important;
}

/* ── Sidebar Override ── */
[data-testid="stSidebar"] {
    background-color: #1C1B1B !important;
    border-right: 1px solid rgba(59, 74, 68, 0.15) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
    color: #BACAC3 !important;
}
[data-testid="stSidebar"] label {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-weight: 700 !important;
    color: #BACAC3 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stNumberInput > div {
    background-color: #201f1f !important;
    border: 1px solid rgba(59, 74, 68, 0.2) !important;
    border-radius: 8px !important;
    color: #E5E2E1 !important;
    font-family: 'Inter', monospace !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #45FDD2 !important;
    box-shadow: 0 0 0 2px rgba(69, 253, 210, 0.1) !important;
}

/* ── Button Override ── */
.stButton > button {
    background: #45FDD2 !important;
    color: #002018 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    box-shadow: 0 0 20px rgba(69, 253, 210, 0.2) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: #00e0b7 !important;
    box-shadow: 0 0 30px rgba(69, 253, 210, 0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #1C1B1B !important;
    border: 1px solid rgba(59, 74, 68, 0.15) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    color: #BACAC3 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background-color: #1C1B1B !important;
    border: 1px solid rgba(59, 74, 68, 0.15) !important;
    border-radius: 8px !important;
}

/* ── Material Symbols icon normalization ── */
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-size: 20px;
    line-height: 1;
    vertical-align: middle;
}

/* ── Keyframe Animations ── */
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(69, 253, 210, 0.4); }
    70%  { box-shadow: 0 0 0 12px rgba(69, 253, 210, 0); }
    100% { box-shadow: 0 0 0 0 rgba(69, 253, 210, 0); }
}
@keyframes pulse-ring-danger {
    0%   { box-shadow: 0 0 0 0 rgba(211, 0, 23, 0.4); }
    70%  { box-shadow: 0 0 0 12px rgba(211, 0, 23, 0); }
    100% { box-shadow: 0 0 0 0 rgba(211, 0, 23, 0); }
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# =============================================================================
# STEP 3: LOGIN SCREEN (Session-State Gate)
# =============================================================================
DEMO_OPERATOR_ID = "STATION_NODE_492"
DEMO_KEY = "kinetic2024"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    LOGIN_HTML = """
    <style>
    /* Full-page login override */
    [data-testid="stAppViewContainer"] {
        background-color: #131313 !important;
        background-image: radial-gradient(circle at 2px 2px, rgba(59,74,68,0.12) 1px, transparent 0) !important;
        background-size: 40px 40px !important;
    }
    /* Decorative glow nodes */
    .login-node-1, .login-node-2, .login-node-3 {
        position: fixed;
        border-radius: 50%;
        background: #45FDD2;
        pointer-events: none;
        z-index: 0;
    }
    .login-node-1 { width: 8px; height: 8px; top: 15%; left: 12%; opacity: 0.35; box-shadow: 0 0 15px rgba(69,253,210,0.5); }
    .login-node-2 { width: 12px; height: 12px; bottom: 22%; right: 8%; opacity: 0.2; box-shadow: 0 0 20px rgba(69,253,210,0.4); }
    .login-node-3 { width: 6px;  height: 6px;  top: 55%; right: 18%; opacity: 0.28; box-shadow: 0 0 12px rgba(69,253,210,0.4); }
    /* Decorative rings */
    .login-ring-outer, .login-ring-inner {
        position: fixed;
        border-radius: 50%;
        border: 1px solid rgba(59,74,68,0.08);
        pointer-events: none;
        z-index: 0;
    }
    .login-ring-outer { width: 520px; height: 520px; top: calc(50% - 260px); left: calc(50% - 260px); }
    .login-ring-inner { width: 320px; height: 320px; top: calc(50% - 160px); left: calc(50% - 160px); border-color: rgba(59,74,68,0.12); }
    /* Bottom-right decorative panel */
    .login-corner-panel {
        position: fixed; bottom: 0; right: 0;
        width: 220px; height: 220px;
        border-top: 1px solid rgba(59,74,68,0.1);
        border-left: 1px solid rgba(59,74,68,0.1);
        transform: rotate(12deg) translate(30%, 30%);
        pointer-events: none; z-index: 0;
    }
    .login-corner-text {
        position: fixed; bottom: 24px; right: 24px;
        font-family: monospace; font-size: 9px;
        color: rgba(186,202,195,0.35);
        text-align: right; line-height: 1.6;
        pointer-events: none; z-index: 1;
    }
    /* Gradient mask */
    .login-mask {
        position: fixed; inset: 0;
        background: linear-gradient(135deg, #131313 0%, transparent 50%, #1C1B1B 100%);
        opacity: 0.75;
        pointer-events: none; z-index: 0;
    }
    /* Hide Streamlit sidebar on login page */
    [data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stSidebarUserContent"] { display: none !important; }
    .block-container { max-width: 480px !important; margin: 0 auto !important; padding-top: 6vh !important; }
    /* Form inputs on login */
    .login-form input {
        background-color: #1C1B1B !important;
        border: 1px solid rgba(59,74,68,0.25) !important;
        border-radius: 8px !important;
        color: #E5E2E1 !important;
        font-family: monospace !important;
    }
    .login-form input:focus {
        border-color: #45FDD2 !important;
        box-shadow: 0 0 0 2px rgba(69,253,210,0.12) !important;
    }
    </style>
    <div class="login-node-1"></div>
    <div class="login-node-2"></div>
    <div class="login-node-3"></div>
    <div class="login-ring-outer"></div>
    <div class="login-ring-inner"></div>
    <div class="login-mask"></div>
    <div class="login-corner-panel"></div>
    <div class="login-corner-text">LAT_23.4902<br/>LON_89.0231<br/>GRID_ACTIVE_READY</div>

    <!-- Branding -->
    <div style="text-align:center; margin-bottom: 2.5rem; position:relative; z-index:2;">
        <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.25rem; font-weight:900;
                   letter-spacing:-0.04em; color:#45FDD2; margin:0 0 10px 0;">KINETIC ENGINE</h1>
        <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
            <span style="height:1px; width:32px; background:rgba(59,74,68,0.35); display:inline-block;"></span>
            <p style="font-family:'Inter',sans-serif; font-size:10px; text-transform:uppercase;
                       letter-spacing:0.2em; color:#BACAC3; margin:0; font-weight:500;">Central Portal Access</p>
            <span style="height:1px; width:32px; background:rgba(59,74,68,0.35); display:inline-block;"></span>
        </div>
    </div>
    """
    st.markdown(LOGIN_HTML, unsafe_allow_html=True)

    # Glassmorphism card wrapper
    st.markdown("""
    <div style="background:rgba(42,42,42,0.7); backdrop-filter:blur(24px);
                border-radius:12px; padding:2rem; position:relative; z-index:2;
                border:1px solid rgba(59,74,68,0.12);
                box-shadow:0 24px 48px rgba(0,0,0,0.5);">
        <h2 style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem;
                   color:#00e0b7; margin:0 0 6px 0; letter-spacing:0.02em;">AUTHENTICATE_SESSION</h2>
        <p style="font-size:13px; color:#BACAC3; margin:0 0 1.8rem 0;">
            Input operational credentials to proceed.</p>
    </div>
    """, unsafe_allow_html=True)

    # Actual Streamlit form (works inside the card via CSS offset)
    with st.form("login_form"):
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        operator_id = st.text_input(
            "OPERATOR ID",
            placeholder="STATION_NODE_492",
            help="Your station node identifier"
        )
        security_key = st.text_input(
            "SECURITY KEY",
            placeholder="••••••••••••",
            type="password",
            help="Operational security key"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  AUTHENTICATE", use_container_width=True)

    if submitted:
        if operator_id == DEMO_OPERATOR_ID and security_key == DEMO_KEY:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.markdown("""
            <div style="background:rgba(211,0,23,0.1); border:1px solid rgba(211,0,23,0.25);
                        border-radius:8px; padding:12px 16px; margin-top:12px; position:relative; z-index:2;">
                <p style="color:#ff6b6b; font-size:12px; font-weight:600; margin:0; letter-spacing:0.05em;">
                    ⚠ AUTH DENIED — Invalid credentials. Try STATION_NODE_492 / kinetic2024</p>
            </div>
            """, unsafe_allow_html=True)

    # Status footer on login page
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-top:3rem; padding:0 4px; opacity:0.5; position:relative; z-index:2;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:8px; height:8px; background:#45FDD2; border-radius:50%;
                        animation: pulse-dot 2s ease-in-out infinite;"></div>
            <span style="font-family:monospace; font-size:9px; color:#BACAC3; letter-spacing:0.1em;">
                SECURE_CORE_ONLINE</span>
        </div>
        <span style="font-family:monospace; font-size:9px; color:#BACAC3;">v4.8.2 // LATENCY: 2ms</span>
    </div>
    """, unsafe_allow_html=True)

    st.stop()  # Halt rendering — don't show anything below this point

# =============================================================================
# STEP 4: LOAD ML ASSETS (Cached) — only reached after authentication
# =============================================================================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_assets()

NUMERICAL_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# =============================================================================
# STEP 5: SIDEBAR — Kinetic Engine Branding + Sensor Inputs
# =============================================================================
with st.sidebar:
    # Station branding header
    st.markdown("""
    <div style="padding:1.5rem 1rem 1rem; background:linear-gradient(180deg,#1C1B1B 0%,transparent 100%);">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span class="material-symbols-outlined" style="color:#45FDD2; font-size:18px;">precision_manufacturing</span>
            <h2 style="font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:700;
                       color:#45FDD2; margin:0; letter-spacing:0.02em;">STATION_04</h2>
        </div>
        <p style="font-size:9px; text-transform:uppercase; letter-spacing:0.2em;
                  color:#BACAC3; margin:0 0 1.2rem 0;">Active Engine</p>
    </div>

    <!-- Nav Items -->
    <div style="display:flex; align-items:center; gap:10px; padding:10px 16px; margin:0 -1rem;
                background:rgba(69,253,210,0.08); border-right:3px solid #45FDD2;
                color:#45FDD2; font-size:13px; font-weight:500; font-family:'Inter',sans-serif; margin-bottom:4px;">
        <span class="material-symbols-outlined" style="color:#45FDD2; font-size:18px;">dashboard</span>
        <span style="color:#45FDD2;">Overview</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px; padding:10px 16px; margin:0 -1rem;
                color:#BACAC3; font-size:13px; font-weight:500; font-family:'Inter',sans-serif; margin-bottom:4px; cursor:pointer;">
        <span class="material-symbols-outlined" style="color:#BACAC3; font-size:18px;">sensors</span>
        <span>Telemetry</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px; padding:10px 16px; margin:0 -1rem;
                color:#BACAC3; font-size:13px; font-weight:500; font-family:'Inter',sans-serif; margin-bottom:4px; cursor:pointer;">
        <span class="material-symbols-outlined" style="color:#BACAC3; font-size:18px;">online_prediction</span>
        <span>Predictions</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px; padding:10px 16px; margin:0 -1rem;
                color:#BACAC3; font-size:13px; font-weight:500; font-family:'Inter',sans-serif; margin-bottom:20px; cursor:pointer;">
        <span class="material-symbols-outlined" style="color:#BACAC3; font-size:18px;">build</span>
        <span>Maintenance</span>
    </div>

    <div style="height:1px; background:rgba(59,74,68,0.2); margin:0 0 1.2rem 0;"></div>
    <p style="font-size:9px; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3;
              font-family:'Space Grotesk',sans-serif; margin:0 0 14px 0; padding:0 2px;">⚙ Sensor Inputs</p>
    """, unsafe_allow_html=True)

    # ── Machine type ──
    machine_type = st.selectbox(
        "Machine Type",
        options=["Low", "Medium", "High"],
        index=0,
        help="Quality variant (L / M / H)",
    )

    st.markdown('<p style="font-size:9px; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3; margin:12px 0 4px; font-family:Space Grotesk,sans-serif;">🌡 Temperature</p>', unsafe_allow_html=True)
    air_temp = st.number_input("Air Temperature [K]", min_value=295.0, max_value=305.0, value=300.0, step=0.1, format="%.1f")
    process_temp = st.number_input("Process Temperature [K]", min_value=305.0, max_value=315.0, value=310.0, step=0.1, format="%.1f")

    st.markdown('<p style="font-size:9px; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3; margin:12px 0 4px; font-family:Space Grotesk,sans-serif;">🔄 Mechanical Load</p>', unsafe_allow_html=True)
    rot_speed = st.number_input("Rotational Speed [rpm]", min_value=1168, max_value=2886, value=1538, step=10)
    torque = st.number_input("Torque [Nm]", min_value=3.8, max_value=76.6, value=40.0, step=0.5, format="%.1f")

    st.markdown('<p style="font-size:9px; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3; margin:12px 0 4px; font-family:Space Grotesk,sans-serif;">🔩 Wear</p>', unsafe_allow_html=True)
    tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=253, value=107, step=1)

    st.markdown("<br/>", unsafe_allow_html=True)
    predict_button = st.button("⚡  Run Prediction", use_container_width=True)

    # Divider
    st.markdown('<div style="height:1px; background:rgba(59,74,68,0.2); margin:1.5rem 0;"></div>', unsafe_allow_html=True)

    # ── Alerts & History panel in sidebar ──
    NOW = datetime.now(timezone.utc).strftime("%H:%M:%S")
    H1 = (datetime.now(timezone.utc).replace(microsecond=0).strftime("%H:%M:%S"))
    st.markdown(f"""
    <div style="background:#1C1B1B; border-radius:10px; overflow:hidden;
                border:1px solid rgba(59,74,68,0.12);">
        <div style="padding:14px 16px; background:rgba(42,42,42,0.5);
                    border-bottom:1px solid rgba(59,74,68,0.12);">
            <p style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700;
                      text-transform:uppercase; letter-spacing:0.2em; color:#E5E2E1; margin:0;">
                Alerts &amp; History</p>
        </div>
        <div style="padding:12px; display:flex; flex-direction:column; gap:10px;">

            <!-- Alert: Critical -->
            <div style="display:flex; gap:12px; padding:10px; border-radius:6px;">
                <span class="material-symbols-outlined" style="color:#D30017; font-size:18px; flex-shrink:0;">warning</span>
                <div>
                    <p style="font-size:11px; font-weight:700; color:#E5E2E1; margin:0 0 3px;">Thermal Spike Detected</p>
                    <p style="font-size:9px; color:#BACAC3; margin:0 0 4px; line-height:1.5;">
                        Core temp exceeded threshold on Spindle 02. Auto-cooling engaged.</p>
                    <span style="font-size:9px; font-family:monospace; color:#D30017; text-transform:uppercase;">{NOW} · Critical</span>
                </div>
            </div>

            <!-- Alert: Anomaly -->
            <div style="display:flex; gap:12px; padding:10px; border-radius:6px;">
                <span class="material-symbols-outlined" style="color:#F08C00; font-size:18px; flex-shrink:0;">error</span>
                <div>
                    <p style="font-size:11px; font-weight:700; color:#E5E2E1; margin:0 0 3px;">Vibration Anomaly</p>
                    <p style="font-size:9px; color:#BACAC3; margin:0 0 4px; line-height:1.5;">
                        Minor harmonic deviance noted in X-axis movement.</p>
                    <span style="font-size:9px; font-family:monospace; color:#F08C00; text-transform:uppercase;">14:15:44 · Anomaly</span>
                </div>
            </div>

            <!-- Alert: Routine -->
            <div style="display:flex; gap:12px; padding:10px; border-radius:6px;">
                <span class="material-symbols-outlined" style="color:#45FDD2; font-size:18px; flex-shrink:0;">check_circle</span>
                <div>
                    <p style="font-size:11px; font-weight:700; color:#E5E2E1; margin:0 0 3px;">Routine Calibration</p>
                    <p style="font-size:9px; color:#BACAC3; margin:0 0 4px; line-height:1.5;">
                        Sensor sync complete. Zero-point verified.</p>
                    <span style="font-size:9px; font-family:monospace; color:#BACAC3; text-transform:uppercase;">13:50:00 · Routine</span>
                </div>
            </div>

            <!-- Alert: Startup -->
            <div style="display:flex; gap:12px; padding:10px; border-radius:6px;">
                <span class="material-symbols-outlined" style="color:#45FDD2; font-size:18px; flex-shrink:0;">check_circle</span>
                <div>
                    <p style="font-size:11px; font-weight:700; color:#E5E2E1; margin:0 0 3px;">System Startup</p>
                    <p style="font-size:9px; color:#BACAC3; margin:0 0 4px; line-height:1.5;">
                        STATION_04 initialized by Oper_092.</p>
                    <span style="font-size:9px; font-family:monospace; color:#BACAC3; text-transform:uppercase;">08:00:00 · Routine</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Sidebar footer -->
    <div style="height:1px; background:rgba(59,74,68,0.2); margin:1.2rem 0;"></div>
    <div style="display:flex; align-items:center; gap:10px; padding:8px 0; color:#BACAC3;
                font-size:12px; font-weight:500; font-family:'Inter',sans-serif; cursor:pointer;">
        <span class="material-symbols-outlined" style="color:#BACAC3; font-size:18px;">settings</span>
        <span>Settings</span>
    </div>
    """, unsafe_allow_html=True)

    # Logout button
    if st.button("↩  System Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# =============================================================================
# STEP 6: INFERENCE ENGINE
# =============================================================================
# Default state before any prediction is made
if "failure_pct" not in st.session_state:
    st.session_state.failure_pct = None
if "health_pct" not in st.session_state:
    st.session_state.health_pct = None

if predict_button:
    type_l = 1 if machine_type == "Low" else 0
    type_m = 1 if machine_type == "Medium" else 0
    raw_input = {
        "Air temperature [K]":      [air_temp],
        "Process temperature [K]":  [process_temp],
        "Rotational speed [rpm]":   [rot_speed],
        "Torque [Nm]":              [torque],
        "Tool wear [min]":          [tool_wear],
        "Type_L":                   [type_l],
        "Type_M":                   [type_m],
    }
    input_df = pd.DataFrame(raw_input).reindex(columns=feature_columns, fill_value=0)
    input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])
    proba = model.predict_proba(input_df)[0][1]
    st.session_state.failure_pct = proba * 100
    st.session_state.health_pct = (1 - proba) * 100
    st.session_state.last_input_df = input_df

# Resolve display values
failure_pct = st.session_state.failure_pct if st.session_state.failure_pct is not None else 0.42
health_pct  = st.session_state.health_pct  if st.session_state.health_pct  is not None else 94.0
is_critical = failure_pct >= 85

# Dynamic color tokens
accent_color  = "#D30017" if is_critical else "#45FDD2"
accent_rgba   = "rgba(211,0,23,0.15)" if is_critical else "rgba(69,253,210,0.10)"
accent_border = "rgba(211,0,23,0.25)" if is_critical else "rgba(69,253,210,0.20)"
status_label  = "CRITICAL THRESHOLD" if is_critical else "NOMINAL STATE"
pulse_anim    = "pulse-ring-danger" if is_critical else "pulse-ring"

# =============================================================================
# STEP 7: TOP APP BAR (injected HTML)
# =============================================================================
NOW_UTC = datetime.now(timezone.utc).strftime("UTC %H:%M:%S")
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            padding:14px 24px; background:rgba(19,19,19,0.85); backdrop-filter:blur(16px);
            border-bottom:1px solid rgba(59,74,68,0.15);
            box-shadow:0 24px 48px rgba(0,0,0,0.5); margin:-1.5rem -1rem 1.5rem; border-radius:0;">
    <div style="display:flex; align-items:center; gap:24px;">
        <h1 style="font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:900;
                   letter-spacing:-0.04em; color:{accent_color}; margin:0; text-transform:uppercase;">
            KINETIC ENGINE</h1>
        <nav style="display:flex; gap:20px; font-family:'Space Grotesk',sans-serif;
                    text-transform:uppercase; letter-spacing:0.15em; font-size:10px;">
            <span style="color:{accent_color}; border-bottom:2px solid {accent_color}; padding-bottom:2px;">Dashboard</span>
            <span style="color:#BACAC3;">Analytics</span>
            <span style="color:#BACAC3;">Diagnostics</span>
            <span style="color:#BACAC3;">Reports</span>
        </nav>
    </div>
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="display:flex; align-items:center; gap:8px; background:#1C1B1B;
                    padding:6px 14px; border-radius:8px; border:1px solid rgba(59,74,68,0.15);">
            <span class="material-symbols-outlined" style="color:{accent_color}; font-size:16px;">precision_manufacturing</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:10px; text-transform:uppercase;
                         letter-spacing:0.15em; color:#BACAC3;">Milling Unit 04</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:7px; height:7px; background:{accent_color}; border-radius:50%;
                        animation:{pulse_anim} 2s ease-in-out infinite;"></div>
            <span style="font-family:monospace; font-size:10px; color:{accent_color}; letter-spacing:0.05em;">LIVE</span>
        </div>
        <span style="font-family:monospace; font-size:10px; color:#BACAC3;">{NOW_UTC}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# STEP 8: HERO ROW — Machine Health Ring + Failure Probability Panel
# =============================================================================
col_ring, col_prob = st.columns([4, 8], gap="medium")

with col_ring:
    ring_border_style = f"border-color:{accent_color}; border-top-color:transparent;"
    inner_glow = f"box-shadow: 0 0 20px {'rgba(211,0,23,0.3)' if is_critical else 'rgba(69,253,210,0.2)'};"
    st.markdown(f"""
    <div style="background:#2A2A2A; padding:2.5rem; border-radius:12px; height:100%;
                display:flex; flex-direction:column; justify-content:center; align-items:center;
                position:relative; overflow:hidden; min-height:260px;">
        <!-- Subtle background texture -->
        <div style="position:absolute; inset:0; opacity:0.05;
                    background:radial-gradient(ellipse at 50% 0%, {accent_color} 0%, transparent 70%); pointer-events:none;"></div>
        <div style="position:relative; z-index:1; text-align:center;">
            <!-- Spinning outer ring -->
            <div style="display:inline-flex; align-items:center; justify-content:center;
                        width:180px; height:180px; border-radius:50%;
                        border:4px solid {accent_color}; border-top-color:transparent;
                        animation:spin-slow 4s linear infinite; margin-bottom:1.5rem;">
                <!-- Static inner ring (counter-spin illusion via wrapper) -->
                <div style="width:152px; height:152px; border-radius:50%;
                            border:3px solid {accent_color}; {inner_glow}
                            display:flex; flex-direction:column; align-items:center; justify-content:center;
                            background:#1C1B1B; animation:spin-slow 4s linear infinite reverse;">
                    <span style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:900;
                                 color:{accent_color}; line-height:1;">{health_pct:.0f}<span style="font-size:1.1rem;">%</span></span>
                    <span style="font-size:8px; text-transform:uppercase; letter-spacing:0.2em;
                                 color:#BACAC3; font-family:'Inter',sans-serif; margin-top:2px;">Machine Health</span>
                </div>
            </div>
            <p style="font-family:'Space Grotesk',sans-serif; font-size:9px; text-transform:uppercase;
                      letter-spacing:0.2em; color:#BACAC3; margin:0;">
                Operational Stability: {'⚠ CRITICAL' if is_critical else 'Optimal'}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_prob:
    # Safe bar width: failure_pct is the danger zone, health is safe
    safe_w = max(0, min(100 - failure_pct, 100))
    danger_w = 100 - safe_w
    st.markdown(f"""
    <div style="background:rgba(42,42,42,0.7); backdrop-filter:blur(24px); padding:2.5rem;
                border-radius:12px; height:100%; min-height:260px;
                border:1px solid rgba(59,74,68,0.10);
                box-shadow:0 24px 48px rgba(0,0,0,0.4); position:relative; overflow:hidden;">
        <!-- Glow accent -->
        <div style="position:absolute; top:-80px; right:-80px; width:240px; height:240px;
                    border-radius:50%; background:radial-gradient(circle, {accent_rgba} 0%, transparent 70%);
                    pointer-events:none;"></div>

        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:2rem; position:relative; z-index:1;">
            <div>
                <p style="font-family:'Space Grotesk',sans-serif; font-size:9px; text-transform:uppercase;
                           letter-spacing:0.3em; color:#BACAC3; margin:0 0 8px 0;">Failure Probability</p>
                <div style="display:flex; align-items:baseline; gap:12px;">
                    <span style="font-family:'Space Grotesk',sans-serif; font-size:5rem; font-weight:900;
                                 letter-spacing:-0.05em; color:#E5E2E1; line-height:1;">
                        {failure_pct:.2f}<span style="font-size:2rem; color:{accent_color};">%</span></span>
                    <div style="padding:4px 12px; background:{accent_rgba};
                                border:1px solid {accent_border}; border-radius:999px;">
                        <span style="font-size:9px; font-weight:700; color:{accent_color};
                                     text-transform:uppercase; letter-spacing:0.1em; font-family:'Space Grotesk',sans-serif;">
                            {status_label}</span>
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <span style="font-family:monospace; font-size:10px; color:#BACAC3;">{NOW_UTC}</span>
                <div style="display:flex; align-items:center; gap:6px; justify-content:flex-end; margin-top:8px;">
                    <div style="width:7px; height:7px; background:{accent_color}; border-radius:50%;
                                animation:{pulse_anim} 2s ease-in-out infinite;"></div>
                    <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; text-transform:uppercase;
                                 letter-spacing:0.15em; color:{accent_color};">Live Feed</span>
                </div>
            </div>
        </div>

        <!-- Dual-color progress bar -->
        <div style="width:100%; height:6px; background:#1C1B1B; border-radius:999px;
                    overflow:hidden; display:flex; gap:2px; position:relative; z-index:1;">
            <div style="height:100%; width:{safe_w:.1f}%; background:{accent_color};
                        box-shadow:0 0 10px rgba(69,253,210,0.5); border-radius:999px 0 0 999px;"></div>
            <div style="height:100%; width:{danger_w:.1f}%; background:#D30017; border-radius:0 999px 999px 0;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:8px; position:relative; z-index:1;">
            <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; text-transform:uppercase;
                         letter-spacing:0.15em; color:#BACAC3;">Safe Operating Zone</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; text-transform:uppercase;
                         letter-spacing:0.15em; color:#BACAC3;">Critical Threshold (85%)</span>
        </div>

        {
            f'''<div style="margin-top:1.5rem; padding:14px 16px; border-radius:8px; position:relative; z-index:1;
                         background:rgba(211,0,23,0.08); border:1px solid rgba(211,0,23,0.2);">
                <p style="color:#ff6b6b; font-size:12px; font-weight:700; margin:0; letter-spacing:0.05em;">
                    🚨 CRITICAL ALERT — Failure probability is {failure_pct:.1f}%.
                    Immediate maintenance intervention recommended.</p>
            </div>'''
            if is_critical and st.session_state.failure_pct is not None else
            f'''<div style="margin-top:1.5rem; padding:14px 16px; border-radius:8px; position:relative; z-index:1;
                         background:rgba(69,253,210,0.06); border:1px solid rgba(69,253,210,0.12);">
                <p style="color:#45FDD2; font-size:12px; font-weight:600; margin:0; letter-spacing:0.03em;">
                    {'✅ Machine Healthy — Failure probability is ' + f"{failure_pct:.1f}%" + '. All readings within normal parameters.' if st.session_state.failure_pct is not None else '👈 Adjust sensor inputs in the sidebar and click Run Prediction.'}</p>
            </div>'''
        }
    </div>
    """, unsafe_allow_html=True)

# Spacer
st.markdown("<br/>", unsafe_allow_html=True)

# =============================================================================
# STEP 9: 2×2 PLOTLY CHART GRID — Glowing Sensor Telemetry
# =============================================================================
np.random.seed(42)
t = np.arange(50)

# Simulate sensor history centered around current input values
temp_data   = np.clip(np.random.normal(air_temp, 0.3, 50).cumsum() / 50 + air_temp - 0.5, air_temp - 2, air_temp + 2)
vib_data    = np.abs(np.random.normal(0.08, 0.015, 50)) + np.linspace(0, 0.01, 50)
rpm_data    = np.random.normal(rot_speed, 30, 50)
wear_data   = np.linspace(0, tool_wear, 50) + np.random.normal(0, 3, 50)

def make_chart(x, y, title, current_val, unit, color="#45FDD2", fill_color="rgba(69,253,210,0.07)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=color, width=2, shape="spline"),
        fill="tozeroy",
        fillcolor=fill_color,
        showlegend=False,
        hovertemplate=f"<b>%{{y:.2f}} {unit}</b><extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=120,
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        showlegend=False,
        hovermode="x unified",
    )
    return fig

def chart_card(col, fig, title, val_str, is_danger=False):
    border_c = "#D30017" if is_danger else "#45FDD2"
    with col:
        st.markdown(f"""
        <div style="background:#1C1B1B; padding:20px 20px 12px; border-radius:10px;
                    border:1px solid rgba(59,74,68,0.08); margin-bottom:0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700;
                             text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3;">{title}</span>
                <span style="font-family:'Inter',sans-serif; font-size:13px; font-weight:700;
                             color:{'#D30017' if is_danger else '#E5E2E1'};">{val_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Place chart inside a container
        with st.container():
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Section header
st.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
    <span class="material-symbols-outlined" style="color:#45FDD2; font-size:18px;">monitoring</span>
    <p style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; text-transform:uppercase;
              letter-spacing:0.25em; color:#BACAC3; margin:0;">Live Sensor Telemetry</p>
</div>
""", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2, gap="medium")

with chart_col1:
    st.markdown(f"""<div style="background:#1C1B1B; padding:20px 20px 4px; border-radius:10px; border:1px solid rgba(59,74,68,0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3;">Core Temperature</span>
            <span style="font-size:13px; font-weight:700; color:#E5E2E1;">{air_temp:.1f} K</span>
        </div>""", unsafe_allow_html=True)
    fig_temp = make_chart(t, temp_data, "Core Temperature", air_temp, "K")
    st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col2:
    st.markdown(f"""<div style="background:#1C1B1B; padding:20px 20px 4px; border-radius:10px; border:1px solid rgba(59,74,68,0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3;">Vibration Analysis</span>
            <span style="font-size:13px; font-weight:700; color:#E5E2E1;">{vib_data[-1]:.3f} mm/s</span>
        </div>""", unsafe_allow_html=True)
    fig_vib = make_chart(t, vib_data, "Vibration Analysis", vib_data[-1], "mm/s")
    st.plotly_chart(fig_vib, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

chart_col3, chart_col4 = st.columns(2, gap="medium")

with chart_col3:
    st.markdown(f"""<div style="background:#1C1B1B; padding:20px 20px 4px; border-radius:10px; border:1px solid rgba(59,74,68,0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3;">Spindle Speed (RPM)</span>
            <span style="font-size:13px; font-weight:700; color:#E5E2E1;">{rot_speed:,}</span>
        </div>""", unsafe_allow_html=True)
    fig_rpm = make_chart(t, rpm_data, "Spindle Speed", rot_speed, "rpm")
    st.plotly_chart(fig_rpm, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col4:
    tool_wear_pct = (tool_wear / 253) * 100
    tw_is_critical = tool_wear_pct > 70
    tw_color = "#D30017" if tw_is_critical else "#45FDD2"
    tw_fill  = "rgba(211,0,23,0.07)" if tw_is_critical else "rgba(69,253,210,0.07)"
    st.markdown(f"""<div style="background:#1C1B1B; padding:20px 20px 4px; border-radius:10px; border:1px solid rgba(59,74,68,0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-family:'Space Grotesk',sans-serif; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:0.2em; color:#BACAC3;">Tool Surface Wear</span>
            <span style="font-size:13px; font-weight:700; color:{'#D30017' if tw_is_critical else '#E5E2E1'};">{100-tool_wear_pct:.0f}% Life Remaining</span>
        </div>""", unsafe_allow_html=True)
    fig_wear = make_chart(t, wear_data, "Tool Wear", tool_wear, "min", color=tw_color, fill_color=tw_fill)
    st.plotly_chart(fig_wear, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# Spacer
st.markdown("<br/>", unsafe_allow_html=True)

# =============================================================================
# STEP 10: TECHNICAL DETAIL EXPANDER
# =============================================================================
if st.session_state.failure_pct is not None:
    with st.expander("🔬  Technical Detail — Processed Input Vector"):
        st.markdown(
            "<p style='font-size:11px; color:#BACAC3; font-family:monospace;'>Values after one-hot encoding and standard scaling:</p>",
            unsafe_allow_html=True
        )
        st.dataframe(st.session_state.last_input_df, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div style="margin-top:2rem; padding:20px 0; border-top:1px solid rgba(59,74,68,0.15);
            display:flex; justify-content:space-between; align-items:center;">
    <span style="font-family:monospace; font-size:9px; color:#BACAC3; letter-spacing:0.05em;">
        KINETIC ENGINE v4.8.2 // AI4I 2020 Predictive Maintenance Dataset</span>
    <span style="font-family:monospace; font-size:9px; color:#BACAC3; letter-spacing:0.05em;">
        Built with Scikit-learn &amp; Streamlit // STATION_04 // GRID_ACTIVE_READY</span>
</div>
""", unsafe_allow_html=True)
