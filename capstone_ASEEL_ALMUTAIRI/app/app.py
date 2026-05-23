"""
Vantrex — Streamlit Web Application
Context-Aware IoT Cyber Activity Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
from typing import Tuple

# ── Path resolution ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vantrex",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Local font loading (base64-encoded for browser delivery) ─────────────────
import base64

def _b64_font(rel_path: str) -> str:
    """Return a base64 data-URI string for a local font file."""
    abs_path = os.path.join(BASE_DIR, rel_path)
    with open(abs_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")

_raleway_normal = _b64_font("fonts/Raleway-VariableFont_wght.ttf")
_raleway_italic = _b64_font("fonts/Raleway-Italic-VariableFont_wght.ttf")

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Local font faces (Raleway variable font, base64-embedded) ── */
@font-face {{
    font-family: 'Raleway';
    src: url('data:font/truetype;base64,{_raleway_normal}') format('truetype');
    font-weight: 100 900; font-style: normal; font-display: swap;
}}
@font-face {{
    font-family: 'Raleway';
    src: url('data:font/truetype;base64,{_raleway_italic}') format('truetype');
    font-weight: 100 900; font-style: italic; font-display: swap;
}}

/* ─────────────────────────────────────────────────────────
   DESIGN TOKENS
───────────────────────────────────────────────────────── */
:root {{
    --bg:            #F0F4FA;
    --card:          #FFFFFF;
    --sidebar-bg:    #0B1629;
    --primary:       #1B5FE6;
    --primary-light: #3B82F6;
    --cyan:          #06B6D4;
    --success:       #10B981;
    --success-bg:    #ECFDF5;
    --danger:        #EF4444;
    --danger-bg:     #FEF2F2;
    --warning:       #F59E0B;
    --warning-bg:    #FFFBEB;
    --text-primary:  #0F172A;
    --text-secondary:#475569;
    --border:        #E2E8F0;
    --shadow:        0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.07);
    --radius:        14px;
    --radius-sm:     8px;
    /* Sidebar palette */
    --sb-text:    #CBD5E1;
    --sb-heading: #FFFFFF;
    --sb-label:   #8BAAC8;
    --sb-muted:   #94A3B8;
}}

/* ─────────────────────────────────────────────────────────
   GLOBAL BASE
   Theme locked to light via .streamlit/config.toml —
   Streamlit itself supplies #0F172A text on #F0F4FA bg.
   We only override font-family here. Color is handled
   per-zone below so sidebar text is never overridden.
───────────────────────────────────────────────────────── */
html, body {{
    font-family: 'Raleway', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {{
    background-color: var(--bg) !important;
}}
.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1280px !important;
}}
/* Font family — targeted, NOT a wildcard.
   The * wildcard overrides Streamlit's Material Symbols icon font
   on the sidebar collapse button, turning glyphs into literal text. */
p, h1, h2, h3, h4, h5, h6, li,
div, span, label, td, th, caption,
input, select, textarea, button,
[data-testid] {{
    font-family: 'Raleway', sans-serif !important;
}}
/*
 * FIX: keyboard_double_arrow_left / keyboard_double icon broken text.
 * Streamlit renders the sidebar collapse button using Material Symbols
 * Rounded ligatures. Restore that font family so glyphs display correctly.
 */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] *,
button[aria-label="Collapse sidebar"],
button[aria-label="Collapse sidebar"] *,
button[aria-label="Expand sidebar"],
button[aria-label="Expand sidebar"] * {{
    font-family: 'Material Symbols Rounded', 'Material Icons', serif !important;
}}

/* ─────────────────────────────────────────────────────────
   SIDEBAR — dark navy bg, all text forced light.
   Every element gets #CBD5E1 as the base. Specific
   inline HTML blocks use  color:X !important  to override
   to brighter/different values where needed.
───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
}}
/* Force ALL sidebar text to a readable light color */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label span {{
    color: var(--sb-text) !important;   /* #CBD5E1 — readable on dark navy */
}}
/* Streamlit widget label containers */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label span,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
    color: var(--sb-text) !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: var(--sb-heading) !important;
    font-weight: 700 !important;
}}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: var(--sb-muted) !important;
}}

/* ─────────────────────────────────────────────────────────
   TABS
───────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'Raleway', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    border-radius: 8px 8px 0 0 !important;
    color: #374151 !important;
    letter-spacing: .01em !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: var(--primary) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--primary) !important;
}}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {{
    color: #1E293B !important;
}}

/* ─────────────────────────────────────────────────────────
   FORM LABELS & INPUT CONTROLS
───────────────────────────────────────────────────────── */
/* Labels above widgets */
.stSelectbox   > label,
.stNumberInput > label,
.stTextInput   > label,
.stFileUploader > label,
[data-testid="stWidgetLabel"] {{
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    letter-spacing: .01em !important;
}}
.stSelectbox   > label p,
.stNumberInput > label p,
.stTextInput   > label p,
.stFileUploader > label p,
[data-testid="stWidgetLabel"] p {{
    font-family: 'Raleway', sans-serif !important;
    font-weight: 600 !important;
    color: #1E293B !important;
}}
/* Input controls — white bg, dark text */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {{
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.91rem !important;
    font-weight: 500 !important;
    color: #0F172A !important;
    background: var(--card) !important;
    transition: border-color .15s !important;
}}
/* Selectbox displayed value text */
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
    color: #0F172A !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 500 !important;
}}
/* Placeholder */
.stSelectbox > div > div input::placeholder,
.stNumberInput > div > div > input::placeholder,
.stTextInput > div > div > input::placeholder {{
    color: #94A3B8 !important;
    font-weight: 400 !important;
}}
/* Focus ring */
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(27,95,230,.12) !important;
}}
/* Open dropdown list */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="popover"] li {{
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.91rem !important;
    font-weight: 500 !important;
    color: #0F172A !important;
    background: #FFFFFF !important;
}}
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {{
    background: #EFF6FF !important;
    color: var(--primary) !important;
}}
/* Checkbox — main area only */
.main .stCheckbox > label,
.main .stCheckbox label p,
.main [data-testid="stCheckbox"] label,
.main [data-testid="stCheckbox"] label p {{
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    color: #1E293B !important;
}}

/* ─────────────────────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: var(--shadow) !important;
}}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] p {{
    color: #475569 !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .07em !important;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-family: 'Raleway', sans-serif !important;
    letter-spacing: -.01em !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}}

/* ─────────────────────────────────────────────────────────
   BUTTONS
───────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    padding: .65rem 1.5rem !important;
    letter-spacing: .03em !important;
    box-shadow: 0 2px 8px rgba(27,95,230,.35) !important;
    transition: opacity .15s, box-shadow .15s !important;
}}
.stButton > button[kind="primary"]:hover {{
    opacity: .9 !important;
    box-shadow: 0 4px 16px rgba(27,95,230,.5) !important;
}}
.stButton > button:not([kind="primary"]),
.stDownloadButton > button {{
    background: var(--card) !important;
    color: var(--primary) !important;
    border: 1.5px solid var(--primary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.93rem !important;
    padding: .55rem 1.2rem !important;
    transition: background .15s !important;
}}
.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:hover {{
    background: #EFF6FF !important;
}}

/* ─────────────────────────────────────────────────────────
   FILE UPLOADER
   Root cause of "uploadUpload": the first span inside
   stFileUploaderDropzoneInstructions is a Material Symbols
   "upload" ligature. Our Raleway override turns it into
   literal text "upload", which stacks with the button label.
   Fix: hide that icon span only.
───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}
/* The inner dropzone area */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--card) !important;
    border: 2px dashed #C7D2DD !important;
    border-radius: var(--radius) !important;
    padding: 2rem 1.5rem !important;
    text-align: center !important;
    transition: border-color .2s, background .2s !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--primary) !important;
    background: #F8FAFF !important;
}}
/* Hide the broken Material Symbols "upload" ligature span.
   It is always the first span-child of the instructions div.
   The button text (Browse files / Upload) stays untouched. */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {{
    display: none !important;
}}
/* Instruction text (drag label + size limit) */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p {{
    font-family: 'Raleway', sans-serif !important;
    color: #475569 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    font-family: 'Raleway', sans-serif !important;
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
}}

/* ─────────────────────────────────────────────────────────
   DATAFRAME / EXPANDER / ALERTS / MISC
───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
}}
/* stExpander — minimal styling only. No summary/arrow manipulation
   to avoid icon font conflicts that produce "arrowdown" text. */
[data-testid="stExpander"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow) !important;
}}
[data-testid="stAlert"] {{
    border-radius: var(--radius-sm) !important;
    border-left-width: 4px !important;
    font-family: 'Raleway', sans-serif !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {{
    font-family: 'Raleway', sans-serif !important;
    font-weight: 500 !important;
}}
hr {{
    border-color: var(--border) !important;
    margin: 1.6rem 0 !important;
}}
.stCaption,
[data-testid="stCaptionContainer"] {{
    color: #475569 !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}}
code, pre, [data-testid="stCodeBlock"] {{
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 0.83rem !important;
    color: #1E293B !important;
}}
[data-testid="stSpinner"] p {{
    font-family: 'Raleway', sans-serif !important;
    color: #475569 !important;
    font-weight: 500 !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Reusable HTML card helpers ────────────────────────────────────────────────
def card(content_html: str, padding: str = "1.5rem") -> None:
    st.markdown(
        f"""<div style="
            background:#fff;border:1px solid #E2E8F0;border-radius:14px;
            padding:{padding};box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.07);
            margin-bottom:1rem;">
            {content_html}
        </div>""",
        unsafe_allow_html=True,
    )

def result_card(pred_label: str, conf: str, risk_text: str, prob: float) -> None:
    if pred_label == "Attack":
        border   = "#EF4444"
        bg       = "#FEF2F2"
        icon     = "🛑"
        headline = "THREAT DETECTED"
        desc     = f"This network activity matches <strong>Attack</strong> behaviour with <strong>{conf}</strong> confidence."
    else:
        border   = "#10B981"
        bg       = "#ECFDF5"
        icon     = "🟢"
        headline = "ACTIVITY NORMAL"
        desc     = f"This network activity appears <strong>benign</strong> with <strong>{conf}</strong> attack probability."

    bar_pct   = int(prob * 100)
    bar_color = "#EF4444" if prob >= 0.5 else "#10B981"

    st.markdown(f"""
    <div style="background:{bg};border:1.5px solid {border};border-radius:14px;
                padding:1.6rem 1.8rem;margin-top:1rem;">
        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem;">
            <span style="font-size:1.6rem;">{icon}</span>
            <span style="font-family:'Raleway',sans-serif;font-weight:800;
                         font-size:1.15rem;color:{border};letter-spacing:.04em;">
                {headline}
            </span>
        </div>
        <p style="font-family:'Raleway',sans-serif;font-size:.96rem;
                  color:#0F172A;margin:0 0 1rem 0;">{desc}</p>
        <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1rem;">
            <div>
                <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:.06em;color:#374151;margin-bottom:.25rem;">
                    PREDICTION
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-weight:600;
                             font-size:1.1rem;color:{border};">{pred_label}</div>
            </div>
            <div>
                <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:.06em;color:#374151;margin-bottom:.25rem;">
                    ATTACK PROB
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-weight:600;
                             font-size:1.1rem;color:#0F172A;">{conf}</div>
            </div>
            <div>
                <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:.06em;color:#374151;margin-bottom:.25rem;">
                    RISK LEVEL
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-weight:600;
                             font-size:1.1rem;color:#0F172A;">{risk_text}</div>
            </div>
        </div>
        <div style="background:#E2E8F0;border-radius:99px;height:8px;overflow:hidden;">
            <div style="width:{bar_pct}%;height:100%;background:{bar_color};
                        border-radius:99px;transition:width .4s;"></div>
        </div>
        <div style="font-size:.75rem;color:#374151;margin-top:.35rem;
                    font-family:'Raleway',sans-serif;">
            Attack probability bar — {bar_pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def section_heading(title: str, subtitle: str = "") -> None:
    sub = f'<p style="font-family:Raleway,sans-serif;font-size:.9rem;color:#374151;margin:0;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
        <h3 style="font-family:Raleway,sans-serif;font-weight:700;font-size:1.15rem;
                   color:#0F172A;margin:0 0 .25rem 0;">{title}</h3>
        {sub}
    </div>
    """, unsafe_allow_html=True)

def badge(text: str, color: str = "#1B5FE6", bg: str = "#EFF6FF") -> str:
    return (f'<span style="display:inline-block;background:{bg};color:{color};'
            f'font-family:Raleway,sans-serif;font-size:.75rem;font-weight:600;'
            f'padding:.2rem .65rem;border-radius:99px;letter-spacing:.03em;">{text}</span>')


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pipeline_path = os.path.join(MODELS_DIR, 'best_pipeline.pkl')
    nn_path       = os.path.join(MODELS_DIR, 'nn_model.keras')
    nn_pre_path   = os.path.join(MODELS_DIR, 'nn_preprocessor.pkl')
    label_path    = os.path.join(MODELS_DIR, 'label_mapping.json')
    feat_path     = os.path.join(MODELS_DIR, 'feature_config.json')

    if not os.path.exists(pipeline_path):
        st.error(
            f"Model file not found: {pipeline_path}\n\n"
            "Please run notebook.ipynb top-to-bottom first to generate the saved models."
        )
        st.stop()

    pipeline = joblib.load(pipeline_path)

    nn_model        = None
    nn_preprocessor = None
    if os.path.exists(nn_path) and os.path.exists(nn_pre_path):
        try:
            import tensorflow as tf
            nn_model        = tf.keras.models.load_model(nn_path)
            nn_preprocessor = joblib.load(nn_pre_path)
        except Exception as e:
            st.warning(f"Neural network could not be loaded: {e}")

    with open(label_path, 'r') as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}

    with open(feat_path, 'r') as f:
        feat_config = json.load(f)

    return pipeline, nn_model, nn_preprocessor, label_mapping, feat_config


pipeline, nn_model, nn_preprocessor, label_mapping, feat_config = load_models()

NUMERIC_FEATURES     = feat_config['numeric_features']
CATEGORICAL_FEATURES = feat_config['categorical_features']
ALL_FEATURES         = feat_config['all_features']


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk_level(prob_attack: float) -> Tuple[str, str]:
    if prob_attack < 0.50:
        return "Low", "🟢"
    elif prob_attack < 0.80:
        return "Medium", "🟡"
    else:
        return "High", "🛑"


def clean_input_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).replace('-', 'missing')
    for col in NUMERIC_FEATURES:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(
                df_out[col].astype(str).str.strip().replace('-', np.nan),
                errors='coerce'
            )
    return df_out


def predict_batch(input_df: pd.DataFrame, use_nn: bool = False):
    results = []
    if use_nn and nn_model is not None and nn_preprocessor is not None:
        X_transformed = nn_preprocessor.transform(input_df)
        probs         = nn_model.predict(X_transformed, verbose=0).flatten()
        preds         = (probs >= 0.5).astype(int)
    else:
        probs = pipeline.predict_proba(input_df)[:, 1]
        preds = pipeline.predict(input_df)

    for pred, prob in zip(preds, probs):
        risk, emoji = get_risk_level(float(prob))
        results.append({
            'Prediction': label_mapping[int(pred)],
            'Confidence (Attack Prob)': f"{float(prob):.2%}",
            'Risk Level': f"{emoji} {risk}",
            '_prob': float(prob),
        })
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="padding:.5rem 0 1.2rem 0;">
        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.35rem;">
            <span style="font-size:1.6rem;">⛊</span>
            <span style="font-weight:800;font-size:1.2rem;
                         color:#FFFFFF !important;letter-spacing:-.01em;">Vantrex</span>
        </div>
        <div style="font-size:.8rem;color:#D7E7FF !important;
                    padding-left:2.4rem;letter-spacing:.02em;font-weight:500;">
            IoT Cyber Activity Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,.08);margin:.4rem 0 1rem 0;">', unsafe_allow_html=True)

    # Dataset info block
    st.markdown("""
    <div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
                border-radius:10px;padding:.9rem 1rem;margin-bottom:1rem;">
        <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:.08em;color:#8BAAC8 !important;margin-bottom:.7rem;">
            Project Info
        </div>
        <div style="font-size:.83rem;color:#CBD5E1 !important;line-height:1.9;">
            <span style="color:#FFFFFF !important;font-weight:600;">Dataset</span><br/>TON-IoT Network<br/><br/>
            <span style="color:#FFFFFF !important;font-weight:600;">Task</span><br/>Binary Classification<br/><br/>
            <span style="color:#FFFFFF !important;font-weight:600;">Classes</span><br/>Normal / Attack
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    st.markdown("""
    <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.08em;color:#8BAAC8 !important;margin-bottom:.5rem;">
        Model Selection
    </div>
    """, unsafe_allow_html=True)

    use_nn = st.checkbox(
        "Use Neural Network (Keras)",
        value=False,
        disabled=(nn_model is None),
        help="Use the Keras Dense NN instead of the sklearn pipeline"
    )
    if nn_model is None:
        st.markdown('<p style="font-size:.78rem;color:#94A3B8 !important;margin-top:.3rem;">NN not loaded — using sklearn pipeline.</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,.08);margin:1rem 0;">', unsafe_allow_html=True)

    # Risk legend
    st.markdown("""
    <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.08em;color:#8BAAC8 !important;margin-bottom:.7rem;">
        Risk Legend
    </div>
    <div style="font-size:.84rem;color:#CBD5E1 !important;line-height:2.1;">
        🟢 <span style="color:#FFFFFF !important;font-weight:600;">Low</span>
            <span style="color:#94A3B8 !important;"> — Attack prob &lt; 50%</span><br/>
        🟡 <span style="color:#FFFFFF !important;font-weight:600;">Medium</span>
            <span style="color:#94A3B8 !important;"> — 50% ≤ prob &lt; 80%</span><br/>
        🛑 <span style="color:#FFFFFF !important;font-weight:600;">High</span>
            <span style="color:#94A3B8 !important;"> — Attack prob ≥ 80%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,.08);margin:1rem 0 .5rem 0;">', unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:.72rem;color:#94A3B8 !important;text-align:center;">
        Capstone Project
    </p>
    """, unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;">
        <span style="font-size:2rem;">⛊</span>
        <h1 style="font-family:Raleway,sans-serif;font-weight:800;font-size:1.85rem;
                   color:#0F172A;margin:0;letter-spacing:-.02em;">
            Vantrex
        </h1>
    </div>
    <p style="font-family:Raleway,sans-serif;font-size:1.05rem;color:#475569;
              margin:0 0 .9rem 0;max-width:680px;">
        Context-Aware IoT Cyber Activity Detection — analyze network traffic features
        and classify activity as <strong style="color:#10B981;">Normal</strong> or
        <strong style="color:#EF4444;">Attack</strong> in real time.
    </p>
""", unsafe_allow_html=True)




# ── Rule-based NLP chatbot ────────────────────────────────────────────────────
def preprocess_question(text):
    """
    Basic NLP preprocessing:
    - lowercase
    - remove URLs
    - remove punctuation
    - normalize whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


CHATBOT_INTENTS = {
    "attack": {
        "keywords": ["attack", "threat", "malicious", "intrusion", "cyber", "hack", "detected"],
        "response": (
            "An **Attack** prediction means the network traffic was classified as malicious or suspicious. "
            "This could indicate an intrusion attempt, port scanning, data exfiltration, or other cyber threat. "
            "The model assigns a high attack probability to these records."
        ),
    },
    "normal": {
        "keywords": ["normal", "benign", "safe", "legitimate", "clean", "ok"],
        "response": (
            "A **Normal** prediction means the network traffic appears to be legitimate and benign. "
            "The model assigned a low attack probability, suggesting no suspicious activity was detected "
            "in the traffic features provided."
        ),
    },
    "f1_score": {
        "keywords": ["f1", "f1-score", "f1 score", "metric", "imbalanced", "precision", "recall", "measure"],
        "response": (
            "The **F1-score** is the primary evaluation metric for this project because the dataset is imbalanced "
            "(~50K Normal vs ~161K Attack records). F1 is the harmonic mean of precision and recall, making it "
            "more informative than plain accuracy when class sizes are unequal. A higher F1 means the model "
            "correctly identifies both Attack and Normal cases."
        ),
    },
    "random_forest": {
        "keywords": ["random forest", "random", "forest", "rf", "best model", "top model", "sklearn", "ensemble"],
        "response": (
            "**Random Forest** was selected as the best ML model in this project. "
            "It is an ensemble method that builds many decision trees and combines their votes. "
            "It outperformed Logistic Regression and Decision Tree on F1-score because it handles "
            "noisy features well and is robust to overfitting on tabular data like the TON-IoT dataset."
        ),
    },
    "confidence": {
        "keywords": ["confidence", "probability", "prob", "score", "certainty", "percent", "percentage"],
        "response": (
            "**Confidence** (shown as Attack Probability %) represents how sure the model is that a record "
            "is an attack. It ranges from 0% to 100%. A value above 50% triggers an Attack prediction. "
            "Higher values close to 100% indicate the model is very certain the traffic is malicious."
        ),
    },
    "risk_level": {
        "keywords": ["risk", "risk level", "high risk", "medium risk", "low risk", "danger", "severity"],
        "response": (
            "**Risk Level** is derived from the attack probability:\n\n"
            "- 🟢 **Low** — Attack prob < 50% (traffic appears normal)\n"
            "- 🟡 **Medium** — 50% ≤ prob < 80% (suspicious, monitor closely)\n"
            "- 🛑 **High** — Attack prob ≥ 80% (strong threat signal, take action)\n\n"
            "High Risk means the model is highly confident the traffic is an attack."
        ),
    },
    "neural_network": {
        "keywords": ["neural network", "neural", "network", "nn", "keras", "deep learning", "dense", "epoch"],
        "response": (
            "The **Neural Network** in this project is a Keras Dense (fully-connected) model trained "
            "on the TON-IoT dataset. It uses Dropout layers to reduce overfitting and was trained for "
            "10+ epochs. You can switch between the sklearn Random Forest pipeline and the Keras NN "
            "using the sidebar checkbox. Both classify traffic as Normal or Attack."
        ),
    },
    "dataset": {
        "keywords": ["dataset", "data", "ton", "iot", "rows", "columns", "features", "csv", "network data"],
        "response": (
            "This project uses the **TON-IoT Network Dataset** — a benchmark cybersecurity dataset "
            "containing 211K rows and 44 columns of network traffic features. "
            "The target column is `label` (0 = Normal, 1 = Attack). "
            "Features include protocol, connection state, byte counts, packet counts, and more."
        ),
    },
    "how_to_use": {
        "keywords": ["how", "use", "start", "begin", "guide", "tutorial", "input", "upload", "predict", "run"],
        "response": (
            "**How to use Vantrex:**\n\n"
            "1. **Manual Input tab** — Fill in individual network traffic feature values "
            "(protocol, ports, bytes, packets, etc.) and click **Run Prediction**.\n"
            "2. **CSV Batch Upload tab** — Upload a `.csv` file with the required feature columns "
            "to classify many records at once and download results.\n"
            "3. **Sidebar** — Toggle the Neural Network (Keras) checkbox to switch models.\n\n"
            "Results show the prediction label, attack probability, and risk level."
        ),
    },
}


def detect_intent(user_question):
    clean_text = preprocess_question(user_question)

    best_intent = None
    best_score = 0

    for intent, data in CHATBOT_INTENTS.items():
        score = 0
        for keyword in data["keywords"]:
            if keyword in clean_text:
                score += 1

        if score > best_score:
            best_score = score
            best_intent = intent

    if best_intent:
        return CHATBOT_INTENTS[best_intent]["response"]

    return (
        "I am not sure I understood the question. You can ask me about: "
        "Attack, Normal, F1-score, Random Forest, confidence, risk level, "
        "Neural Network, dataset, or how to use the app."
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Manual Input  ", "  CSV Batch Upload  ", "  🤖 AI Assistant  "])


# ─── TAB 1: Manual Input ─────────────────────────────────────────────────────
with tab1:
    section_heading(
        "Single Record Prediction",
        "Fill in the IoT network traffic feature values below and run the classifier."
    )

    # ── Input form card ──
    st.markdown("""
    <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
                padding:1.6rem 1.8rem 1rem 1.8rem;
                box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.07);
                margin-bottom:1.2rem;">
        <div style="font-family:Raleway,sans-serif;font-size:.72rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:.08em;color:#94A3B8;
                    margin-bottom:1rem;">Input Features</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<p style="font-family:Raleway,sans-serif;font-weight:600;font-size:.88rem;color:#374151;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem;">Network Basics</p>', unsafe_allow_html=True)
        proto = st.selectbox(
            "Protocol (proto)",
            ['tcp', 'udp', 'icmp', 'arp', 'ipv6-icmp', 'missing'],
            index=0
        )
        service = st.selectbox(
            "Service (service)",
            ['missing', 'http', 'dns', 'ssl', 'dhcp', 'ssh', 'smtp', 'ftp', 'irc', 'ftp-data'],
            index=0
        )
        conn_state = st.selectbox(
            "Connection State (conn_state)",
            ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'S3', 'OTH', 'RSTOS0', 'missing'],
            index=0
        )
        src_port = st.number_input("Source Port", min_value=0, max_value=65535, value=80)
        dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=443)

    with col2:
        st.markdown('<p style="font-family:Raleway,sans-serif;font-weight:600;font-size:.88rem;color:#374151;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem;">Traffic Volume</p>', unsafe_allow_html=True)
        duration  = st.number_input("Duration (seconds)", min_value=0.0, value=0.5, format="%.6f")
        src_bytes = st.number_input("Source Bytes (src_bytes)", min_value=0, value=512)
        dst_bytes = st.number_input("Destination Bytes (dst_bytes)", min_value=0, value=256)
        missed_bytes = st.number_input("Missed Bytes", min_value=0, value=0)
        src_pkts  = st.number_input("Source Packets (src_pkts)", min_value=0, value=5)
        dst_pkts  = st.number_input("Destination Packets (dst_pkts)", min_value=0, value=3)

    with col3:
        st.markdown('<p style="font-family:Raleway,sans-serif;font-weight:600;font-size:.88rem;color:#374151;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem;">Additional Metrics</p>', unsafe_allow_html=True)
        src_ip_bytes           = st.number_input("Src IP Bytes (src_ip_bytes)", min_value=0, value=560)
        dst_ip_bytes           = st.number_input("Dst IP Bytes (dst_ip_bytes)", min_value=0, value=310)
        http_status_code       = st.number_input("HTTP Status Code (0 if N/A)", min_value=0, max_value=999, value=0)
        http_request_body_len  = st.number_input("HTTP Request Body Len", min_value=0, value=0)
        http_response_body_len = st.number_input("HTTP Response Body Len", min_value=0, value=0)

    # Build input row with all required features
    input_dict = {
        'src_port': src_port,
        'dst_port': dst_port,
        'duration': duration,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'missed_bytes': missed_bytes,
        'src_pkts': src_pkts,
        'src_ip_bytes': src_ip_bytes,
        'dst_pkts': dst_pkts,
        'dst_ip_bytes': dst_ip_bytes,
        'dns_qclass': 0,
        'dns_qtype': 0,
        'dns_rcode': 0,
        'http_request_body_len': http_request_body_len,
        'http_response_body_len': http_response_body_len,
        'http_status_code': http_status_code,
        'http_trans_depth': 0,
        'proto': proto,
        'service': service,
        'conn_state': conn_state,
        'dns_AA': 'missing',
        'dns_RD': 'missing',
        'dns_RA': 'missing',
        'dns_rejected': 'missing',
        'ssl_version': 'missing',
        'ssl_cipher': 'missing',
        'ssl_resumed': 'missing',
        'ssl_established': 'missing',
        'http_method': 'missing',
        'http_version': 'missing',
        'http_orig_mime_types': 'missing',
        'http_resp_mime_types': 'missing',
        'weird_name': 'missing',
        'weird_notice': 'missing',
    }
    input_df = pd.DataFrame([input_dict])[ALL_FEATURES]

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing traffic pattern…"):
            try:
                results    = predict_batch(input_df, use_nn=use_nn)
                r          = results[0]
                pred_label = r['Prediction']
                conf       = r['Confidence (Attack Prob)']
                risk       = r['Risk Level']
                prob       = r['_prob']

                result_card(pred_label, conf, risk, prob)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)


# ─── TAB 2: CSV Batch Upload ──────────────────────────────────────────────────
with tab2:
    section_heading(
        "Batch CSV Prediction",
        f"Upload a CSV file with the {len(ALL_FEATURES)} required feature columns to classify multiple records at once."
    )

    # Custom <details> element — avoids Streamlit's Material Symbols expander
    # arrow, which our Raleway CSS override turns into "arrowdown" text.
    cols_joined = ", ".join(ALL_FEATURES)
    st.markdown(f"""
    <details style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;
                    box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
                    margin-bottom:1rem;overflow:hidden;">
        <summary style="padding:.75rem 1.1rem;cursor:pointer;
                        font-family:'Raleway',sans-serif;font-weight:600;
                        font-size:.92rem;color:#1E293B;">
            View required column list
        </summary>
        <div style="padding:.75rem 1rem 1rem;border-top:1px solid #E2E8F0;">
            <div style="font-family:'Courier New',Courier,monospace;font-size:.79rem;
                        color:#1E293B;background:#F8FAFC;padding:.85rem 1rem;
                        border-radius:6px;line-height:1.8;
                        white-space:pre-wrap;word-break:break-word;">{cols_joined}</div>
        </div>
    </details>
    """, unsafe_allow_html=True)

    # Upload area
    st.markdown("""
    <div style="font-family:Raleway,sans-serif;font-size:.85rem;color:#374151;margin-bottom:.5rem;">
        Accepted format: <strong>.csv</strong> &nbsp;·&nbsp; Encoding: UTF-8
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your CSV file here, or click to browse", type=['csv'])

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)

            # File summary card
            st.markdown(f"""
            <div style="background:#F8FAFF;border:1px solid #C7D7FD;border-radius:10px;
                        padding:.85rem 1.1rem;margin:.8rem 0 1.1rem 0;
                        display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center;">
                <div>
                    <div style="font-family:Raleway,sans-serif;font-size:.7rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:.07em;color:#6B7280;">Rows</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                font-weight:700;color:#1B5FE6;">{upload_df.shape[0]:,}</div>
                </div>
                <div>
                    <div style="font-family:Raleway,sans-serif;font-size:.7rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:.07em;color:#6B7280;">Columns</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                font-weight:700;color:#1B5FE6;">{upload_df.shape[1]}</div>
                </div>
                <div>
                    <div style="font-family:Raleway,sans-serif;font-size:.7rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:.07em;color:#6B7280;">File</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.9rem;
                                font-weight:600;color:#374151;">{uploaded_file.name}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            missing_cols = [c for c in ALL_FEATURES if c not in upload_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                input_batch = upload_df[ALL_FEATURES].copy()
                input_batch = clean_input_df(input_batch)

                st.markdown("**Preview — first 5 rows:**")
                st.dataframe(input_batch.head(5), use_container_width=True)

                st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner(f"Classifying {len(input_batch):,} records…"):
                        results    = predict_batch(input_batch, use_nn=use_nn)
                        results_df = pd.DataFrame(results).drop(columns=['_prob'])
                        results_df.insert(0, 'Row', range(1, len(results_df) + 1))

                        attack_count = (results_df['Prediction'] == 'Attack').sum()
                        normal_count = (results_df['Prediction'] == 'Normal').sum()
                        total        = len(results_df)

                        # Summary metric cards
                        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric("Total Records", f"{total:,}")
                        with mc2:
                            st.metric("Attack Detected", f"{attack_count:,}",
                                      delta=f"{attack_count/total:.1%} of total",
                                      delta_color="inverse")
                        with mc3:
                            st.metric("Normal Traffic", f"{normal_count:,}",
                                      delta=f"{normal_count/total:.1%} of total",
                                      delta_color="normal")

                        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

                        # Status bar
                        atk_pct = int(attack_count / total * 100) if total else 0
                        st.markdown(f"""
                        <div style="margin-bottom:1rem;">
                            <div style="font-family:Raleway,sans-serif;font-size:.78rem;
                                        font-weight:600;color:#374151;margin-bottom:.4rem;">
                                Threat Ratio
                            </div>
                            <div style="background:#E2E8F0;border-radius:99px;height:10px;overflow:hidden;">
                                <div style="width:{atk_pct}%;height:100%;
                                            background:linear-gradient(90deg,#F59E0B,#EF4444);
                                            border-radius:99px;"></div>
                            </div>
                            <div style="display:flex;justify-content:space-between;
                                        font-family:Raleway,sans-serif;font-size:.75rem;
                                        color:#374151;margin-top:.3rem;">
                                <span>🟢 Normal {normal_count:,}</span>
                                <span>🛑 Attack {attack_count:,}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.dataframe(results_df, use_container_width=True)

                        csv_out = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇ Download Results CSV",
                            data=csv_out,
                            file_name='voltguard_predictions.csv',
                            mime='text/csv',
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)


# ─── TAB 3: AI Assistant ─────────────────────────────────────────────────────
with tab3:
    section_heading(
        "🤖 Vantrex AI Assistant",
        "Ask simple questions about the model, prediction results, metrics, risk levels, or how to use the app."
    )

    example_questions = [
        "What is Attack?",
        "What is Normal?",
        "What does F1-score mean?",
        "Why is Random Forest the best model?",
        "What does confidence mean?",
        "What does High Risk mean?",
        "What is the Neural Network?",
        "What dataset is used?",
        "How do I use this app?",
    ]

    selected_example = st.selectbox(
        "Choose an example question (or type your own below):",
        options=["— select an example —"] + example_questions,
    )

    user_input = st.text_input(
        "Your question:",
        value="" if selected_example == "— select an example —" else selected_example,
        placeholder="e.g. What does confidence mean?",
    )

    if st.button("Ask Assistant", type="primary"):
        question = user_input.strip()
        if not question:
            st.warning("Please type a question first.")
        else:
            answer = detect_intent(question)
            st.info(answer)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #E2E8F0;margin-top:2.5rem;padding-top:1.1rem;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">
    <span style="font-family:Raleway,sans-serif;font-size:.78rem;color:#94A3B8;">
        ⛊ Vantrex &nbsp;·&nbsp; Capstone Project &nbsp;·&nbsp; TON-IoT Network Dataset
    </span>
    <span style="font-family:Raleway,sans-serif;font-size:.78rem;color:#94A3B8;">
        scikit-learn (LR · DT · RF) &nbsp;+&nbsp; Keras Dense NN &nbsp;·&nbsp;
    </span>
</div>
""", unsafe_allow_html=True)
