"""TruthLens: a binary fake-news classifier trained on WELFake articles."""
from __future__ import annotations

import html
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.dataset import DATA_PATH, load_dataset as load_welfake, LABELS as DATA_LABELS, stratified_splits
from src.predict import ModelNotTrainedError, load_artifacts, predict_article

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "outputs" / "metrics" / "evaluation.json"
LABELS = ["FAKE", "REAL"]
COLORS = ["#F43F5E", "#34D399"]
SAMPLE_CLAIMS = [
    "The city council approved a new public transit route connecting the north and south districts.",
    "The state budget allocates additional funding to public schools next year.",
    "The governor said unemployment declined during the last quarter.",
]

st.set_page_config(page_title="TruthLens · AI Truth Intelligence", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {--bg:#070A12;--surface:#0D1220;--card:#111827;--raised:#172033;--line:rgba(148,163,184,.15);--cyan:#22D3EE;--blue:#3B82F6;--violet:#8B5CF6;--green:#34D399;--amber:#F59E0B;--rose:#F43F5E;--text:#F1F5F9;--muted:#94A3B8}
html,body,[class*="css"] {font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.stApp {background:radial-gradient(ellipse at 76% -12%,rgba(34,211,238,.07),transparent 34%),radial-gradient(ellipse at 4% 38%,rgba(139,92,246,.045),transparent 30%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(7,10,18,.68)} [data-testid="stMainBlockContainer"]{max-width:1420px;padding-top:1.8rem;padding-bottom:2.6rem}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1424,#08111e);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] > div{padding-top:1.15rem}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:var(--muted)}
section[data-testid="stSidebar"] .stButton{margin:.12rem 0}
section[data-testid="stSidebar"] .stButton button{height:auto;min-height:2.55rem;text-align:left;justify-content:flex-start;border-radius:12px;border:1px solid transparent;background:transparent;color:#CBD5E1;padding:.43rem .78rem;font-weight:650;font-size:.91rem;transition:background .18s,border-color .18s,box-shadow .18s,transform .18s}
section[data-testid="stSidebar"] .stButton button:hover{background:#13243a;border-color:var(--line);color:white}
section[data-testid="stSidebar"] .stButton button[kind="primary"]{background:linear-gradient(90deg,rgba(34,211,238,.15),rgba(59,130,246,.1));border:1px solid rgba(34,211,238,.27);box-shadow:inset 3px 0 #22D3EE,0 0 18px rgba(34,211,238,.07);color:#ECFEFF}
.brand{display:flex;gap:12px;align-items:center;padding:9px 2px 19px;border-bottom:1px solid var(--line);margin-bottom:20px}.lens{display:grid;place-items:center;width:38px;height:38px;border-radius:13px;background:linear-gradient(145deg,#164e63,#1d4ed8);color:#cffafe;font-size:21px;box-shadow:0 0 26px #22d3ee28}.brandname{font-size:18px;font-weight:850;letter-spacing:.11em;color:#f8fafc}.brandtag{font-size:9px;letter-spacing:.19em;color:#94a3b8;margin-top:2px}
.navlabel,.eyebrow{font-size:10px;font-weight:800;letter-spacing:.17em;color:#8294ad;text-transform:uppercase}.navlabel{margin:13px 0 5px;padding-left:7px}
.statusbox,.panel,.feature,.metric,.resultbox,.signal{background:linear-gradient(145deg,rgba(16,29,48,.96),rgba(11,20,36,.96));border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 35px rgba(0,0,0,.14)}
.statusbox{padding:13px;margin-top:12px;background:linear-gradient(145deg,rgba(17,40,49,.8),rgba(13,18,32,.98));border-color:rgba(52,211,153,.17);box-shadow:0 10px 28px rgba(0,0,0,.18),inset 0 1px rgba(255,255,255,.025)}.online{font-size:10px;letter-spacing:.1em;font-weight:800;color:#86efac}.online i{display:inline-block;width:7px;height:7px;border-radius:50%;background:#34d399;margin-right:7px;box-shadow:0 0 5px 2px rgba(52,211,153,.34),0 0 13px rgba(52,211,153,.55)}.statusmodel{margin-top:12px}.statusmodel span{display:block;color:#8294ad;font-size:8px;font-weight:800;letter-spacing:.16em}.statusmodel b{display:block;color:#e2e8f0;font-size:12px;margin-top:3px}.statusline{display:flex;justify-content:space-between;align-items:center;margin-top:7px;color:#94a3b8;font-size:9px;letter-spacing:.08em}.statusline em{font-style:normal;font-size:8px;font-weight:800;letter-spacing:.12em}.statusline .ready{color:#6ee7b7}.statusline .waiting{color:#fbbf24}.smallmuted{font-size:11px;color:var(--muted)}
.hero{position:relative;overflow:hidden;padding:36px 42px;border:1px solid rgba(34,211,238,.17);border-radius:24px;background:radial-gradient(ellipse at 90% 5%,rgba(139,92,246,.12),transparent 38%),linear-gradient(125deg,#111827,#0d1220 65%);margin:0 0 24px}.hero:after{content:'◉';position:absolute;right:6%;top:12%;font-size:150px;color:#22d3ee08}.hero h1{font-size:clamp(42px,5vw,68px);line-height:1.02;letter-spacing:-.055em;margin:16px 0 14px;color:#f1f5f9}.accent{background:linear-gradient(90deg,#22d3ee,#60a5fa,#a78bfa);background-clip:text;-webkit-background-clip:text;color:transparent}.hero p{color:#aab8ca;font-size:16px;max-width:690px;line-height:1.65}.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(34,211,238,.2);border-radius:99px;padding:7px 11px;color:#a5f3fc;background:#22d3ee0b;font-size:10px;font-weight:800;letter-spacing:.1em}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.feature{padding:20px;min-height:156px;height:100%;display:flex;flex-direction:column;position:relative;overflow:hidden;transition:transform .18s,border-color .18s,box-shadow .18s}.feature:before{content:"";position:absolute;left:20px;right:20px;top:0;height:2px;background:linear-gradient(90deg,#22d3ee,#3b82f6,transparent);opacity:.58}.feature:hover{transform:translateY(-3px);border-color:#22d3ee55;box-shadow:0 15px 36px rgba(0,0,0,.24),0 0 22px rgba(34,211,238,.05)}.feature .num{font-size:10px;color:#7e90aa;letter-spacing:.15em}.feature h3{font-size:15px;margin:12px 0 7px;display:flex;align-items:center}.feature p{font-size:12px;color:var(--muted);line-height:1.6;margin:0}.feature .ico{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:9px;background:rgba(34,211,238,.1);font-size:17px;color:var(--cyan);margin-right:9px}
.statusstrip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:15px 0 28px}.stripitem{background:#0d1422;border:1px solid var(--line);border-radius:13px;padding:12px 14px}.stripitem b{display:block;color:#e2e8f0;font-size:13px;margin-top:5px}.stripitem span{font-size:9px;letter-spacing:.1em;color:#8294ad;text-transform:uppercase}
h1{letter-spacing:-.04em!important}h2{letter-spacing:-.03em!important}.subhead{color:#9eacc0;font-size:14px;margin-top:-8px;margin-bottom:22px}.panel{padding:21px;margin-bottom:15px}.panel h3{margin:0 0 5px;font-size:15px}.panelhead{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:15px}.metric{padding:17px 18px;min-height:112px;position:relative;overflow:hidden}.metric:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--tone,#22D3EE)}.metric .label{font-size:9px;font-weight:800;letter-spacing:.13em;color:#91a0b5}.metric .value{font-size:27px;font-weight:780;letter-spacing:-.04em;margin:8px 0 3px;color:#f1f5f9}.metric .hint{font-size:10px;color:#8190a5}.resultbox{padding:24px;background:radial-gradient(circle at 100% 0%,rgba(139,92,246,.1),transparent 38%),linear-gradient(135deg,#111827,#0d1220);border-color:rgba(139,92,246,.24)}.prediction{font-size:35px;text-transform:uppercase;letter-spacing:.04em;font-weight:850;margin:8px 0}.resultmeta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0 16px}.resultmeta>div{padding:11px 13px;border:1px solid rgba(148,163,184,.12);background:rgba(7,10,18,.35);border-radius:12px}.resultmeta span{display:block;color:#94a3b8;font-size:9px;font-weight:800;letter-spacing:.1em}.resultmeta b{display:block;color:#e2e8f0;font-size:13px;margin-top:5px}.bartrack{height:8px;background:#25344a;border-radius:20px;overflow:hidden}.barfill{height:100%;border-radius:20px;background:linear-gradient(90deg,#22d3ee,#3b82f6,#8b5cf6)}.probrow{display:grid;grid-template-columns:120px 1fr 58px;gap:12px;align-items:center;margin:12px 0;color:#cbd5e1;font-size:11px}.probrow .bartrack{height:7px}.probrow.active{color:#fff;font-weight:800}.probrow.active .barfill{background:linear-gradient(90deg,#22d3ee,#8b5cf6)}
.signal{padding:17px;margin-bottom:10px}.signalrow{display:flex;justify-content:space-between;align-items:center;gap:12px;font-size:13px}.support{color:#6ee7b7}.away{color:#c4b5fd}.score{font-variant-numeric:tabular-nums;font-weight:750}.disclaimer{padding:12px 15px;border-radius:12px;background:#f59e0b0b;border:1px solid #f59e0b2b;color:#fcd34d;font-size:11px;line-height:1.55;margin:12px 0}.stepwrap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.step{padding:18px;border-radius:16px;background:#111827;border:1px solid var(--line);min-height:115px}.step .n{font-weight:850;color:#22d3ee;font-size:11px;letter-spacing:.12em}.step b{display:block;margin:10px 0 5px;font-size:13px}.step span{color:#94a3b8;font-size:11px;line-height:1.5}.step .stepico{font-size:18px;color:#a78bfa;margin-right:7px}
.empty-state{max-width:700px;margin:34px auto 12px;padding:42px 34px;text-align:center;border:1px solid rgba(34,211,238,.18);border-radius:21px;background:radial-gradient(ellipse at 50% 0%,rgba(34,211,238,.09),transparent 55%),linear-gradient(145deg,rgba(16,29,48,.98),rgba(11,20,36,.98));box-shadow:0 18px 48px rgba(0,0,0,.22),inset 0 1px rgba(255,255,255,.03)}.empty-lens{display:grid;place-items:center;width:58px;height:58px;margin:0 auto 18px;border-radius:18px;color:#67e8f9;font-size:27px;background:linear-gradient(145deg,rgba(34,211,238,.15),rgba(59,130,246,.14));border:1px solid rgba(34,211,238,.2);box-shadow:0 0 28px rgba(34,211,238,.08)}.empty-state h3{color:#f1f5f9;font-size:14px;letter-spacing:.14em;margin:0 0 10px}.empty-state p{max-width:470px;margin:0 auto;color:#94a3b8;font-size:13px;line-height:1.7}.empty-cta{max-width:280px;margin:0 auto}
.history-row{padding:10px 12px;margin:4px 0 8px;border:1px solid rgba(148,163,184,.11);border-radius:11px;background:rgba(7,10,18,.4)}.history-row b{font-size:11px;color:#c4b5fd;letter-spacing:.07em}.history-row span{font-size:10px;color:#94a3b8;margin-left:10px}.history-row p{font-size:11px;color:#cbd5e1;margin:5px 0 0;line-height:1.5}
.pipeline-list{display:grid;grid-template-columns:1fr;gap:6px;margin:12px 0 24px}.pipeline-step{display:grid;grid-template-columns:46px minmax(150px,205px) 1fr;align-items:center;gap:15px;min-height:68px;padding:10px 16px;border:1px solid rgba(148,163,184,.13);border-radius:14px;background:linear-gradient(100deg,rgba(17,24,39,.96),rgba(13,18,32,.88));position:relative}.pipeline-step:not(:last-child):after{content:"";position:absolute;left:38px;bottom:-8px;height:9px;border-left:1px solid rgba(34,211,238,.36);z-index:1}.pipeline-marker{display:grid;place-items:center;width:36px;height:36px;border-radius:11px;color:#67e8f9;background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.17);font-size:16px}.pipeline-title{color:#e2e8f0;font-size:12px;font-weight:800;letter-spacing:.04em}.pipeline-desc{color:#94a3b8;font-size:11px;line-height:1.5}
.stTextArea textarea{background:#090e19!important;border:1px solid #263244!important;border-radius:14px!important;color:#f1f5f9!important;line-height:1.6!important}.stTextArea textarea:focus{border-color:#22d3ee!important;box-shadow:0 0 0 2px #22d3ee20!important}.stButton>button[kind="primary"]{background:linear-gradient(105deg,#0891b2,#0369a1);border:0;border-radius:12px;color:white;font-weight:800;box-shadow:0 6px 18px rgba(8,145,178,.2);transition:filter .18s,transform .18s,box-shadow .18s}.stButton>button[kind="primary"]:hover{filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 9px 22px rgba(8,145,178,.28);border:0}.stButton>button[kind="secondary"]{border:1px solid var(--line);border-radius:12px;background:#111827;color:#dbeafe;transition:background .18s,border-color .18s,transform .18s}.stButton>button[kind="secondary"]:hover{background:#172033;border-color:rgba(34,211,238,.3);transform:translateY(-1px)}
[data-testid="stMetric"]{background:#101d30;border:1px solid var(--line);border-radius:15px;padding:14px}[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:11px}[data-testid="stMetricValue"]{color:#f8fafc!important}
div[data-testid="stTabs"] button{color:#9eacc0}div[data-testid="stTabs"] button[aria-selected="true"]{color:#a5f3fc;border-bottom-color:#22d3ee}
@media(max-width:900px){.statusstrip{grid-template-columns:repeat(2,1fr)}.grid3{grid-template-columns:1fr}.feature{min-height:138px}.stepwrap{grid-template-columns:repeat(2,1fr)}.hero{padding:26px}.probrow{grid-template-columns:90px 1fr 48px}.resultmeta{grid-template-columns:1fr}.panel{padding:17px}.pipeline-step{grid-template-columns:42px 1fr;gap:8px 12px}.pipeline-desc{grid-column:2}.pipeline-step:not(:last-child):after{left:33px}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data
def load_dataset():
    return load_welfake(DATA_PATH)


@st.cache_resource
def get_artifacts() -> dict:
    return load_artifacts()


METRICS = load_metrics()
TEST = METRICS.get("test_metrics", {})
VALIDATION = METRICS.get("validation_metrics", {})
MODEL_NAME = METRICS.get("model_name", "Saved classifier")
ARTIFACT_PATHS = {
    "model": ROOT / "models" / "fake_news_model.joblib",
    "vectorizer": ROOT / "models" / "fake_news_vectorizer.joblib",
    "metadata": ROOT / "models" / "fake_news_metadata.json",
}
ARTIFACT_FILES_PRESENT = all(path.is_file() for path in ARTIFACT_PATHS.values())
try:
    LOADED_ARTIFACTS = get_artifacts() if ARTIFACT_FILES_PRESENT else None
except Exception:
    LOADED_ARTIFACTS = None
LOADED_DATASET = DATA_PATH.is_file()
MODEL_READY = LOADED_ARTIFACTS is not None
VECTORIZER_READY = MODEL_READY and "vectorizer" in LOADED_ARTIFACTS
DATASET_READY = LOADED_DATASET is not None
PIPELINE_READY = MODEL_READY and bool(METRICS) and DATASET_READY

PAGES = {
    "Analyze": ("✦", "Check a claim", "WORKSPACE"),
    "Explainability": ("⌁", "Why the model decided", "WORKSPACE"),
    "Model Lab": ("◫", "Performance & metrics", "WORKSPACE"),
    "Dataset": ("▤", "WELFake articles", "DATA & SCIENCE"),
    "Methodology": ("⌘", "How TruthLens works", "DATA & SCIENCE"),
}

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "claim" not in st.session_state:
    st.session_state.claim = ""
if "article_title" not in st.session_state:
    st.session_state.article_title = ""
if "history" not in st.session_state:
    st.session_state.history = []


def fmt_pct(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def go(page: str) -> None:
    st.session_state.page = page


def set_sample(text: str) -> None:
    st.session_state.article_title = ""
    st.session_state.claim = text
    st.session_state.pop("last_result", None)


def set_random_sample() -> None:
    set_sample(random.choice(SAMPLE_CLAIMS))


def clear_claim() -> None:
    st.session_state.article_title = ""
    st.session_state.claim = ""
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_record", None)


def clear_current_result() -> None:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_record", None)


def clear_history() -> None:
    st.session_state.history = []


def open_history_item(index: int) -> None:
    item = st.session_state.history[index]
    st.session_state.article_title = ""
    st.session_state.claim = item["claim"]
    st.session_state.last_result = item["result"]


with st.sidebar:
    st.markdown('<div class="brand"><div class="lens">◉</div><div><div class="brandname">TRUTH<span style="color:#22d3ee">LENS</span></div><div class="brandtag">AI TRUTH INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="navlabel">OVERVIEW</div>', unsafe_allow_html=True)
    st.button("⌂   Dashboard", key="nav_home", type="primary" if st.session_state.page == "Home" else "secondary", width="stretch", on_click=go, args=("Home",))
    for group in ("WORKSPACE", "DATA & SCIENCE"):
        st.markdown(f'<div class="navlabel">{group}</div>', unsafe_allow_html=True)
        for page, (icon, subtitle, section) in PAGES.items():
            if section == group:
                st.button(f"{icon}   {page}   ·   {subtitle}", key=f"nav_{page}", type="primary" if st.session_state.page == page else "secondary", width="stretch", on_click=go, args=(page,))
    state_dot = "#34d399" if PIPELINE_READY else "#f59e0b"
    st.markdown(f'<div class="statusbox"><div class="online" style="color:{state_dot}"><i style="background:{state_dot};box-shadow:0 0 5px 2px {state_dot}55,0 0 13px {state_dot}88"></i>{"SYSTEM OPERATIONAL" if PIPELINE_READY else "SETUP REQUIRED"}</div><div class="statusmodel"><span>MODEL</span><b>{html.escape(MODEL_NAME if MODEL_READY else "Artifacts unavailable")}</b></div><div class="statusmodel"><span>DATASET</span><b>{"WELFake articles" if DATASET_READY else "Files unavailable"}</b></div><div class="statusline"><span>MODEL</span><em class="{"ready" if MODEL_READY else "waiting"}">{"READY" if MODEL_READY else "WAITING"}</em></div><div class="statusline"><span>VECTORIZER</span><em class="{"ready" if VECTORIZER_READY else "waiting"}">{"READY" if VECTORIZER_READY else "WAITING"}</em></div><div class="statusline"><span>DATASET</span><em class="{"ready" if DATASET_READY else "waiting"}">{"READY" if DATASET_READY else "WAITING"}</em></div><div class="statusline"><span>PIPELINE</span><em class="{"ready" if PIPELINE_READY else "waiting"}">{"READY" if PIPELINE_READY else "WAITING"}</em></div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div><div class="smallmuted">TruthLens v1.0<br>NLP · ML · Explainable AI</div>', unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown('<div class="disclaimer">The displayed probabilities are Logistic Regression model confidence based on patterns learned from WELFake. They are not independent fact verification.</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, hint: str = "", tone: str = "#22D3EE") -> None:
    st.markdown(f'<div class="metric" style="--tone:{tone}"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div><div class="hint">{html.escape(hint)}</div></div>', unsafe_allow_html=True)


def probability_rows(probabilities: dict, selected: str) -> None:
    for index, label in enumerate(LABELS):
        value = float(probabilities.get(label, 0))
        active = label == selected
        st.markdown(f'<div class="probrow {"active" if active else ""}"><span>{label.upper()}</span><div class="bartrack"><div class="barfill" style="width:{max(0,min(100,value*100)):.2f}%;background:{COLORS[index] if not active else "linear-gradient(90deg,#22d3ee,#8b5cf6)"}"></div></div><span style="text-align:right">{value:.1%}</span></div>', unsafe_allow_html=True)


def render_explanation(result: dict) -> None:
    exp = result.get("explanation") or {}
    if not exp.get("available"):
        st.info("Local feature contributions are unavailable for this model.")
        return
    features = exp.get("features", [])
    supporting = sorted((f for f in features if f.get("direction") == "supports"), key=lambda f: abs(f.get("contribution", 0)), reverse=True)
    away = sorted((f for f in features if f.get("direction") != "supports"), key=lambda f: abs(f.get("contribution", 0)), reverse=True)
    cols = st.columns(2)
    for col, title, values, cls, sign in ((cols[0], "SUPPORTING SIGNALS", supporting, "support", "+"), (cols[1], "FEATURES PUSHING AWAY", away, "away", "−")):
        with col:
            st.markdown(f'<div class="panel"><div class="eyebrow">{title}</div><div class="smallmuted" style="margin:8px 0 15px">Relative to {html.escape(str(exp.get("competitor") or "other classes"))}</div>', unsafe_allow_html=True)
            if not values:
                st.markdown('<div class="smallmuted">No strong terms in this direction for this claim.</div>', unsafe_allow_html=True)
            for item in values:
                st.markdown(f'<div class="signal"><div class="signalrow"><span class="{cls}">{sign} &nbsp;{html.escape(str(item["feature"]))}</span><span class="score {cls}">{item["contribution"]:+.4f}</span></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def plot_confusion(matrix, labels, title: str):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#101D30")
    ax.set_facecolor("#101D30")
    sns.heatmap(matrix, annot=True, fmt="d", cmap=sns.color_palette(["#0B1424", "#155e75", "#22D3EE"], as_cmap=True), cbar=True, linewidths=.7, linecolor="#26364c", xticklabels=labels, yticklabels=labels, annot_kws={"color": "#F8FAFC", "fontsize": 10}, ax=ax)
    ax.set_title(title, color="#F8FAFC", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted label", color="#94A3B8", labelpad=10)
    ax.set_ylabel("True label", color="#94A3B8", labelpad=10)
    ax.tick_params(colors="#CBD5E1", labelsize=9)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    return fig


def make_report(record: dict) -> dict:
    result = record["result"]
    return {
        "title": "TruthLens Analysis",
        "claim": record["claim"],
        "prediction": result["label"],
        "model_confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "model": result["model_name"],
        "timestamp": record["timestamp"],
        "explanation": result.get("explanation"),
        "disclaimer": "TruthLens predicts based on learned textual patterns. It does not independently verify facts against the real world.",
    }


def render_analysis_history() -> None:
    history = st.session_state.history
    if not history:
        return
    st.markdown('<div class="panel"><div class="panelhead"><div><div class="eyebrow">SESSION ONLY</div><h3 style="margin-top:7px">Recent analyses</h3></div></div>', unsafe_allow_html=True)
    for index, item in enumerate(history[:5]):
        result = item["result"]
        preview = " ".join(item["claim"].split())
        if len(preview) > 92:
            preview = preview[:89] + "…"
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f'<div class="history-row"><b>{html.escape(result["label"].upper())}</b><span>{result["confidence"]:.2%} model confidence</span><p>{html.escape(preview)}</p></div>', unsafe_allow_html=True)
        with right:
            st.button("Open", key=f"history_open_{index}", width="stretch", on_click=open_history_item, args=(index,))
    st.button("Clear history", key="clear_history", type="secondary", on_click=clear_history)
    st.markdown('</div>', unsafe_allow_html=True)


page = st.session_state.page
if page == "Home":
    st.markdown(f'<div class="hero"><span class="pill"><span style="color:#34d399">●</span> AI RESEARCH INTERFACE &nbsp;·&nbsp; {"MODEL READY" if MODEL_READY else "ARTIFACTS REQUIRED"}</span><h1>TRUTHLENS<br><span class="accent">See the signal behind the story.</span></h1><p>Analyze language patterns, compare model predictions, and understand what influenced the classifier.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid3"><div class="feature"><span class="num">01 / INSPECT</span><h3><span class="ico">✦</span>ANALYZE</h3><p>Paste a claim and inspect the model’s prediction.</p></div><div class="feature"><span class="num">02 / INTERPRET</span><h3><span class="ico" style="color:#a78bfa;background:#8b5cf61a">⌁</span>UNDERSTAND</h3><p>See which language patterns influenced the result.</p></div><div class="feature"><span class="num">03 / EXPLORE</span><h3><span class="ico" style="color:#60a5fa;background:#3b82f61a">◫</span>EXPLORE</h3><p>Compare models, metrics, dataset and methodology.</p></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="statusstrip"><div class="stripitem"><span>Model</span><b>{html.escape(MODEL_NAME if MODEL_READY else "Unavailable")}</b></div><div class="stripitem"><span>Dataset</span><b>{"WELFake · ready" if DATASET_READY else "Dataset file missing"}</b></div><div class="stripitem"><span>Pipeline</span><b>{"TF-IDF · ready" if PIPELINE_READY else "Setup required"}</b></div><div class="stripitem"><span>Test status</span><b>{"Held-out · "+fmt_pct(TEST.get("accuracy")) if TEST else "Metrics unavailable"}</b></div></div>', unsafe_allow_html=True)
    c1,c2=st.columns([1,1]);
    with c1:
        st.markdown('<div class="panel"><div class="eyebrow">START HERE</div><h3 style="margin-top:10px">Analyze a news article</h3><p class="smallmuted">Enter a headline and article text to see the binary prediction and both class probabilities.</p></div>',unsafe_allow_html=True)
        st.button("✦  Analyze a claim", type="primary", on_click=go, args=("Analyze",), key="home_analyze")
    with c2:
        st.markdown('<div class="panel"><div class="eyebrow">RESEARCH NOTE</div><h3 style="margin-top:10px">A benchmark model, with limits</h3><p class="smallmuted">Binary test metrics are shown directly from saved evaluation results. Text classification cannot independently verify a claim.</p></div>',unsafe_allow_html=True)
    disclaimer()

elif page == "Analyze":
    eyebrow("ANALYZE THE SIGNAL")
    st.title("Analyze a News Article")
    st.markdown('<div class="subhead">Enter a headline and article text to inspect the binary classifier output.</div>',unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">ARTICLE INPUT</div>',unsafe_allow_html=True)
        uploaded=st.file_uploader("Upload a UTF-8 text file (optional)", type=["txt"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                uploaded_text=uploaded.getvalue().decode("utf-8-sig")[:100000]
                if st.session_state.get("uploaded_name") != uploaded.name:
                    st.session_state.claim=uploaded_text
                    st.session_state.uploaded_name=uploaded.name
                    clear_current_result()
            except UnicodeDecodeError: st.error("The text file could not be decoded as UTF-8. Paste the claim or choose a UTF-8 .txt file.")
        article_title=st.text_input("Headline (optional)",placeholder="Paste the news headline here…",key="article_title",on_change=clear_current_result)
        claim_text=st.text_area("Article text", placeholder="Paste the article body here…", height=260, max_chars=100000, key="claim", help="Up to 100,000 characters. The classifier requires at least three meaningful words.",on_change=clear_current_result)
        count_cols=st.columns([1,1,2])
        count_cols[0].caption(f"{len(claim_text.split())} words")
        count_cols[1].caption(f"{len(claim_text)} / 100,000 characters")
        count_cols[2].caption("Headline and article text stay in this session and are not saved to a database.")
        sample_rows=[st.columns(2),st.columns(2)]
        sample_actions=[
            ("Sample claim 01","sample_claim_0",set_sample,(SAMPLE_CLAIMS[0],)),
            ("Sample claim 02","sample_claim_1",set_sample,(SAMPLE_CLAIMS[1],)),
            ("Sample claim 03","sample_claim_2",set_sample,(SAMPLE_CLAIMS[2],)),
            ("Random claim","sample_claim_random",set_random_sample,()),
        ]
        for idx,(label,key,callback,args) in enumerate(sample_actions):
            with sample_rows[idx//2][idx%2]:
                st.button(label,key=key,width="stretch",on_click=callback,args=args)
        st.caption("Demo prompts are illustrative. Their predictions are model outputs, not verified facts.")
        action_cols=st.columns([4,1])
        with action_cols[0]:
            analyze=st.button("✦  Analyze with TruthLens  →",type="primary",width="stretch",key="analyze_claim")
        with action_cols[1]:
            st.button("Clear",key="clear_claim",type="secondary",width="stretch",on_click=clear_claim)
    if analyze:
        clear_current_result()
        if not article_title.strip() and not claim_text.strip():
            st.info("Enter an article headline or text, or choose a sample to begin.")
        elif not MODEL_READY:
            st.error("The saved model files are not available. Follow the setup steps in the README, then restart TruthLens.")
        else:
            try:
                with st.spinner("Analyzing language patterns…"):
                    result=predict_article(article_title,claim_text,get_artifacts())
                claim_display="\n\n".join(part.strip() for part in (article_title,claim_text) if part.strip())
                record={"claim":claim_display,"result":result,"timestamp":datetime.now().astimezone().isoformat(timespec="seconds")}
                st.session_state.last_result=result
                st.session_state.last_record=record
                st.session_state.history.insert(0,record)
                st.session_state.history=st.session_state.history[:10]
            except ValueError as exc:
                st.warning(str(exc))
            except ModelNotTrainedError:
                st.error("A required model artifact is missing. Follow the README setup steps and try again.")
            except Exception:
                st.error("TruthLens could not complete this analysis. Check the model setup and try again.")
    result=st.session_state.get("last_result")
    if result:
        record=st.session_state.get("last_record",{"claim":st.session_state.claim,"result":result,"timestamp":"Unknown"})
        st.markdown(f'<div class="resultbox"><div class="eyebrow">TRUTHLENS VERDICT</div><div class="prediction" style="color:#c4b5fd">{html.escape(result["label"])}</div><div class="resultmeta"><div><span>FAKE PROBABILITY · MODEL CONFIDENCE</span><b>{result["fake_probability"]:.1%}</b></div><div><span>REAL PROBABILITY · MODEL CONFIDENCE</span><b>{result["real_probability"]:.1%}</b></div><div><span>MODEL</span><b>{html.escape(result["model_name"])}</b></div></div><div class="smallmuted" style="margin-top:8px">These are model confidence scores from Logistic Regression, not independent fact verification.</div></div>',unsafe_allow_html=True)
        left,right=st.columns([1.05,.95])
        with left:
            st.markdown('<div class="panel"><div class="eyebrow">BINARY MODEL OUTPUT</div><h3 style="margin-top:9px">Class probabilities</h3>',unsafe_allow_html=True)
            probability_rows(result["probabilities"],result["label"])
            st.markdown('</div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel"><div class="eyebrow">EXPLAINABILITY</div><h3 style="margin-top:9px">Language features</h3><p class="smallmuted">Local contributions compare the predicted class with its strongest score competitor.</p></div>',unsafe_allow_html=True)
            render_explanation(result)
        disclaimer()
        report=make_report(record)
        report_text=("TruthLens Analysis\n"
            f"Claim: {report['claim']}\n"
            f"Prediction: {report['prediction']}\n"
            f"Predicted class confidence: {report['prediction']}: {report['model_confidence']:.1%}\n"
            f"Binary probabilities: {json.dumps(report['probabilities'],ensure_ascii=False)}\n"
            f"Model: {report['model']}\nTimestamp: {report['timestamp']}\n"
            f"Explanation: {json.dumps(report['explanation'],ensure_ascii=False)}\n"
            f"Disclaimer: {report['disclaimer']}\n")
        with st.expander("Copy or download this result"):
            st.caption("Use the copy control on this plain-text block, or download a TXT or JSON report.")
            st.markdown("**Copy Result**")
            st.code(report_text,language=None)
            dl_txt,dl_json=st.columns(2)
            with dl_txt:
                st.download_button("Download TXT",data=report_text,file_name="truthlens-analysis.txt",mime="text/plain",width="stretch",key="download_report_txt")
            with dl_json:
                st.download_button("Download JSON",data=json.dumps(report,ensure_ascii=False,indent=2),file_name="truthlens-analysis.json",mime="application/json",width="stretch",key="download_report_json")
    render_analysis_history()

elif page == "Explainability":
    eyebrow("MODEL INTERPRETATION"); st.title("Why did the model think this?")
    st.markdown('<div class="subhead">Language features that influenced the classifier’s decision for your latest analyzed claim.</div>',unsafe_allow_html=True)
    result=st.session_state.get("last_result")
    if result:
        render_explanation(result)
        st.markdown('<div class="disclaimer">These language features influence the classifier’s decision. They do not independently prove whether a claim is factually true.</div>',unsafe_allow_html=True)
        disclaimer()
    else:
        st.markdown('<div class="empty-state"><div class="empty-lens">⌕</div><h3>NO CLAIM ANALYZED</h3><p>Run an analysis to see which language patterns influenced the classifier’s prediction.</p></div>',unsafe_allow_html=True)
        _,cta_col,_=st.columns([1,1,1])
        with cta_col:
            st.button("✦  Analyze a claim →",type="secondary",width="stretch",on_click=go,args=("Analyze",),key="explain_empty_analyze")

elif page == "Model Lab":
    head,cta=st.columns([5,1.5],vertical_alignment="center")
    with head:
        eyebrow("MODEL MONITORING"); st.title("Model Lab")
        st.markdown('<div class="subhead">Validation and held-out test performance for the WELFake binary news classifier.</div>',unsafe_allow_html=True)
    with cta:
        st.button("✦  Analyze an article →",key="model_lab_analyze",type="secondary",on_click=go,args=("Analyze",),width="content")
    if not TEST:
        st.warning("Evaluation metrics are unavailable. Run python src/train.py to train the WELFake model.")
    else:
        cols=st.columns(4)
        for col,label,key,tone in zip(cols,["TEST ACCURACY","MACRO F1","MACRO PRECISION","MACRO RECALL"],["accuracy","f1_macro","precision_macro","recall_macro"],["#22D3EE","#8B5CF6","#3B82F6","#22C55E"]):
            with col: metric_card(label,fmt_pct(TEST.get(key)),"WELFake held-out test split",tone)
        validation_tab,test_tab=st.tabs(["Validation", "Held-out test"])
        with validation_tab:
            st.markdown('<div class="panel"><div class="eyebrow">TRAINING DESIGN</div><h3 style="margin-top:7px">Single binary Logistic Regression model</h3><p class="smallmuted">TF-IDF and Logistic Regression were fitted on the stratified training split only. Validation data was used for evaluation and the test split was held out until final reporting.</p></div>',unsafe_allow_html=True)
            if VALIDATION:
                metrics=VALIDATION
                cols=st.columns(4)
                for col,label,key in zip(cols,["VALIDATION ACCURACY","MACRO F1","MACRO PRECISION","MACRO RECALL"],["accuracy","f1_macro","precision_macro","recall_macro"]):
                    with col: metric_card(label,fmt_pct(metrics.get(key)),"WELFake validation split")
        with test_tab:
            left,right=st.columns([1.2,.8])
            with left:
                st.markdown('<div class="panel"><div class="eyebrow">CONFUSION MATRIX</div><h3 style="margin-top:6px">WELFake test split</h3>',unsafe_allow_html=True)
                fig=plot_confusion(TEST.get("confusion_matrix",[]),LABELS,"WELFake binary confusion matrix")
                st.pyplot(fig,width="stretch");plt.close(fig);st.markdown('</div>',unsafe_allow_html=True)
            with right:
                st.markdown('<div class="panel"><div class="eyebrow">PER-CLASS METRICS</div><h3 style="margin-top:6px">Classification report</h3>',unsafe_allow_html=True)
                report=TEST.get("classification_report",{})
                df=pd.DataFrame([{ "Class":label,"Precision":fmt_pct(values.get("precision")),"Recall":fmt_pct(values.get("recall")),"F1":fmt_pct(values.get("f1-score")),"Support":int(values.get("support",0))} for label,values in report.items() if label in LABELS])
                st.dataframe(df,hide_index=True,width="stretch");st.markdown('</div>',unsafe_allow_html=True)
        disclaimer()

elif page == "Dataset":
    eyebrow("DATA EXPLORER"); st.title("WELFake Articles")
    st.markdown('<div class="subhead">Binary article corpus used to train the production fake-news classifier.</div>',unsafe_allow_html=True)
    try:
        frame,summary=load_dataset()
        splits=stratified_splits(frame)
        dims={name:len(part) for name,part in splits.items()}
        avg=float(frame["article"].map(lambda value: len(str(value).split())).mean())
        cols=st.columns(4)
        for col,(label,value,tone) in zip(cols,[("TOTAL ARTICLES",f'{summary["total_samples"]:,}',"#22D3EE"),("FAKE",f'{summary["class_distribution"]["FAKE"]:,}',"#F43F5E"),("REAL",f'{summary["class_distribution"]["REAL"]:,}',"#34D399"),("AVG. ARTICLE LENGTH",f"{avg:.0f} words","#8B5CF6")]):
            with col: metric_card(label,value,"After empty and duplicate removal",tone)
        st.markdown(f'<div class="panel"><div class="eyebrow">STRATIFIED SPLIT SIZES</div><p class="smallmuted">Train {dims["train"]:,} · Validation {dims["validation"]:,} · Test {dims["test"]:,}. Exact duplicate articles are removed before splitting.</p><p class="smallmuted">WELFake label mapping: 0 → FAKE, 1 → REAL. Subject/date metadata is excluded from model features.</p></div>',unsafe_allow_html=True)
        chart,data=st.columns([1,1])
        with chart:
            st.markdown('<div class="panel"><div class="eyebrow">CLASS DISTRIBUTION</div><h3 style="margin-top:6px">Binary class totals</h3>',unsafe_allow_html=True)
            counts=summary["class_distribution"]
            fig,ax=plt.subplots(figsize=(7,4),facecolor="#101D30");ax.set_facecolor("#101D30")
            bars=ax.bar(LABELS,[counts[label] for label in LABELS],color=COLORS,width=.62)
            ax.set_ylabel("Articles",color="#94A3B8");ax.tick_params(colors="#CBD5E1")
            ax.bar_label(bars,fmt="%d",padding=3,color="#CBD5E1",fontsize=9)
            for spine in ax.spines.values():spine.set_visible(False)
            fig.tight_layout();st.pyplot(fig,width="stretch");plt.close(fig);st.markdown('</div>',unsafe_allow_html=True)
        with data:
            st.markdown('<div class="panel"><div class="eyebrow">CLEANING SUMMARY</div><h3 style="margin-top:6px">Leakage prevention</h3>',unsafe_allow_html=True)
            st.write({"source rows":summary["source_rows"],"empty articles removed":summary["empty_articles_removed"],"duplicate articles removed":summary["duplicate_articles_removed"],"conflicting duplicate rows removed":summary["conflicting_duplicate_rows_removed"]})
            st.markdown('</div>',unsafe_allow_html=True)
        selected=st.selectbox("Split",list(splits),format_func=lambda value:value.title())
        st.dataframe(splits[selected][["label","title","text"]].head(25),hide_index=True,width="stretch")
        st.info("WELFake is a labeled benchmark, not a live fact-checking source. Its model confidence reflects learned dataset patterns.")
    except (FileNotFoundError,ValueError) as exc:
        st.warning(str(exc))
    disclaimer()

elif page == "Methodology":
    eyebrow("RESEARCH PIPELINE"); st.title("How TruthLens works")
    st.markdown('<div class="subhead">A full-article binary classification pipeline trained on WELFake.</div>',unsafe_allow_html=True)
    steps=[
        ("◉","SOURCE ARTICLES","WELFake title, text, and binary label are loaded; subject/date are not used."),
        ("⌁","LABELS","The published WELFake label encoding maps 0 to FAKE and 1 to REAL."),
        ("Aa","TEXT CLEANING","Title and body are joined, lowercased, and cleaned of URLs, markup, and excess whitespace."),
        ("▦","DUPLICATE CONTROL","Empty articles are removed; exact duplicate content and label-conflicting duplicate groups are removed before splitting."),
        ("⚙","STRATIFIED SPLIT","A fixed seed creates 80% training, 10% validation, and 10% held-out test data."),
        ("▤","TF-IDF","The vectorizer is fit only on training article text; validation and test are transform-only."),
        ("◫","LOGISTIC REGRESSION","The binary Logistic Regression model is fit only on the training matrix."),
        ("✓","EVALUATION","Validation and test metrics include accuracy, macro precision/recall/F1, confusion matrix, and class report."),
        ("✦","INFERENCE","The app loads the saved WELFake vectorizer/model and displays both predict_proba outputs."),
        ("⌕","EXPLAINABILITY","Feature contributions use this binary model's TF-IDF values and Logistic Regression coefficients."),
    ]
    st.markdown('<div class="panel"><div class="eyebrow">END-TO-END METHOD</div><h3 style="margin-top:7px">From article text to model output</h3><div class="pipeline-list">'+''.join(f'<div class="pipeline-step"><div class="pipeline-marker">{icon}</div><div class="pipeline-title">{title}</div><div class="pipeline-desc">{desc}</div></div>' for icon,title,desc in steps)+'</div></div>',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="panel"><div class="eyebrow">PRODUCTION MODEL</div><h3 style="margin-top:8px">WELFake · Logistic Regression</h3><p class="smallmuted">The production classifier uses the WELFake binary article labels. The historical LIAR model is not loaded by inference.</p></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="eyebrow">MODEL CONFIDENCE</div><h3 style="margin-top:8px">Not independent verification</h3><p class="smallmuted">Probabilities come directly from Logistic Regression predict_proba(). They measure model confidence based on the training corpus and do not verify claims against evidence.</p></div>',unsafe_allow_html=True)
    disclaimer()
