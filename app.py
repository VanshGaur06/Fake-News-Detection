"""TruthLens: Single-page Fake News Detection Dashboard."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.dataset import DATA_DIR, CLASSES
from src.predict import load_artifacts, predict_claim

ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "outputs" / "metrics" / "evaluation.json"

st.set_page_config(
    page_title="TruthLens · Fake News Detector",
    page_icon="🔎",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 4px;
        color: #F8FAFC;
    }
    .sub-title {
        font-size: 15px;
        color: #94A3B8;
        margin-bottom: 24px;
    }
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
    }
    .verdict-box {
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin: 16px 0;
    }
    .verdict-fake {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
    }
    .verdict-real {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22C55E;
    }
    .badge-fake {
        background-color: #EF4444;
        color: white;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
    }
    .badge-real {
        background-color: #22C55E;
        color: white;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Curated samples from train dataset that predict with high confidence (>70%) and match ground truth
SAMPLE_DATASET_CLAIMS = [
    {
        "label": "🔴 Fake Sample 1 (False)",
        "speaker": "Dwayne Bohac",
        "ground_truth": "FAKE (PolitiFact: False)",
        "claim": "Says the Annies List political group supports third-trimester abortions on demand.",
    },
    {
        "label": "🔴 Fake Sample 2 (Pants-on-Fire)",
        "speaker": "Social Media Blog",
        "ground_truth": "FAKE (PolitiFact: Pants on Fire)",
        "claim": "Say Oregon Reps. Peter DeFazio and Earl Blumenauer are socialists who are openly serving in the U.S. Congress.",
    },
    {
        "label": "🟢 Real Sample 1 (Poverty Data)",
        "speaker": "Bernie Sanders",
        "ground_truth": "REAL (PolitiFact: Half-True / Nuanced)",
        "claim": "Since 2000, nearly 12 million Americans have slipped out of the middle class and into poverty.",
    },
    {
        "label": "🟢 Real Sample 2 (Economic Report)",
        "speaker": "Economic Analysis",
        "ground_truth": "REAL (PolitiFact: Mostly-True)",
        "claim": "The economy bled $24 billion due to the government shutdown.",
    },
]


@st.cache_resource
def get_artifacts():
    try:
        return load_artifacts()
    except Exception:
        return None


@st.cache_data
def get_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


artifacts = get_artifacts()
metrics = get_metrics()

# App Header
st.markdown('<div class="main-title">🔎 TruthLens: Fake News Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Natural Language Processing & Machine Learning for Claim Verification on the LIAR Dataset</div>', unsafe_allow_html=True)

# SECTION 1: Claim Verification Input & Samples
st.write("### 1. Test a Statement")
st.caption("Click a verified sample from the training dataset or type your own claim:")

# Clickable Sample Buttons
cols = st.columns(len(SAMPLE_DATASET_CLAIMS))
if "user_text" not in st.session_state:
    st.session_state["user_text"] = SAMPLE_DATASET_CLAIMS[0]["claim"]
if "selected_info" not in st.session_state:
    st.session_state["selected_info"] = f"Speaker: {SAMPLE_DATASET_CLAIMS[0]['speaker']} | Ground Truth: {SAMPLE_DATASET_CLAIMS[0]['ground_truth']}"

for i, sample in enumerate(SAMPLE_DATASET_CLAIMS):
    if cols[i].button(sample["label"], key=f"sample_{i}", use_container_width=True):
        st.session_state["user_text"] = sample["claim"]
        st.session_state["selected_info"] = f"Speaker: {sample['speaker']} | Ground Truth: {sample['ground_truth']}"

st.info(f"ℹ️ **Selected Sample Info:** {st.session_state['selected_info']}")

user_statement = st.text_area(
    "Statement Text",
    value=st.session_state["user_text"],
    height=90,
    placeholder="Enter a claim or news statement here...",
)

col_run, col_clear = st.columns([1, 5])
verify_clicked = col_run.button("Verify Claim", type="primary", use_container_width=True)

if verify_clicked or "last_prediction" not in st.session_state:
    if not user_statement.strip():
        st.warning("Please enter a statement to test.")
    elif not artifacts:
        st.error("Model artifacts not found. Run 'python src/train.py' in the terminal first.")
    else:
        try:
            st.session_state["last_prediction"] = predict_claim(user_statement, artifacts)
        except Exception as e:
            st.error(f"Error: {e}")

# SECTION 2: Prediction Result & Word Explanation
if "last_prediction" in st.session_state and st.session_state["last_prediction"]:
    pred = st.session_state["last_prediction"]
    is_fake = pred["prediction"] == "FAKE"
    v_class = "verdict-fake" if is_fake else "verdict-real"
    b_class = "badge-fake" if is_fake else "badge-real"

    st.markdown(f"""
    <div class="verdict-box {v_class}">
        <span class="{b_class}">{pred['prediction']}</span>
        <h3 style="margin-top: 10px; margin-bottom: 4px;">Confidence: {pred['confidence']:.1%}</h3>
        <p style="color: #94A3B8; margin-bottom: 0;">Probabilities: FAKE = {pred['fake_probability']:.1%} | REAL = {pred['real_probability']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Word contribution explanation graph
    st.write("#### Word-Level Influence on Prediction")
    st.caption("How specific words pushed the score (Green = Supports REAL, Red = Supports FAKE):")

    exp_words = pred.get("explanation", {}).get("words", [])
    if exp_words:
        df_exp = pd.DataFrame(exp_words)
        fig_exp, ax_exp = plt.subplots(figsize=(9, 3.2))
        fig_exp.patch.set_facecolor("#0F172A")
        ax_exp.set_facecolor("#1E293B")
        colors = ["#EF4444" if impact == "FAKE" else "#22C55E" for impact in df_exp["impact"]]

        bars = ax_exp.barh(df_exp["word"], df_exp["weight"], color=colors, height=0.55)
        ax_exp.axvline(0, color="#64748B", linestyle="--", linewidth=0.8)
        ax_exp.set_xlabel("Feature Contribution (TF-IDF × Coefficient)", color="#CBD5E1", fontsize=10)
        ax_exp.tick_params(colors="#CBD5E1", labelsize=9)
        ax_exp.invert_yaxis()
        for spine in ax_exp.spines.values():
            spine.set_color("#334155")

        st.pyplot(fig_exp)
        plt.close(fig_exp)
    else:
        st.caption("No strong vocabulary signals detected for this statement.")

st.markdown("---")

# SECTION 3: Model Performance & Graphs
st.write("### 2. Model Performance & Evaluation Metrics")
test_metrics = metrics.get("test_metrics", {})

if test_metrics:
    # 4 Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-box'><div class='metric-val'>{test_metrics.get('accuracy', 0):.1%}</div><div class='metric-lbl'>Test Accuracy</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><div class='metric-val'>{test_metrics.get('f1_macro', 0):.1%}</div><div class='metric-lbl'>Macro F1-Score</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><div class='metric-val'>{test_metrics.get('precision_macro', 0):.1%}</div><div class='metric-lbl'>Macro Precision</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-box'><div class='metric-val'>{test_metrics.get('recall_macro', 0):.1%}</div><div class='metric-lbl'>Macro Recall</div></div>", unsafe_allow_html=True)

    st.write("")
    graph_col1, graph_col2 = st.columns(2)

    # Graph 1: Confusion Matrix
    with graph_col1:
        st.write("#### Confusion Matrix (Held-out Test Split)")
        cm = np.array(test_metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
        fig_cm.patch.set_facecolor("#0F172A")
        ax_cm.set_facecolor("#1E293B")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cbar=False,
            annot_kws={"size": 13, "weight": "bold", "color": "white"},
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted Label", color="#CBD5E1", fontsize=10)
        ax_cm.set_ylabel("True Label", color="#CBD5E1", fontsize=10)
        ax_cm.tick_params(colors="#CBD5E1")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Graph 2: Model Comparison Bar Chart
    with graph_col2:
        st.write("#### Candidate Model Comparison (Validation Split)")
        comp = metrics.get("model_comparison", {})
        if comp:
            models = list(comp.keys())
            accs = [comp[m]["accuracy"] * 100 for m in models]
            f1s = [comp[m]["f1_macro"] * 100 for m in models]

            x = np.arange(len(models))
            width = 0.35

            fig_comp, ax_comp = plt.subplots(figsize=(5.5, 3.8))
            fig_comp.patch.set_facecolor("#0F172A")
            ax_comp.set_facecolor("#1E293B")

            rects1 = ax_comp.bar(x - width/2, accs, width, label="Accuracy (%)", color="#38BDF8")
            rects2 = ax_comp.bar(x + width/2, f1s, width, label="Macro F1 (%)", color="#A855F7")

            ax_comp.set_ylabel("Score (%)", color="#CBD5E1", fontsize=10)
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(models, color="#CBD5E1", fontsize=9, rotation=12)
            ax_comp.tick_params(colors="#CBD5E1")
            ax_comp.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)
            ax_comp.set_ylim(40, 75)
            for spine in ax_comp.spines.values():
                spine.set_color("#334155")

            st.pyplot(fig_comp)
            plt.close(fig_comp)
        else:
            st.caption("No comparison data available.")

st.markdown("---")
st.caption("TruthLens Fake News Detection · AIML Mini Project · Trained on PolitiFact LIAR Dataset")
